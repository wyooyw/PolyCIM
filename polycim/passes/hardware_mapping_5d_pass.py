import json
import math
import os
import time
from collections import OrderedDict
from itertools import combinations, permutations
from typing import Optional

import islpy as isl
import numpy as np
from tqdm import tqdm

import polycim.utils.utils as utils
from polycim.config import CIMConfig
from polycim.depth_first.timeout import timeout
from polycim.op.base_operator import BasicOperator
from polycim.passes.base import DepthFirstPass, Schedule, SchedulePassResult
from polycim.passes.loop_padding import loop_padding_dim


def get_cim_operator(n_rows, n_cols):
    cim_operator = BasicOperator(
        domain=isl.BasicSet(f"{{ [i,j]: 0 <= i < {n_rows} and 0 <= j < {n_cols} }}"),
        access_I=isl.BasicMap("{ [i,j] -> I[i] }"),
        access_O=isl.BasicMap("{ [i,j] -> O[j] }"),
        access_W=isl.BasicMap("{ [i,j] -> W[i,j] }"),
    )
    return cim_operator

def get_access_bitmap(op):
    """
        Order: [output, Input, Weight]
    """
    I_index_set = utils.get_dominate_iters_of_pw_multi_aff(op.access_I.as_pw_multi_aff())
    W_index_set = utils.get_dominate_iters_of_pw_multi_aff(op.access_W.as_pw_multi_aff())
    O_index_set = utils.get_dominate_iters_of_pw_multi_aff(op.access_O.as_pw_multi_aff())

    bitmap = utils.convert_index_set_to_bitmap(O_index_set, I_index_set, W_index_set)
    bitmap = OrderedDict(sorted(bitmap.items(), key=lambda x: int(x[0][1:])))
    return bitmap

def map_software_index_to_hardware_index(software_access_bitmap, hardware_access_bitmap):
    def rename_access_bitmap(access_bitmap, char):
        new_bitmap = OrderedDict()
        for key, value in access_bitmap.items():
            new_bitmap[char+key[1:]] = tuple(value)
        return new_bitmap
    software_access_bitmap = rename_access_bitmap(software_access_bitmap, "s")
    hardware_access_bitmap = rename_access_bitmap(hardware_access_bitmap, "h")
    mapping = OrderedDict()
    for skey,svalue in software_access_bitmap.items():
        for hkey,hvalue in hardware_access_bitmap.items():
            if svalue == hvalue:
                _mapping = mapping.get(svalue, [set(),set()])
                _mapping[0].add(skey)
                _mapping[1].add(hkey)
                mapping[svalue] = _mapping
    # ipdb.set_trace()
    h2s_mapping = OrderedDict()
    for software_indexs, hardware_indexs in mapping.values():
        assert len(hardware_indexs) == 1
        h2s_mapping[hardware_indexs.pop()] = sorted(software_indexs, key=lambda x: int(x[1:]))

    for hardware_index in hardware_access_bitmap.keys():
        if hardware_index not in h2s_mapping:
            # TODO: support no software index mapping to hardware index
            return None
            h2s_mapping[hardware_index] = {1}

    return h2s_mapping

def generate_combinations(elements):
    all_combinations = []
    n = len(elements)
    for r in range(1, n+1):
        if int(os.environ.get("NEW_ALGO", 0))==1:
            combinations_r = permutations(elements, r)
        else:
            combinations_r = combinations(elements, r)
        
        all_combinations.extend(combinations_r)
    return all_combinations

def sort_by_name(obj):
    return sorted(obj, key=lambda x: int(x[1:]))

def generate_combinations_exclude(elements, exclude):
    elements = sort_by_name(set(elements) - set(exclude))
    return generate_combinations(elements)

def get_all_software_to_hardware_index_mapping(h2s_mapping):
    """
    Example:
        h2s_mapping: {'h0': {'s1', 's3'}, 'h1': {'s5', 's4'}}
        return: [
            {'h0': {'s1'}: 'h1':{'s4'} },
            {'h0': {'s1'}: 'h1':{'s5'} },
            {'h0': {'s1'}: 'h1':{'s4, s5'} },
            {'h0': {'s3'}: 'h1':{'s4'} },
            {'h0': {'s3'}: 'h1':{'s5'} }
            {'h0': {'s3'}: 'h1':{'s4, s5'} },
            {'h0': {'s1', 's3'}: 'h1':{'s4'} },
            {'h0': {'s1', 's3'}: 'h1':{'s5'} },
            {'h0': {'s1', 's3'}: 'h1':{'s4, s5'} },
        ]
    """
    hardware_indexs = list(h2s_mapping.keys())
    visited_s_axis = set()
    current_mapping = dict()
    all_mapping = []
    def dfs(step, visited_s_axis):
        if step == len(hardware_indexs):
            all_mapping.append(current_mapping.copy())
            return
        h_axis = hardware_indexs[step]
        
        for s_axis_combine in generate_combinations_exclude(h2s_mapping[h_axis], visited_s_axis):
            
            # ipdb.set_trace()
            visited_s_axis = visited_s_axis.union(set(s_axis_combine))
            current_mapping[h_axis] = s_axis_combine
            dfs(step+1, visited_s_axis)
            current_mapping[h_axis] = None
            visited_s_axis = visited_s_axis - set(s_axis_combine)
    dfs(0, visited_s_axis)
    return all_mapping

def get_coalescing_schedule_from_mapping(mapping, software_op):
    """
    mapping:  {'h0': ('s1',), 'h1': ('s4', 's5')}
    scheudle: { [s0, s1, s2, s3, s4, s5] -> [s0, s2, s3, s1, 3s4 + s5] }
    """
    new_domain = utils.rename_all_dims_for_basic_set(software_op.domain,'s')
    domain_iter_names = new_domain.get_var_names(isl.dim_type.set)
    # print(mapping)
    # shape = utils.get_box_hull_shape(new_domain)
    bounds = {iter_name: 
                (int(str(new_domain.dim_min_val(i))),int(str(new_domain.dim_max_val(i))))
                     for i,iter_name in enumerate(domain_iter_names)}
    schedule_range = list()
    # import pdb; pdb.set_trace()
    # Put the axis not in mapping to the front
    s_axis_in_mapping = set([item for value in mapping.values() for item in value])
    s_axis_not_in_mapping = set(bounds.keys()) - s_axis_in_mapping
    s_axis_not_in_mapping = sort_by_name(s_axis_not_in_mapping)
    schedule_range.extend(s_axis_not_in_mapping)

    # If the axis in mapping is not the last axis, put it to the back
    # s_axis_in_mapping = set([value[-1] for value in mapping.values()])
    # s_axis_not_in_mapping = set(bounds.keys()) - s_axis_in_mapping
    # s_axis_not_in_mapping = sort_by_name(s_axis_not_in_mapping)
    # schedule_range.extend(s_axis_not_in_mapping)
    
    # Put the axis in mapping to the back
    def merge(axis_list, bounds):
        for i, axis in enumerate(axis_list):
            if i==0:
                result = axis_list[0]
            else:
                last_axis_upperbound = int(math.ceil(bounds[axis_list[i]][1]+1))
                result = f"({result}) * {last_axis_upperbound} + {axis}"
        return result

    h_axis_sorted = sort_by_name(mapping.keys())
    for h_axis in h_axis_sorted:
        s_axis = mapping[h_axis]
        expression = merge(s_axis, bounds)
        schedule_range.append(expression)
    # print(schedule_range)
    # Make the schedule
    s_axis_sorted = sort_by_name(bounds.keys())
    schedule = isl.BasicMap("{ [%s] -> [%s] }" % (",".join(s_axis_sorted), ",".join(schedule_range)))

    return schedule

def get_reverse_coalescing_schedule_from_mapping(mapping, software_op):
    """
    mapping:  {'h0': ('s4', 's5')}
    scheudle: { [s0, s1, s2, s3, s4, s5] -> [s0, s1, s2, s3, 3s4 + s5] }
    reverse_scheudle: { [s0, s1, s2, s3, s45] -> [s0, s1, s2, s3, s4, s5] : s4 = floor(s45/3) and s5 = s45%3 }
    """
    new_domain = utils.rename_all_dims_for_basic_set(software_op.domain,'s')
    domain_iter_names = new_domain.get_var_names(isl.dim_type.set)
    # print(mapping)
    # shape = utils.get_box_hull_shape(new_domain)
    bounds = {iter_name: 
                (int(str(new_domain.dim_min_val(i))),int(str(new_domain.dim_max_val(i))))
                     for i,iter_name in enumerate(domain_iter_names)}
    factors = {iter_name: ub+1 for iter_name, (lb, ub) in bounds.items()}

    schedule_domain = []
    s_axis_in_mapping = set([item for value in mapping.values() for item in value])
    s_axis_not_in_mapping = set(bounds.keys()) - s_axis_in_mapping
    s_axis_not_in_mapping = sort_by_name(s_axis_not_in_mapping)
    schedule_domain.extend(s_axis_not_in_mapping)

    h_axis_sorted = sort_by_name(mapping.keys())
    schedule_domain.extend(h_axis_sorted)

    schedule_range = sort_by_name(domain_iter_names)

    constraints = []
    for h_axis in h_axis_sorted:
        s_axis_list = mapping[h_axis]
        mod_factor = factors[s_axis_list[0]]
        div_factor = 1
        for s_axis in s_axis_list[1:]:
            mod_factor *= factors[s_axis]
            div_factor *= factors[s_axis]
        for i,s_axis in enumerate(s_axis_list):
            constraints.append(f"{s_axis} = floor(({h_axis}%{mod_factor})/{div_factor})")
            mod_factor //= factors[s_axis]
            if i < len(s_axis_list) - 1:
                div_factor //= factors[s_axis_list[i+1]]

    schedule = isl.BasicMap("{ [%s] -> [%s] : %s }" % (",".join(schedule_domain), ",".join(schedule_range), " and ".join(constraints)))
    return schedule


def _get_hardware_tiling_schedule(n_software_dim, tiling_factor):
    assert type(tiling_factor)==list
    assert len(tiling_factor) <= n_software_dim
    n_hardware_dim = len(tiling_factor)
    schedule_range = list()
    for i in range(n_software_dim- n_hardware_dim):
        schedule_range.append(f"s{i}")
    for i in range(n_software_dim- n_hardware_dim, n_software_dim):
        schedule_range.append(f"floor(s{i}/{tiling_factor[i-(n_software_dim- n_hardware_dim)]})")
    for i in range(n_software_dim- n_hardware_dim, n_software_dim):
        schedule_range.append(f"s{i}%{tiling_factor[i-(n_software_dim- n_hardware_dim)]}")
    schedule_domain = list()
    for i in range(n_software_dim):
        schedule_domain.append(f"s{i}")
    schedule = isl.BasicMap("{ [%s] -> [%s] }" % (",".join(schedule_domain), ",".join(schedule_range)))
    return schedule

def get_hardware_tiling_schedule(software_schedule, tiling_factor):
    n_software_dim = software_schedule.range().as_set().n_dim()
    hardware_tiling_schedule = _get_hardware_tiling_schedule(n_software_dim, tiling_factor)
    return hardware_tiling_schedule

def hardware_tiling(all_schedule, tiling_factors):
    new_schedules = []
    for schedule in all_schedule:
        hardware_tiling_schedule = get_hardware_tiling_schedule(schedule, tiling_factors)
        # schedule = schedule.apply_range(hardware_tiling_schedule)
        new_schedules.append((schedule, hardware_tiling_schedule))
    return new_schedules

compute_time_lowerbound_cache = dict()
def clear_compute_time_lowerbound_cache():
    global compute_time_lowerbound_cache
    compute_time_lowerbound_cache = dict()

def filter_all_mapping_by_compute_times(domain, all_mapping, min_compute_times):
    global compute_time_lowerbound_cache
    domain = utils.rename_all_dims_for_basic_set(domain, 's')
    all_iters = domain.get_var_names(isl.dim_type.set)
    new_all_mapping = []
    for mapping in all_mapping:
        mapping_iters = set()
        for value in mapping.values():
            mapping_iters |= set(value)
        keep_iters = set(all_iters) - mapping_iters
        keep_iters =list(keep_iters)
        keep_iters = sorted(keep_iters)
        cache_key = ",".join(keep_iters)
        if cache_key in compute_time_lowerbound_cache:
            compute_time_lowerbound = compute_time_lowerbound_cache[cache_key]
        else:
            projected_domain = domain.project_out_except(names=list(keep_iters),types=[isl.dim_type.set])
            compute_time_lowerbound = projected_domain.count_val()
            compute_time_lowerbound_cache[cache_key] = compute_time_lowerbound
        # import pdb; pdb.set_trace()
        if compute_time_lowerbound > min_compute_times:
            continue

        new_all_mapping.append(mapping)
    return new_all_mapping


def hardware_merge_tiling(op, macro_row, macro_col, min_compute_times):
    clear_compute_time_lowerbound_cache()
    # domain_name_to_size = utils.get_static_box_shape(op.domain)

    n_rows = macro_row
    n_cols = macro_col
    cim_op = get_cim_operator(n_rows, n_cols)
    software_access_bitmap = get_access_bitmap(op)
    hardware_access_bitmap = get_access_bitmap(cim_op)
    
    mapping = map_software_index_to_hardware_index(software_access_bitmap, hardware_access_bitmap)
    if mapping is None: 
        return None
    all_mapping = get_all_software_to_hardware_index_mapping(mapping)
    all_mapping = filter_all_mapping_by_compute_times(op.domain, all_mapping, min_compute_times)
    all_schedules = [get_coalescing_schedule_from_mapping(mapping, op) for mapping in all_mapping]
    for schedule in all_schedules:
        assert schedule.intersect_domain(op.domain).reverse().is_single_valued(), f"{schedule} should not be single valued!"
    
    all_schedules = hardware_tiling(all_schedules, [n_rows, n_cols])
    # for schedule in all_schedules:
    #     assert schedule.intersect_domain(op.domain).reverse().is_single_valued(), f"{schedule} should be single valued!"
    # exit()
    return all_schedules

@timeout(seconds=10)
def count_val(domain):
    return int(str(domain.count_val()))
    
def filter_op_by_execution_time_pass(op_list, macro_row, macro_col):
    begin_time = time.time()

    total_flops = int(str(op_list[0].domain.count_val()))
    min_compute_times_limit = math.ceil(total_flops / ( macro_row * macro_col))
    
    if len(op_list) == 0:
        print(f"[filter_op_by_execution_time_pass]: \n    0 inputs. skip.")
        return op_list

    exe_time_list = []

    # filter op with n_div < 5
    op_list = [op for op in op_list if op.domain.dim(isl.dim_type.div) < 5]
    min_exe_time = 999999999
    for idx,op in enumerate(tqdm(op_list, desc="filter op by outer execute time")):
        n_dim = op.domain.dim(isl.dim_type.set)
        outer_domain = op.domain.project_out(isl.dim_type.set, n_dim - 2, 2)
        # exe_time = int(str(outer_domain.count_val()))
        exe_time = count_val(outer_domain)
        if exe_time is None:
            exe_time = 999999999
        exe_time_list.append(exe_time)
        if exe_time < min_exe_time:
            min_exe_time = exe_time
            print("Current min compute time: ", min_exe_time)
        if exe_time <= min_compute_times_limit:
            break

    exe_time_list = np.array(exe_time_list)
    sorted_indices = np.argsort(exe_time_list)

    new_op_list = []
    new_op_list_execution_time = []
    min_value = exe_time_list[sorted_indices[0]]
    num_ops = len(op_list)
    for i,index in enumerate(sorted_indices):
        if i < 5 or exe_time_list[index] == min_value:
            new_op_list.append(op_list[index])
            new_op_list_execution_time.append(exe_time_list[index])
    new_op_list_execution_time = np.array(new_op_list_execution_time)

    end_time = time.time()

    print(f"""
[filter_op_by_execution_time_pass]:
    {len(op_list)} ops input.
        Execution time: 
            max={exe_time_list.max()}, 
            min={exe_time_list.min()}, 
            mean={int(exe_time_list.mean())} .
    {len(new_op_list)} ops output.
        Execution time: 
            max={new_op_list_execution_time.max()}, 
            min={new_op_list_execution_time.min()}, 
            mean={int(new_op_list_execution_time.mean())},
            all={new_op_list_execution_time} .
        Min Average Use Cell: {total_flops / new_op_list_execution_time.min():2}

        Pass time: {end_time - begin_time:.2f}s
""")
    return new_op_list, new_op_list_execution_time

def fast_count_val(op, merge_schedule ,tiling_schedule):
    domain = utils.rename_dims(op.domain, isl.dim_type.set, "s")

    dominate_iters_per_dim = utils.get_dominate_iters_of_pw_multi_aff_per_out(merge_schedule.as_pw_multi_aff())
    dominate_merge_iters = set()
    for dominate_iters in dominate_iters_per_dim:
        if len(dominate_iters) >= 2:
            dominate_merge_iters |= set(dominate_iters)

    domain_names = domain.get_var_names(isl.dim_type.set)
    name_to_pos = {name:pos for pos,name in enumerate(domain_names)}

    new_domain = domain
    for iter_name in dominate_merge_iters:
        iter_pos = name_to_pos[iter_name]
        dim_max_val = domain.dim_max_val(iter_pos).get_num_si()
        dim_min_val = domain.dim_min_val(iter_pos).get_num_si()
        new_domain = new_domain.drop_constraints_involving_dims(isl.dim_type.set, iter_pos, 1)
        new_domain = new_domain.lower_bound_val(isl.dim_type.set, iter_pos, isl.Val(dim_min_val))
        new_domain = new_domain.upper_bound_val(isl.dim_type.set, iter_pos, isl.Val(dim_max_val))
    
    domain_after_schedule = domain.apply(merge_schedule)
    domain_after_tiling = domain_after_schedule.apply(tiling_schedule)

    n_dim = domain_after_tiling.dim(isl.dim_type.set)
    outer_domain = domain_after_tiling.project_out(isl.dim_type.set, n_dim - 2, 2)

    count_val = domain_after_tiling.count_val()
    return count_val

def fast_count_val_v2(op, merge_schedule, macro_row, macro_col):
    domain = utils.rename_dims(op.domain, isl.dim_type.set, "s")

    dominate_iters_per_dim = utils.get_dominate_iters_of_pw_multi_aff_per_out(merge_schedule.as_pw_multi_aff())

    domain_names = domain.get_var_names(isl.dim_type.set)
    name_to_size = {iter_name:domain.dim_max_val(iter_pos).get_num_si()+1 for iter_pos,iter_name in enumerate(domain_names)}

    size_0 = 1
    for dominate_iter in dominate_iters_per_dim[-2]:
        size_0 *= name_to_size[dominate_iter]

    size_1 = 1
    for dominate_iter in dominate_iters_per_dim[-1]:
        size_1 *= name_to_size[dominate_iter]

    # import pdb; pdb.set_trace()

    size_middle_0 = math.ceil(size_0 / macro_row)
    size_middle_1 = math.ceil(size_1 / macro_col)
    size_middle_ub = size_middle_0 * size_middle_1

    merge_iters = set(dominate_iters_per_dim[-2]) | set(dominate_iters_per_dim[-1])
    keep_iters = list(set(domain_names) - set(merge_iters))
    top_domain = domain.project_out_except(keep_iters, [isl.dim_type.set])
    top_count = top_domain.count_val()

    count_val_ub = size_middle_ub * top_count

    
    return count_val_ub

def hardware_merge_tiling_pass(op_list, macro_row, macro_col):
    new_op_list = []
    schedule_fail_op_cnt = 0
    time_schedule = 0
    time_apply = 0

    time_list = []
    dim_size_list = []
    min_compute_times = int(str(op_list[0].domain.count_val()))
    min_compute_times_limit = math.ceil(min_compute_times /(macro_row * macro_col) )
    early_stop = False
    for op_idx,op in enumerate(tqdm(op_list)):
        
        assert op.access_I.is_single_valued(), f"{op.access_I} should be single valued!"
        assert op.access_W.is_single_valued(), f"{op.access_W} should be single valued!"
        assert op.access_O.is_single_valued(), f"{op.access_O} should be single valued!"
        
        begin_time = time.time()
        schedules = hardware_merge_tiling(op, macro_row, macro_col, min_compute_times = min_compute_times)
        time_schedule += (time.time() - begin_time)
        
        if schedules is None:
            schedule_fail_op_cnt += 1
            continue
        # print("-------------------------------------")

        begin_time = time.time()
        for idx,(merge_schedule, tile_schedule) in enumerate(schedules):
            
            new_op_after_merge = op.apply_schedule(merge_schedule, skip_simplify=True)
            new_op_after_tiling = new_op_after_merge.apply_schedule(tile_schedule, skip_simplify=True)
            new_op = new_op_after_tiling
            new_op_list.append(new_op)

            if len(new_op_list) % 8 == 0:
                n_dim = new_op.domain.dim(isl.dim_type.set)
                outer_domain = new_op.domain.project_out(isl.dim_type.set, n_dim - 2, 2)
                exe_time = count_val(outer_domain)
                if exe_time is not None:
                    min_compute_times = min(min_compute_times, exe_time)
                    if min_compute_times <= min_compute_times_limit:
                        early_stop = True
                        break

            # print(f"{min_compute_times=}")
            
        time_apply += (time.time() - begin_time)

        if early_stop:
            break

    print(f"[hardware_merge_tiling_pass]: \n    {len(op_list)} ops input.\n    {schedule_fail_op_cnt} op schedule fail.\n    {len(new_op_list)} ops output.")
    print(f"    Schedule time: {time_schedule:.2f}s\n    Apply time: {time_apply:.2f}s\n ")
    return new_op_list




def get_mapping_from_op(op):
    software_access_bitmap = get_access_bitmap(op)
    # [output, Input, Weight]
    hardware_access_bitmap = OrderedDict(
        [("h0", [0, 1, 1]), ("h1", [1, 0, 1]), ("h2", [1, 1, 0])]
    )

    mapping = map_software_index_to_hardware_index(
        software_access_bitmap, hardware_access_bitmap
    )

    return mapping

class HardwareMapping5DSchedule(Schedule):
    def __init__(self, s2h_mapping, tiling_schedule, reorder_schedule):
        super().__init__()
        self.s2h_mapping = s2h_mapping
        self.tiling_schedule = str(tiling_schedule)
        self.reorder_schedule = str(reorder_schedule)
    
    def dumps(self):
        return json.dumps({
            "s2h_mapping":self.s2h_mapping,
            "tiling_schedule":self.tiling_schedule,
            "reorder_schedule":self.reorder_schedule
        })
    
    def parse(self, data):
        assert isinstance(data, dict)
        self.s2h_mapping = data["s2h_mapping"]
        self.tiling_schedule = data["tiling_schedule"]
        self.reorder_schedule = data["reorder_schedule"]

def get_macro_5d_hardware_tiling_schedule(op, software_schedule, cim_cfg):
    """
    OrderedDict([('h0', [0, 1, 1]), ('h1', [1, 0, 1]), ('h2', [1, 1, 0])])

    macros = Buffer(<N_ROW, N_COMP, N_GROUP, N_GROUP_VCOL>, int8, __MACRO__)
    """
    n_software_dim = software_schedule.range().as_set().n_dim()
    n_hardware_dim = 3

    new_op = op.apply_schedule(software_schedule, skip_simplify=True)
    new_domain = new_op.domain
    sizes = utils.get_box_hull_shape(new_domain)
    print(f"{sizes=}")
    n_h0 = sizes[-3]
    n_igroup = min(math.ceil(math.ceil(n_h0 / cim_cfg.n_comp) / cim_cfg.n_row), cim_cfg.n_group)
    n_ogroup = cim_cfg.n_group // n_igroup

    use_group = n_ogroup * n_igroup

    keep_iters = []
    for i in range(n_software_dim - n_hardware_dim):
        keep_iters.append(f"s{i}")

    # tiling
    # h0: share output
    # h1: share input
    # h2: share weight
    n_reduce = cim_cfg.n_comp * cim_cfg.n_row * n_igroup
    tile_domain_iters = keep_iters + ["h0", "h1", "h2"]
    tile_range_iters = keep_iters + [
        f"floor(floor(h0/{cim_cfg.n_comp}) / {cim_cfg.n_row}) % {n_igroup}", # igroup
        f"floor(h0/{cim_cfg.n_comp}) % {cim_cfg.n_row}", # row
        f"h0 % {cim_cfg.n_comp}", # comp
        f"h1 % {cim_cfg.n_group_vcol}", # col
        f"floor(h2 / {n_ogroup})", # time1
        f"floor(h1 / {cim_cfg.n_group_vcol})", # time2
        f"floor(h0 / {n_reduce})", # time3
        f"h2 % {n_ogroup}", # ogroup
    ]
    tile_domain_iters_def = ", ".join(tile_domain_iters)
    tile_range_iters_def = ", ".join(tile_range_iters)
    tile_def = f"{{ [{tile_domain_iters_def}] -> [{tile_range_iters_def}] }}"
    print(f"{tile_def=}")
    tile_schedule = isl.BasicMap(tile_def)

    # merge inner-group and outer-group
    # merge_domain_iters = keep_iters + ["igroup","row","comp","col","time","ogroup"]
    # merge_range_iters = keep_iters + ["time", "row", "comp", f"ogroup * {n_igroup} + igroup", "col"]
    # merge_domain_iters_def = ", ".join(merge_domain_iters)
    # merge_range_iters_def = ", ".join(merge_range_iters)
    # merge_group_def = f"{{ [{merge_domain_iters_def}] -> [{merge_range_iters_def}] }}"
    # merge_group_schedule = isl.BasicMap(merge_group_def)

    # reorder dims
    reorder_domain_iters = keep_iters + [
        "igroup",
        "row",
        "comp",
        "col",
        "time1",
        "time2",
        "time3",
        "ogroup",
    ]
    reorder_range_iters = keep_iters + [
        "time1",
        "time2",
        "time3",
        "row",
        "comp",
        "igroup",
        "ogroup",
        "col",
    ]
    reorder_domain_iters_def = ", ".join(reorder_domain_iters)
    reorder_range_iters_def = ", ".join(reorder_range_iters)
    reorder_def = f"{{ [{reorder_domain_iters_def}] -> [{reorder_range_iters_def}] }}"
    reorder_schedule = isl.BasicMap(reorder_def)

    return tile_schedule, reorder_schedule, use_group


class HardwareMapping5DPass(DepthFirstPass):
    def __init__(self, 
            args,
            cim_config: CIMConfig,
            fix_schedule: Optional[HardwareMapping5DSchedule]=None, 
            schedule_as_key: bool=False,
        ):
        super().__init__(
            fix_schedule=fix_schedule, 
            schedule_as_key=schedule_as_key
        )
        self.args = args
        self.cim_config = cim_config
        assert self.fix_schedule is None or isinstance(self.fix_schedule, HardwareMapping5DSchedule)

    def apply(self, operator):
        mapping = get_mapping_from_op(operator)
        operator.history_schedules.append({"s2h_mapping":mapping})
        coalescing_schedule = get_coalescing_schedule_from_mapping(mapping, operator)

        tile_schedule, reorder_schedule, use_group = get_macro_5d_hardware_tiling_schedule(operator, coalescing_schedule, self.cim_config)
        print(f"Origin shape: {utils.get_box_hull_shape(operator.domain)}")
        new_op = operator.apply_schedule(coalescing_schedule, 
            skip_simplify=True, 
            name="coalescing"
        )
        print(f"After coalescing shape: {utils.get_box_hull_shape(new_op.domain)}")
        new_op = new_op.apply_schedule(tile_schedule, 
            skip_simplify=True, 
            name="tiling"
        )
        print(f"After tiling shape: {utils.get_box_hull_shape(new_op.domain)}")
        new_op = new_op.apply_schedule(reorder_schedule, 
            skip_simplify=True, 
            name="reorder"
        )
        print(f"After reorder shape: {utils.get_box_hull_shape(new_op.domain)}")

        n_dim = new_op.domain.dim(isl.dim_type.set)
        new_op = loop_padding_dim(new_op, n_dim-1, self.cim_config.n_group_vcol)
        new_op = loop_padding_dim(new_op, n_dim-4, self.cim_config.n_comp)
        # TODO: padding group dimension
        print(f"After padding shape: {utils.get_box_hull_shape(new_op.domain)}")
        new_op.set_attr("n_macro_iters", 5)
        new_op.set_attr("n_use_group", use_group)
        new_op.set_attr("n_use_comp", self.cim_config.n_comp)

        schedule = HardwareMapping5DSchedule(mapping, tile_schedule, reorder_schedule)
        result = SchedulePassResult(new_op, schedule)
        return [result]

if __name__=="__main__":
    operator = BasicOperator(
        domain = isl.BasicSet(
            "{ [oc,oh,ow,kh,kw]: 0 <= oc < 64 and 0<=oh<64 and 0<=ow<64 and 0<=kh<3 and 0<=kw<3 }"
        ),
        access_I = isl.BasicMap("{ [oc,oh,ow,kh,kw] -> I[oh + kh, ow + kw] }"),
        access_O = isl.BasicMap("{ [oc,oh,ow,kh,kw] -> O[oc, oh, ow] }"),
        access_W = isl.BasicMap("{ [oc,oh,ow,kh,kw] -> W[oc, kh, kw] }"),
    )

    new_ops = hardware_merge_tiling_pass([operator])
    # filter_op_by_execution_time_pass(new_ops)
    for i,op in enumerate(new_ops):
        print(f"{i}")
        print(f"  {op.domain.as_set()=}\n")
        print(f"  {op.access_I=}\n")
        print(f"  {op.access_O=}\n")
        print(f"  {op.access_W=}\n")
        print("\n------------------------\n")