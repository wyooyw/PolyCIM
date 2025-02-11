import islpy as isl
import polycim.utils.utils as utils
from polycim.op.base_operator import BasicOperator
from itertools import combinations
import math
from collections import OrderedDict
import pdb
import numpy as np
from tqdm import tqdm
from polycim.config import CIMConfig, get_config

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
    I_index_set = utils.extract_index_set(op.access_I)
    W_index_set = utils.extract_index_set(op.access_W)
    O_index_set = utils.extract_index_set(op.access_O)
    bitmap = utils.convert_index_set_to_bitmap(O_index_set, I_index_set, W_index_set)
    bitmap = OrderedDict(sorted(bitmap.items(), key=lambda x: int(x[0][1:])))
    return bitmap

def map_software_index_to_hardware_index(software_access_bitmap, hardware_access_bitmap):
    def rename_access_bitmap(access_bitmap, char):
        new_bitmap = OrderedDict()
        for key, value in access_bitmap.items():
            new_bitmap[char+key[1]] = tuple(value)
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
        combinations_r = combinations(elements, r)
        all_combinations.extend(combinations_r)
    return all_combinations

def sort_by_name(obj):
    return sorted(obj, key=lambda x: int(x[1:]))

def generate_combinations_exclude(elements, exclude):
    elements = sort_by_name(set(elements) - set(exclude))
    return generate_combinations(elements)

def get_all_software_to_hardware_index_mapping(h2s_mapping, map_all=False):
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

        map_all: map all software axis to hardware axis
    """
    hardware_indexs = list(h2s_mapping.keys())

    if map_all:
        mapping = dict()
        for h_axis, s_axis_list in h2s_mapping.items():
            mapping[h_axis] = tuple(s_axis_list)
        return [mapping]


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

def get_schedule_from_mapping(mapping, software_op):
    """
    mapping:  {'h0': ('s1',), 'h1': ('s4', 's5')}
    scheudle: { [s0, s1, s2, s3, s4, s5] -> [s0, s2, s3, s1, 3s4 + s5] }
    """
    new_domain = utils.rename_all_dims_for_basic_set(software_op.domain,'s')
    # print(mapping)
    bounds = utils.get_bound_for_all_dims(new_domain)
    schedule_range = list()

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

def get_macro_4d_hardware_tiling_schedule(op, software_schedule, cim_cfg):
    """
    OrderedDict([('h0', [0, 1, 1]), ('h1', [1, 0, 1]), ('h2', [1, 1, 0])])

    macros = Buffer(<N_ROW, N_COMP, N_GROUP, N_GROUP_VCOL>, int8, macro)
    """
    n_software_dim = software_schedule.range().as_set().n_dim()
    n_hardware_dim = 3

    new_domain = op.apply_schedule(software_schedule, skip_simplify=True).domain
    sizes = utils.get_box_hull_shape(new_domain)
    n_h0 = sizes[-3]
    n_igroup = math.ceil(
        math.ceil(n_h0 / cim_cfg.n_comp)
        / cim_cfg.n_row
    )
    n_ogroup = cim_cfg.n_group // n_igroup

    keep_iters = []
    for i in range(n_software_dim - n_hardware_dim):
        keep_iters.append(f"s{i}")
    
    # tiling
    tile_domain_iters = keep_iters + ["h0","h1","h2"]
    tile_range_iters = keep_iters + [
        f"floor(floor(h0/{cim_cfg.n_comp}) / {cim_cfg.n_row})",
        f"floor(h0/{cim_cfg.n_comp}) % {cim_cfg.n_row}", 
        f"h0 % {cim_cfg.n_comp}",
        f"h1",
        f"floor(h2 / {n_ogroup})",
        f"h2 % {n_ogroup}",
    ]
    tile_domain_iters_def = ", ".join(tile_domain_iters)
    tile_range_iters_def = ", ".join(tile_range_iters)
    tile_def = f"{{ [{tile_domain_iters_def}] -> [{tile_range_iters_def}] }}"
    print(f"{tile_def=}")
    tile_schedule = isl.BasicMap(tile_def)

    # merge inner-group and outer-group
    merge_domain_iters = keep_iters + ["igroup","row","comp","col","time","ogroup"]
    merge_range_iters = keep_iters + ["time", "row", "comp", f"ogroup * {n_igroup} + igroup", "col"]
    merge_domain_iters_def = ", ".join(merge_domain_iters)
    merge_range_iters_def = ", ".join(merge_range_iters)
    merge_group_def = f"{{ [{merge_domain_iters_def}] -> [{merge_range_iters_def}] }}"
    merge_group_schedule = isl.BasicMap(merge_group_def)

    return tile_schedule, merge_group_schedule

def macro_4d_hardware_tiling(op, all_schedule, cim_cfg):
    new_schedules = []
    for schedule in all_schedule:
        tile_schedule, merge_group_schedule = get_macro_4d_hardware_tiling_schedule(op, schedule, cim_cfg)
        # schedule = schedule.apply_range(hardware_tiling_schedule)
        new_schedules.append((schedule, tile_schedule, merge_group_schedule))
    return new_schedules

def get_macro_5d_hardware_tiling_schedule(op, software_schedule, cim_cfg):
    """
    OrderedDict([('h0', [0, 1, 1]), ('h1', [1, 0, 1]), ('h2', [1, 1, 0])])

    macros = Buffer(<N_ROW, N_COMP, N_GROUP, N_GROUP_VCOL>, int8, macro)
    """
    n_software_dim = software_schedule.range().as_set().n_dim()
    n_hardware_dim = 3

    new_domain = op.apply_schedule(software_schedule, skip_simplify=True).domain
    sizes = utils.get_box_hull_shape(new_domain)
    n_h0 = sizes[-3]
    n_igroup = math.ceil(
        math.ceil(n_h0 / cim_cfg.n_comp)
        / cim_cfg.n_row
    )
    n_ogroup = cim_cfg.n_group // n_igroup

    keep_iters = []
    for i in range(n_software_dim - n_hardware_dim):
        keep_iters.append(f"s{i}")
    
    # tiling
    tile_domain_iters = keep_iters + ["h0","h1","h2"]
    tile_range_iters = keep_iters + [
        f"floor(floor(h0/{cim_cfg.n_comp}) / {cim_cfg.n_row})",
        f"floor(h0/{cim_cfg.n_comp}) % {cim_cfg.n_row}", 
        f"h0 % {cim_cfg.n_comp}",
        f"h1",
        f"floor(h2 / {n_ogroup})",
        f"h2 % {n_ogroup}",
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
    reorder_domain_iters = keep_iters + ["igroup","row","comp","col","time","ogroup"]
    reorder_range_iters = keep_iters + ["time", "row", "comp", "ogroup", "igroup", "col"]
    reorder_domain_iters_def = ", ".join(reorder_domain_iters)
    reorder_range_iters_def = ", ".join(reorder_range_iters)
    reorder_def = f"{{ [{reorder_domain_iters_def}] -> [{reorder_range_iters_def}] }}"
    reorder_schedule = isl.BasicMap(reorder_def)


    return tile_schedule, reorder_schedule

def macro_5d_hardware_tiling(op, all_schedule, cim_cfg):
    new_schedules = []
    for schedule in all_schedule:
        tile_schedule, reorder_schedule = get_macro_5d_hardware_tiling_schedule(op, schedule, cim_cfg)
        # schedule = schedule.apply_range(hardware_tiling_schedule)
        new_schedules.append((schedule, tile_schedule, reorder_schedule))
    return new_schedules

def hardware_merge_tiling(op, cim_cfg):
    # n_rows = macro_row
    # n_cols = macro_col
    # cim_op = get_cim_operator(n_rows, n_cols)
    software_access_bitmap = get_access_bitmap(op)
    # [output, Input, Weight]
    hardware_access_bitmap = OrderedDict([('h0', [0, 1, 1]), ('h1', [1, 0, 1]), ('h2', [1, 1, 0])])
    
    mapping = map_software_index_to_hardware_index(software_access_bitmap, hardware_access_bitmap)
    if mapping is None: 
        return None
    all_mapping = get_all_software_to_hardware_index_mapping(mapping, map_all=True)
    all_schedules = [get_schedule_from_mapping(mapping, op) for mapping in all_mapping]
    for schedule in all_schedules:
        assert schedule.intersect_domain(op.domain).reverse().is_single_valued(), f"{schedule} should not be single valued!"
    
    # all_schedules = macro_4d_hardware_tiling(op, all_schedules, cim_cfg)
    all_schedules = macro_5d_hardware_tiling(op, all_schedules, cim_cfg)
    # for schedule in all_schedules:
    #     assert schedule.intersect_domain(op.domain).reverse().is_single_valued(), f"{schedule} should be single valued!"
    # exit()
    return all_schedules

def filter_op_by_execution_time_pass(op_list):
    if len(op_list) == 0:
        print(f"[filter_op_by_execution_time_pass]: \n    0 inputs. skip.")
        return op_list

    exe_time_list = []
    for op in tqdm(op_list):
        n_dim = op.domain.dim(isl.dim_type.set)
        outer_domain = op.domain.project_out(isl.dim_type.set, n_dim - 2, 2)
        exe_time = int(str(outer_domain.count_val()))
        exe_time_list.append(exe_time)

    exe_time_list = np.array(exe_time_list)
    sorted_indices = np.argsort(exe_time_list)

    new_op_list = []
    new_op_list_execution_time = []
    min_value = exe_time_list[sorted_indices[0]]
    num_ops = len(op_list)
    for i,index in enumerate(sorted_indices):
        if i < 3 or exe_time_list[index] == min_value:
            new_op_list.append(op_list[index])
            new_op_list_execution_time.append(exe_time_list[index])
        # print(f"{i}\n")
        # print(f"outer_count_val: {exe_time_list[index]}\n")
        # op = op_list[index]
        # print(f"skewing: {op.history_schedules[0]}\n")
        # print(f"merge: {op.history_schedules[2]}\n")
        # print(f"tiling: {op.history_schedules[3]}\n")
        # print("------------------------------------\n")
    new_op_list_execution_time = np.array(new_op_list_execution_time)

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
""")

    return new_op_list

def hardware_merge_tiling_pass(op_list, cim_cfg=None):
    # cim_cfg = CIMConfig(
    #     n_row=4,
    #     n_group_vcol=4,
    #     n_comp=4,
    #     n_group=4,
    # )
    cim_cfg = get_config()
    new_op_list = []
    schedule_fail_op_cnt = 0
    for op in tqdm(op_list):
        assert op.access_I.is_single_valued(), f"{op.access_I} should be single valued!"
        assert op.access_W.is_single_valued(), f"{op.access_W} should be single valued!"
        assert op.access_O.is_single_valued(), f"{op.access_O} should be single valued!"
        
        schedules = hardware_merge_tiling(op, cim_cfg)
        if schedules is None:
            schedule_fail_op_cnt += 1
            continue
        # print("-------------------------------------")
        for idx,(merge_schedule, tile_schedule, merge_group_schedule) in enumerate(schedules):
            # print(f"{idx=}")
            # print(f"{merge_schedule=}")
            # print(f"{tile_schedule=}")
            # print("-------------------------------------")
            new_op = op.apply_schedule(merge_schedule, skip_simplify=True)            
            new_op = new_op.apply_schedule(tile_schedule, skip_simplify=True)
            new_op = new_op.apply_schedule(merge_group_schedule, skip_simplify=True)

            # check domain not exceed macro's size
            # N_ROW, N_COMP, N_GROUP, N_GROUP_VCOL
            sizes = utils.get_box_hull_shape(new_op.domain)
            assert sizes[-5] <= cim_cfg.n_row, f"{sizes=}. {sizes[-4]=} should be less than {cim_cfg.n_row=}"
            assert sizes[-4] <= cim_cfg.n_comp, f"{sizes=}. {sizes[-3]=} should be less than {cim_cfg.n_comp=}"
            assert sizes[-3] * sizes[-2] <= cim_cfg.n_group, f"{sizes=}. {sizes[-2]=} should be less than {cim_cfg.n_group=}"
            assert sizes[-1] <= cim_cfg.n_group_vcol, f"{sizes=}. {sizes[-1]=} should be less than {cim_cfg.n_group_vcol=}"

            new_op_list.append(new_op)
    print(f"[hardware_merge_tiling_pass]: \n    {len(op_list)} ops input.\n    {schedule_fail_op_cnt} op schedule fail.\n    {len(new_op_list)} ops output.")
    return new_op_list

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