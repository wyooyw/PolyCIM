from polycim.passes.multi_level_tiling import enumerate_tiling_factors
import polycim.op.benchmark as benchmark
from polycim.passes.multi_level_tiling import (
    multi_level_splitting_var_level, 
    combine_tilesize_by_symmetry_info
)
from polycim.utils.math import factorize
import islpy as isl
import polycim.utils.utils as utils
import itertools
from tqdm import tqdm
from polycim.passes.affine_transform import (
    parse_operator,
    find_base,
    base_to_coor_transform_schedule,
    shift_to_positive
)
from functools import partial
from collections import OrderedDict
import numpy as np
import time
from sympy import Matrix
from polycim.passes.hardware_merge_tiling import (
    get_coalescing_schedule_from_mapping,
    get_reverse_coalescing_schedule_from_mapping,
    _get_hardware_tiling_schedule
)
from polycim.config import get_config
from polycim.utils.draw import (
    draw,
    extract_frame_info
)
from polycim.depth_first.count_minimal_macro import count_minimal_needed_macro
from polycim.op.base_operator import BasicOperator
import concurrent.futures
import datetime
import os
from polycim.depth_first.timeout import timeout
from functools import reduce
import math
from polycim.passes.loop_padding import loop_padding_to_box_all, shift_to_zero
from polycim.depth_first.mapping_multiple_macro import mapping_multiple_macro
from polycim.codegen_.codegen_cimdsl import codegen_pass
from polycim.passes.tensorize import tensorize_pass
from polycim.passes.backend import backend_compile_and_profile_pass
from polycim.codegen_.codegen_data_layout_convert import (
    gcc_compile_data_layout_convert_code
)
from polycim.cli.arguments import get_args
from dataclasses import asdict
import json
execution_times = {}

def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Update the cumulative execution time for the function
        if func.__name__ not in execution_times:
            execution_times[func.__name__] = 0
        execution_times[func.__name__] += execution_time
        
        return result
    return wrapper

def get_symmetry_info(operator, dim_factors):
    pass

def get_tqdm(ls, desc=""):
    return tqdm(ls, desc=desc)
    # return ls

def change_dim_types_for_pre_tiling(split_factors, dim_types):
    new_dim_types = []
    for dim, dim_factors in enumerate(split_factors):
        if len(dim_factors) == 1:
            new_dim_types.append(dim_types[dim])
        elif len(dim_factors) == 2:
            new_dim_types.append(f"{dim_types[dim]}_o")
            new_dim_types.append(f"{dim_types[dim]}_i")
        elif len(dim_factors) == 3:
            new_dim_types.append(f"{dim_types[dim]}_o")
            new_dim_types.append(f"{dim_types[dim]}_m")
            new_dim_types.append(f"{dim_types[dim]}_i")
        else:
            raise ValueError(f"dim_factors={dim_factors} is not supported")
    return new_dim_types

def remove_all_one_factors(factors):
    new_factors = []
    for factor in factors:
        new_factor = [f for f in factor if f!=1]
        if len(new_factor) == 0:
            new_factor = [1]
        new_factors.append(new_factor)
    return new_factors

@timing_decorator
def pre_tiling(config):
    operator = config["op"]
    symmetry_info = config.get("symmetry_info", None)
    dim_types = config.get("dim_types", None)
    max_tiling_level = config.get("max_tiling_level", 2)
    not_tiling = config.get("not_tiling", None)

    domain = operator.domain
    assert max_tiling_level in (2,3), f"{max_tiling_level=}"

    assert domain.is_box(), f"domain={domain} is not box"
    n_iter = domain.dim(isl.dim_type.set)

    domain_shape = utils.get_static_box_shape(domain)
    dim_factors = []
    for i,dim_size in enumerate(domain_shape):
        if not_tiling is not None and i in not_tiling:
            dim_factors.append(((dim_size,),))
            continue
        factors = factorize(dim_size, max_tiling_level)
        # factors = [tuple(factor) for factor in factors if factor[0]!=1 and factor[1]!=1]
        factors = remove_all_one_factors(factors)
        factors = list({tuple(factor) for factor in factors})
        print(f"{factors=}")
        # factors = reversed(factors)
        # factors = [(dim_size,),*factors]
        dim_factors.append(tuple(factors))
    dim_factors = tuple(dim_factors)

    if symmetry_info is None:
        combination_list = list(itertools.product(*dim_factors))
    else:
        combination_list = combine_tilesize_by_symmetry_info(dim_factors, symmetry_info)
    # for combination in combination_list:
    #     print(f"{combination=}")
    # exit()
    new_combination_list = []
    # combination_list = [
    #     ((1,), (4, 14), (4, 14), (17, 3), (17, 3))
    # ]
    # combination_list = combination_list[10:14]
    for idx,combination in enumerate(combination_list):
        new_operator = multi_level_splitting_var_level(operator, combination)
        new_dim_types = change_dim_types_for_pre_tiling(combination, dim_types)
        new_combination_list.append((new_operator, combination, new_dim_types))
        # yield (new_operator, combination, new_dim_types)、
    return new_combination_list
    # return new_combination_list
    # for combination in combination_list:
    #     new_operator = multi_level_splitting_var_level(operator, combination)
    #     yield new_operator

class Base:
    def __init__(self, corrdinate, reuse_array_id, is_trival):
        self.corrdinate = tuple(corrdinate)
        self.reuse_array_id = reuse_array_id
        self.is_trival = is_trival

    def __str__(self):
        return f"Base(corrdinate={self.corrdinate}, reuse_array_id={self.reuse_array_id}, is_trival={self.is_trival})"

    def __eq__(self, other):
        if isinstance(other, Base):
            return self.corrdinate == other.corrdinate
        return False

    def __hash__(self):
        return hash(self.corrdinate)

def record_points(point, record):
    multi_val = point.get_multi_val()
    if multi_val.is_zero():
        return
    val = [int(str(multi_val.get_val(i))) for i in range(len(multi_val))]
    record.append(val)


def get_nontrival_points(set_):
    points = []
    record_points_fn = partial(record_points, record=points)
    set_.foreach_point(record_points_fn)
    return points

def nonzero_equal(corr1, corr2):
    nonzero1 = [int(i!=0) for i in corr1]
    nonzero2 = [int(i!=0) for i in corr2]
    return all(i1 == i2 for i1,i2 in zip(nonzero1, nonzero2))

def cover(corr1, corr2):
    nonzero1 = [int(i!=0) for i in corr1]
    nonzero2 = [int(i!=0) for i in corr2]
    return all(i1 >= i2 for i1,i2 in zip(nonzero1, nonzero2)) and any(i1 > i2 for i1,i2 in zip(nonzero1, nonzero2))

def filter_cover_points(points):
    new_points = []
    for point in points:
        can_cover = False
        for other_point in points:
            if point!=other_point and cover(point, other_point):
                can_cover = True
                break
        if not can_cover:
            new_points.append(point)
    return new_points

def filter_times_points(points):
    """
    For any two points A and B, if there exists a positive integer k such that A = B*k, then A is not valid.
    """
    filtered_points = []
    
    for i, point_a in enumerate(points):
        nonzero_elements = [i for i in point_a if i!=0]
        # gcd of nonzero_elements
        gcd = reduce(lambda x,y: math.gcd(x,y), nonzero_elements)
        if gcd == 1:
            filtered_points.append(point_a)
    return filtered_points

def filter_tile_direction_points(points, **kwargs):
    assert "tile_sizes" in kwargs
    tile_sizes = kwargs["tile_sizes"]
    tile_pair = []
    i_iter = 0
    for tile_size in tile_sizes:
        if len(tile_size) == 2:
            tile_pair.append((i_iter, i_iter + 1))
        elif len(tile_size) == 3:
            tile_pair.append((i_iter, i_iter + 1))
            tile_pair.append((i_iter, i_iter + 2))
            tile_pair.append((i_iter + 1, i_iter + 2))
        i_iter += len(tile_size)
    
    new_points = []
    for point in points:
        nonzero = [int(i!=0) for i in point]
        is_valid = True
        for tile_out_iter,tile_in_iter in tile_pair:
            if nonzero[tile_out_iter] == 1 and nonzero[tile_in_iter] == 1:
                is_valid = False
                break
        if is_valid:
            new_points.append(point)
    return new_points

def filter_many_direction_points(points):
    new_points = []
    for point in points:
        nonzero = [int(i!=0) for i in point]
        if sum(nonzero) <= 2:
            new_points.append(point)
    return new_points

def is_base_trival(point, dim_types):
    assert len(point)==len(dim_types), f"{point=} {dim_types=}"
    use_dim_types = []
    for p,dim_type in zip(point, dim_types):
        if p!=0:
            use_dim_types.append(dim_type)
    
    if len(use_dim_types)==1:
        return True
    if len(use_dim_types)==2 and (
        "oh_i" in use_dim_types or "ow_i" in use_dim_types or "oh" in use_dim_types or "ow" in use_dim_types
    ) and (
        "kh" in use_dim_types or "kw" in use_dim_types
    ):
        return True
    return False


def base_construction(operator, **kwargs):
    n_dim, n_array, dim_sizes, max_reuse_factor_for_arrays, hyperplanes_for_arrays = (
        parse_operator(
            domain=operator.domain,
            access_relations=[operator.access_I, operator.access_O],
        )
    )
    bases = OrderedDict()
    for array_id in [0,1]:
        hyperplanes = hyperplanes_for_arrays[array_id]
        reuse_bases = find_base(
            n_dim=n_dim, 
            dim_sizes=dim_sizes, 
            min_reuse_factor=1, 
            hyperplanes=hyperplanes, 
            exclude_null_space_of=None, 
            lex_lt_set=None
        )
        # import pdb; pdb.set_trace()
        points = get_nontrival_points(reuse_bases)
        points = filter_times_points(points)
        points = filter_tile_direction_points(points, **kwargs)
        points = filter_cover_points(points)
        points = filter_many_direction_points(points)
        # TODO: filter points
        for point in points:
            is_trival = is_base_trival(point, kwargs["dim_types"])
            base = Base(point, array_id, is_trival)
            bases[base] = None

    for i in range(n_dim):
        point = [0 for d in range(n_dim)]
        point[i] = 1
        base = Base(tuple(point), -1, True)
        bases[base] = None

    bases = list(bases.keys())

    return bases


def get_rank_key(base_combination):
    num_non_zero = 0
    for base in base_combination:
        if base.reuse_array_id in [0,1]:
            nonzero = [int(i!=0) for i in base.corrdinate]
            num_non_zero += sum(nonzero)
    return num_non_zero

def select_bases(bases, operator, **kwargs):
    n_dim = operator.domain.dim(isl.dim_type.set)
    selected_bases = []
    
    # Find all combinations of n_dim bases
    base_combinations = itertools.combinations(bases, n_dim)
    
    # Start timing
    start_time = time.time()

    for i,combination in enumerate(base_combinations):
        if satisfies_constraints(combination, operator, **kwargs):
            rank_key = get_rank_key(combination)
            selected_bases.append((rank_key, combination))
    selected_bases = sorted(selected_bases, key=lambda x: x[0])
    selected_bases = [x[1] for x in selected_bases]

    
    # import pdb; pdb.set_trace()
    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time
    # print(f"Time taken to select bases: {elapsed_time:.4f} seconds")
    # print(f"{len(selected_bases)} selected bases from {len(bases)} bases")
    return selected_bases

def satisfies_constraints(combination, operator, **kwargs):
    # Extract the coordinates from each item in the combination
    matrix = np.array([item.corrdinate for item in combination])

    # check reuse constraint
    # reuse_ids = {item.reuse_array_id for item in combination}
    # if not (0 in reuse_ids and 1 in reuse_ids):
    #     return False
    reuse_ids = {0:0, 1:0}
    reuse_by_skewed_base_ids = {0:0, 1:0}
    for item in combination:
        nonzero_count = sum([int(i!=0) for i in item.corrdinate])
        is_skewed_base = nonzero_count >= 2
        if item.reuse_array_id in (0,1):
            reuse_ids[item.reuse_array_id] += 1
            if is_skewed_base:
                reuse_by_skewed_base_ids[item.reuse_array_id] += 1
    if reuse_ids[0] < 1 or reuse_ids[1] < 1:
        return False
    if reuse_by_skewed_base_ids[0] > 2 or reuse_by_skewed_base_ids[1] > 2:
        return False

    # Check if the matrix is square and invertible
    if matrix.shape[0] != matrix.shape[1]:
        return False
    
    # Calculate the determinant to check if the matrix is invertible
    if np.linalg.det(matrix) == 0:
        return False
    # return True
    # filter by tiles
    reuse_bases = [base for base in combination if base.reuse_array_id in (0,1)]
    participate_skewing_dims = set()
    for base in reuse_bases:
        nonzero_dims = [i for i,corr in enumerate(base.corrdinate) if corr!=0]
        if len(nonzero_dims) > 1:
            participate_skewing_dims.update(nonzero_dims)
    tile_sizes = kwargs["tile_sizes"]

    cur_dim = 0
    tile_check_is_valid = True
    for i,tile_size in enumerate(tile_sizes):
        if len(tile_size) > 1:
            is_valid = False
            for j in range(len(tile_size)):
                _dim = cur_dim + j
                if _dim in participate_skewing_dims:
                    is_valid = True
                    break
            
            if not is_valid:
                tile_check_is_valid = False
                break
        
        cur_dim += len(tile_size)

    if not tile_check_is_valid:
        return False

    return True

@timing_decorator
def affine_transform(operator, **kwargs):
    # 1. base construction
    bases = base_construction(operator, **kwargs)  

    # 2. base selection
    # select n_dim base from bases that satisfy:
    #  constraint1: for each array, at least one base is selected to reuse it.
    #  constraint2: the selected bases are linear independent
    # find all selection combinations
    selected_bases_list = select_bases(bases, operator, **kwargs)
    # print(f"{kwargs['tile_sizes']=}")
    # print(f"{len(selected_bases_list)=}\n")
    # return []

    # 3. affine transform
    new_op_list = []
    for selected_bases in selected_bases_list:
        rev_selected_bases = tuple(list(selected_bases)[::-1])
        matrix = Matrix([base.corrdinate for base in rev_selected_bases])

        schedule = base_to_coor_transform_schedule(matrix)

        new_op = operator.apply_schedule(schedule, name="affine")
        new_op = shift_to_positive(new_op)
        if kwargs["pad"]:
            new_op = loop_padding_to_box_all(new_op)

        base_str = ""
        for base in selected_bases:
            base_str += str(base) + "\n"
        new_op.history_schedules.append({"bases":base_str})
        new_op.history_schedules.append({"matrix":matrix})

        new_op_list.append((new_op, selected_bases))

    
    return new_op_list

def get_mapping_from_bases(bases):
    reuse_axis = dict()
    all_array_ids = {base.reuse_array_id for base in bases if base.reuse_array_id != -1}
    for array_id in all_array_ids:
        reuse_axis[array_id] = list()

    for i,base in enumerate(bases):
        if base.reuse_array_id != -1:
            reuse_axis[base.reuse_array_id].append(f"s{i}")

    mapping = OrderedDict()
    for array_id, axis_list in reuse_axis.items():
        # h0: row, reuse output;
        # h1: col, reuse input
        hardware_axis = f"h{(1-array_id)}"

        mapping[hardware_axis] = tuple(axis_list)

    return mapping

@timing_decorator
def coalesce_and_tiling(operator, bases, cim_cfg, return_schedule=False):
    mapping = get_mapping_from_bases(bases)
    operator.history_schedules.append({"s2h_mapping":mapping})
    # print(f"{mapping=}")
    coalescing_schedule = get_coalescing_schedule_from_mapping(mapping, operator)
    reverse_coalescing_schedule = get_reverse_coalescing_schedule_from_mapping(mapping, operator)
    # print(f"{coalescing_schedule=}")
    tiling_factor = [
        cim_cfg.n_comp, cim_cfg.n_group_vcol
    ]
    tiling_schedule = _get_hardware_tiling_schedule(coalescing_schedule.range().dim(isl.dim_type.set), tiling_factor)
    # print(f"{tiling_schedule=}")
    if return_schedule:
        return [(coalescing_schedule, tiling_schedule)]
    else:
        new_op = operator.apply_schedule(coalescing_schedule, 
            # reverse_schedule=reverse_coalescing_schedule, 
            skip_simplify=True, 
            name="coalescing"
        )
        new_op = new_op.apply_schedule(tiling_schedule, skip_simplify=True, name="tiling")
        return [new_op]

def execute_with_timeout(func, *args, timeout=10):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(func, *args)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            print("Function execution timed out!")
            return None

@timing_decorator
def count_val_upto(domain, n):
    return domain.count_val_upto(n)


@timeout(seconds=4)
def count_val(domain):
    return int(str(domain.count_val()))

def get_num_div(domain):
    if type(domain) == isl.BasicSet:
        return domain.dim(isl.dim_type.div)
    elif type(domain) == isl.Set:
        total_divs = 0
        for bset in domain.get_basic_sets():
            total_divs += bset.dim(isl.dim_type.div)
        return total_divs
    else:
        raise ValueError(f"Unsupported domain type: {type(domain)}")

def get_show_num_div(domain):
    s = str(domain)
    if "exists" not in s:
        return 0

    s = s.split("exists")[1]
    s = s.split(":")[0]
    # num of 'e' in s
    return s.count("e")

def count_cim_compute_from_bases(op, bases, cim_cfg):
    n_dim = op.domain.dim(isl.dim_type.set)
    dim_sizes = utils.get_box_hull_shape(op.domain)
    
    new_dim_sizes = []
    for base in bases:
        corrdinate = base.corrdinate
        assert len(corrdinate)==n_dim
        size_per_dim = [dim_size//abs(base_corr) for dim_size, base_corr in zip(dim_sizes, corrdinate) if base_corr!=0]
        dim_size = max(size_per_dim)
        new_dim_sizes.append(dim_size)
    return new_dim_sizes


class SearchSpace:
    def __init__(self, cim_cfg, pad_count, delay_apply, num_macros, enable_weight_rewrite):
        assert isinstance(pad_count, bool)
        assert isinstance(delay_apply, bool)
        assert isinstance(enable_weight_rewrite, bool)
        self.cim_cfg = cim_cfg
        self.pad_count = pad_count
        self.delay_apply = delay_apply
        self.num_macros = num_macros
        self.enable_weight_rewrite = enable_weight_rewrite
        self.record_padding_friendly = (not pad_count) and delay_apply

    def search(self, config):
        result = []
        min_compute_times = config.get("min_compute_times_ub", int(1e9))
        min_compute_ops = list()
        stats = {"count_val": list()}
        for op1,tile_sizes,dim_types1 in get_tqdm(self.pre_tiling(
            config
        ), desc="pre_tiling"):            
            # print("a")
            begin_time = time.time()
            # for op2,bases in get_tqdm(self.affine_transform(op1, tile_sizes=tile_sizes, pad=self.pad_count, dim_types=dim_types1), desc="affine_transform"):
            for op2,bases in self.affine_transform(op1, tile_sizes=tile_sizes, pad=self.pad_count, dim_types=dim_types1):
                
                for op3 in self.coalesce_and_tiling(op2, bases, return_schedule=self.delay_apply):

                    if self.delay_apply:
                        coalescing_schedule, tiling_schedule = op3
                        domain = op2.domain
                        domain = coalescing_schedule.intersect_domain(domain).range()
                        domain = tiling_schedule.intersect_domain(domain).range()
                    else:
                        domain = op3.domain

                    if self.pad_count:
                        # get padding exe time
                        box_hull_shape = utils.get_box_hull_shape(domain)
                        outer_box_hull_shape = box_hull_shape[:-2]
                        exe_time = reduce(lambda x,y: x*y, outer_box_hull_shape)
                    else:
                        # result.append(op3)
                        n_dim = domain.dim(isl.dim_type.set)
                        outer_domain = domain.project_out(isl.dim_type.set, n_dim - 2, 2)
                        
                        # Use the execute_with_timeout function
                        exe_time = count_val(outer_domain)
                        if isinstance(exe_time, Exception):
                            print(f"error: {exe_time}")
                            exit()

                        if exe_time is None:
                            print(f"timeout")
                            continue
                        else:
                            assert isinstance(exe_time, int)
                    
                    if self.record_padding_friendly:
                        coalescing_schedule, tiling_schedule = op3
                        pad_domain = loop_padding_to_box_all(op2).domain
                        pad_domain = coalescing_schedule.intersect_domain(pad_domain).range()
                        pad_domain = tiling_schedule.intersect_domain(pad_domain).range()
                        pad_outer_domain = pad_domain.project_out(isl.dim_type.set, n_dim - 2, 2)
                        pad_exe_time = count_val(pad_outer_domain)
                        assert pad_exe_time is not None
                        assert isinstance(pad_exe_time, int)
                            
                        if pad_exe_time == exe_time:
                            padding_friendly = True
                        elif pad_exe_time > exe_time:
                            padding_friendly = False
                        else:
                            raise ValueError(f"pad_exe_time < exe_time: {pad_exe_time=}, {exe_time=}")

                    if exe_time is not None and exe_time <= min_compute_times:
                        
                        if self.delay_apply:
                            op3 = op2.apply_schedule(coalescing_schedule, skip_simplify=True, name="coalescing")
                            op3 = op3.apply_schedule(tiling_schedule, skip_simplify=True, name="tiling")

                        need_macro = count_minimal_needed_macro(op3, self.cim_cfg)
                        if self.num_macros >= need_macro or self.enable_weight_rewrite:
                            
                            record_info = {
                                "need_macros": need_macro,
                                "exe_time": exe_time,
                            }
                            if self.record_padding_friendly:
                                record_info["padding_friendly"] = padding_friendly
                                record_info["pad_exe_time"] = pad_exe_time

                            if exe_time < min_compute_times:
                                min_compute_ops = [op3]
                                min_compute_ops_info = [record_info]
                            else:
                                assert exe_time == min_compute_times
                                min_compute_ops.append(op3)
                                min_compute_ops_info.append(record_info)
                            min_compute_times = exe_time
                            print(f"min_compute_times={min_compute_times}, record_info={record_info}")
                
            end_time = time.time()
            
            # dump_schedules(min_compute_op)
            # print(f"time={end_time - begin_time}")
            # print("\n")
            # print(f"min_compute_times={min_compute_times}")
            # draw(min_compute_op, self.cim_cfg)
            # exit()
        return min_compute_times, min_compute_ops, min_compute_ops_info, stats
        

    def pre_tiling(self, config):
        return pre_tiling(config)
        # return [op]

    def affine_transform(self, op, **kwargs):
        return affine_transform(op, **kwargs)

    def coalesce_and_tiling(self, op, bases, return_schedule=False):
        return coalesce_and_tiling(op, bases, self.cim_cfg, return_schedule)

def show_result(min_compute_times, min_compute_ops, cim_cfg, flops, is_print=True):
    flops_per_cim_compute = flops / min_compute_times
    peak_flops_per_cim_compute = cim_cfg.n_comp * cim_cfg.n_group_vcol
    use_rate_percent = flops_per_cim_compute / peak_flops_per_cim_compute * 100
    
    result = OrderedDict()
    result["cim_cfg"] = asdict(cim_cfg)
    result["flops"] = flops
    result["min_compute_times"] = min_compute_times
    result["len(min_compute_ops)"] = len(min_compute_ops)
    result["flops_per_cim_compute"] = flops_per_cim_compute
    result["peak_flops_per_cim_compute"] = peak_flops_per_cim_compute
    result["use_rate"] = use_rate_percent

    if is_print:
        print(json.dumps(result, indent=4))
    return result

# dump_index = 0
def dump_schedules(origin_op, new_op, **kwargs):
    cim_cfg = kwargs["cim_cfg"]

    schedule_keys = ["pre_tiling", "affine", "shift_to_positive", "coalescing", "tiling"]
    comment_keys = ["tiling_factors", "bases", "s2h_mapping"]
    # global dump_index
    schedule_dict = OrderedDict()
    schedule_dict["tiling_factors"] = None
    schedule_dict["pre_tiling"] = None
    schedule_dict["bases"] = None
    schedule_dict["affine"] = None
    schedule_dict["shift_to_positive"] = None
    schedule_dict["s2h_mapping"] = None
    schedule_dict["coalescing"] = None
    schedule_dict["tiling"] = None
    
    for name_schedule in new_op.history_schedules:
        if type(name_schedule) == dict and list(name_schedule.keys())[0] in schedule_dict:
            name = list(name_schedule.keys())[0]
            schedule = name_schedule[name]
            schedule_dict[name] = str(schedule)

    
    dump_code = "\"\"\"\n"
    for key,value in kwargs.items():
        dump_code += f"{key} = {value}\n"
    dump_code += "\"\"\"\n"
    dump_code += f"import islpy as isl\n"
    dump_code += f"import time\n"
    dump_code += f"from polycim.op.base_operator import BasicOperator\n"
    dump_code += f"from polycim.utils.draw import draw, extract_frame_info\n"
    dump_code += f"from polycim.config import CIMConfig\n"

    cim_config_str = f"""
cim_cfg = CIMConfig(
    n_row={cim_cfg.n_row},
    n_group_vcol={cim_cfg.n_group_vcol},
    n_comp={cim_cfg.n_comp},
    n_group={cim_cfg.n_group},
    n_macro_per_group={cim_cfg.n_macro_per_group},
    n_macro={cim_cfg.n_macro}
)\n
"""
    dump_code += cim_config_str

    origin_op_str = f"""
op = BasicOperator(
    domain = isl.BasicSet(
        \"{origin_op.domain}\"
    ),
    access_I = isl.BasicMap(\"{origin_op.access_I}\"),
    access_O = isl.BasicMap(\"{origin_op.access_O}\"),
    access_W = isl.BasicMap(\"{origin_op.access_W}\"),
)
"""
    # dump_code += f"domain = isl.BasicSet(\"{init_domain}\")\n\n"
    dump_code += origin_op_str
    for key, value in schedule_dict.items():
        if key in schedule_keys:
            dump_code += f"schedule_{key} = isl.BasicMap(\"{value}\")\n"
            # dump_code += f"domain = schedule_{key}.intersect_domain(domain).range()\n\n"
            dump_code += f"op = op.apply_schedule(schedule_{key}, skip_simplify=True, name=\"{key}\")\n\n"
        else:
            dump_code += f"\"\"\"\n{key} = \n{value}\n\"\"\"\n"

    dump_code += """
domain = op.domain
n_dim = domain.dim(isl.dim_type.set)
begin_time = time.time()
outer_domain = domain.project_out(isl.dim_type.set, n_dim - 2, 2)
val = outer_domain.count_val()
dur_time = time.time() - begin_time
print(f"outer_domain.count_val {val=}, {dur_time=}")
draw(op, cim_cfg)
    """
    # save_dir = "dump_code"
    # os.makedirs(save_dir, exist_ok=True)
    # with open(os.path.join(save_dir, f"dump_code_{dump_index}.py"), "w") as f:
    #     f.write(dump_code)
    # dump_index += 1
    # print("dump_code saved to dump_code.py")
    # exit()
    return dump_code

def dump_op(save_dir, origin_op, min_compute_times, min_compute_ops, min_compute_ops_info, cim_cfg, flops):
    os.makedirs(save_dir, exist_ok=True)
    for op_idx,op in enumerate(min_compute_ops):
        save_dir_solution = os.path.join(save_dir, f"solution_{op_idx}")
        os.makedirs(save_dir_solution, exist_ok=True)

        # save schedule code
        dump_code = dump_schedules(origin_op, op, min_compute_times=min_compute_times, cim_cfg=cim_cfg, flops=flops)
        with open(os.path.join(save_dir_solution, f"schedule_code.py"), "w") as f:
            f.write(dump_code)
        
        # save mapping pictures
        for idx, value in enumerate(extract_frame_info(op, cim_cfg, different_weight=True)):
            timestamp, frame_info = value
            frame_str = f"Index: {idx}.    Timestamp: {timestamp}\n"
            frame_str += frame_info.get_str(brief=False)
            picure_save_path = os.path.join(save_dir_solution, f"frame_{idx}.txt")
            with open(picure_save_path, "w") as f:
                f.write(frame_str)
            print(f"mapping pictures to {picure_save_path}")
            break

        # save result
        result_json = show_result(min_compute_times, min_compute_ops, cim_cfg, flops, is_print=False)
        min_compute_op_info = min_compute_ops_info[op_idx]
        result_json["min_compute_op_need_macros"] = min_compute_op_info.get("need_macros", None)
        result_json["min_compute_op_padding_friendly"] = min_compute_op_info.get("padding_friendly", None)
        with open(os.path.join(save_dir_solution, f"result.json"), "w") as f:
            json.dump(result_json, f, indent=4)

    print(f"op save to {save_dir}")
            

def run_op_list(op_list, save_dir, pad_count, delay_apply, num_macros, enable_weight_rewrite, cim_config):
    enable_mapping_multiple_macro = pad_count

    search_space = SearchSpace(cim_config, 
                              pad_count=pad_count, 
                              delay_apply=delay_apply,
                              num_macros=num_macros,
                              enable_weight_rewrite=enable_weight_rewrite)
    for name,config in op_list.items():
        if isinstance(config, BasicOperator):
            config = {"op": config}
        print(f"{name=}")
        begin_time = time.time()
        min_compute_times, min_compute_ops, min_compute_ops_info, stats = search_space.search(config)
        end_time = time.time()
        print(f"\ntime={end_time - begin_time}")

        op = config["op"]
        flops = int(str(op.domain.count_val()))
        show_result(min_compute_times, min_compute_ops, cim_config, flops)
        
        dump_op(os.path.join(save_dir, name), op, min_compute_times, min_compute_ops, min_compute_ops_info, cim_config, flops)        
        
        if enable_mapping_multiple_macro:
            new_op = mapping_multiple_macro(min_compute_ops[0], cim_config, enable_weight_rewrite=enable_weight_rewrite)
        # print("\n")
        
        # save data layout convert code
        data_layout_convert_code = new_op.attr["data_layout_convert_code"]
        save_op_dir = os.path.join(save_dir, name, "0")
        os.makedirs(save_op_dir, exist_ok=True)
        for key, value in data_layout_convert_code.items():
            code_path = os.path.join(save_op_dir, f"convert_{key}.cpp")   
            with open(code_path, "w") as f:
                f.write(value)
            exe_path = os.path.join(save_op_dir, f"convert_{key}.o")
            gcc_compile_data_layout_convert_code(code_path, exe_path)

        new_op = tensorize_pass([new_op])[0]
        new_op = codegen_pass([new_op])[0]
        result_list = backend_compile_and_profile_pass(
            [new_op], 
            save_dir=os.path.join(save_dir, name),
            config_file=get_args().config_path
        )
        
        # save stats["count_val"] into a csv file
        # header: count_time, exe_time
        # with open(os.path.join(save_dir, f"{name}.csv"), "w") as f:
        #     f.write("count_time,exe_time,num_div,num_show_div\n")
        #     for count_time, exe_time, num_div, num_show_div in stats["count_val"]:
        #         f.write(f"{count_time:.2f},{exe_time},{num_div},{num_show_div}\n")

@timing_decorator
def main():
    pad_count = True
    delay_apply = True
    num_macros = 16
    enable_weight_rewrite = True

    symmetry_info_for_dwconv2d = ((1,3),(2,4))
    dim_types_for_dwconv2d = ["c", "oh", "ow", "kh", "kw"]

    op_list = OrderedDict()
    op_list["C1"] = {
        "op": benchmark.get_op_dwconv2d(ic=1, oh=112, ow=112, kh=3, kw=3, stride=1, dilation=1, virtual_axis=False),
        "symmetry_info": symmetry_info_for_dwconv2d,
        "dim_types": dim_types_for_dwconv2d,
    }
    op_list["C2"] = {
        "op": benchmark.get_op_dwconv2d(ic=1, oh=56, ow=56, kh=3, kw=3, stride=1, dilation=1, virtual_axis=False),
        "symmetry_info": symmetry_info_for_dwconv2d,
        "dim_types": dim_types_for_dwconv2d,
    }
    op_list["C3"] = {
        "op": benchmark.get_op_dwconv2d(ic=1, oh=28, ow=28, kh=5, kw=5, stride=1, dilation=1, virtual_axis=False),
        "symmetry_info": symmetry_info_for_dwconv2d,
        "dim_types": dim_types_for_dwconv2d,
    }
    op_list["C4"] = {
        "op": benchmark.get_op_dwconv2d(ic=1, oh=14, ow=14, kh=3, kw=3, stride=1, dilation=1, virtual_axis=False),
        "symmetry_info": symmetry_info_for_dwconv2d,
        "dim_types": dim_types_for_dwconv2d,
    }
    op_list["C5"] = {
        "op": benchmark.get_op_dwconv2d(ic=1, oh=14, ow=14, kh=5, kw=5, stride=1, dilation=1, virtual_axis=False),
        "symmetry_info": symmetry_info_for_dwconv2d,
        "dim_types": dim_types_for_dwconv2d,
    }
    op_list["C6"] = {
        "op": benchmark.get_op_dwconv2d(ic=1, oh=7, ow=7, kh=5, kw=5, stride=1, dilation=1, virtual_axis=False),
        "symmetry_info": symmetry_info_for_dwconv2d,
        "dim_types": dim_types_for_dwconv2d,
    }
    op_list["C7"] = {
        "op": benchmark.get_op_dwconv2d(ic=1, oh=7, ow=7, kh=3, kw=3, stride=1, dilation=1, virtual_axis=False),
        "symmetry_info": symmetry_info_for_dwconv2d,
        "dim_types": dim_types_for_dwconv2d,
    }
    op_list["C8"] = {
        "op": benchmark.get_op_dwconv2d(ic=1, oh=56, ow=56, kh=7, kw=7, stride=1, dilation=1, virtual_axis=False),
        "symmetry_info": symmetry_info_for_dwconv2d,
        "dim_types": dim_types_for_dwconv2d,
    }
    op_list["C9"] = {
        "op": benchmark.get_op_dwconv2d(ic=1, oh=8, ow=7, kh=7, kw=7, stride=1, dilation=1, virtual_axis=False),
        # "symmetry_info": symmetry_info_for_dwconv2d,
        "dim_types": dim_types_for_dwconv2d,
    }
    op_list["C10"] = {
        "op": benchmark.get_op_dwconv2d(ic=1, oh=56, ow=56, kh=51, kw=51, stride=1, dilation=1, virtual_axis=False),
        "symmetry_info": symmetry_info_for_dwconv2d,
        "dim_types": dim_types_for_dwconv2d,
        "not_tiling": [3,4]
    }
    op_list["C11"] = {
        "op": benchmark.get_op_dwconv2d(ic=1, oh=7, ow=7, kh=13, kw=13, stride=1, dilation=1, virtual_axis=False),
        "symmetry_info": symmetry_info_for_dwconv2d,
        "dim_types": dim_types_for_dwconv2d
    }
    op_list["C12"] = {
        "op": benchmark.get_op_dwconv2d(ic=1, oh=28, ow=28, kh=3, kw=3, stride=1, dilation=2, virtual_axis=False),
        "symmetry_info": symmetry_info_for_dwconv2d,
        "dim_types": dim_types_for_dwconv2d,
        "max_tiling_level": 3
    }
    op_list["C13"] = {
        "op": benchmark.get_op_dwconv2d(ic=1, oh=28, ow=28, kh=5, kw=5, stride=1, dilation=2, virtual_axis=False),
        "symmetry_info": symmetry_info_for_dwconv2d,
        "dim_types": dim_types_for_dwconv2d,
        "max_tiling_level": 3
    }
    # # op_list["C14"] = benchmark.get_op_dwconv2d(b=1, oc=1, ic=1, oh=28, ow=28, kh=5, kw=5, stride=1, dilation=2)
    # op_list["C15"] = {
    #     "op": benchmark.get_op_dwconv3d(ic=1, ox=28, oy=28, oz=28, kx=5, ky=5, kz=5, stride=1),
    #     "symmetry_info": ((1,4),(2,5),(3,6)),
    #     "dim_types": ["c", "ox", "oy", "oz", "kx", "ky", "kz"],
    #     "min_compute_times_ub": 19300
    # }

    # op_list["test"] = {
    #     "op": benchmark.get_op_dwconv2d(ic=1, oh=14, ow=14, kh=4, kw=4, stride=1, dilation=1, virtual_axis=False),
    #     "symmetry_info": symmetry_info_for_dwconv2d,
    #     "dim_types": dim_types_for_dwconv2d,
    #     "not_tiling": [1,2]
    # }
    # op_list["test"] = (
    #     benchmark.get_op_dwconv2d(ic=4, oh=16, ow=16, kh=3, kw=3, stride=1, dilation=1, virtual_axis=False),
    #     # None
    #     # symmetry_info
    #     ((1,3),(2,4))
    # )

    curr_time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    cim_cfg = get_config()
    curr_time_str = curr_time_str + f"_{cim_cfg.n_comp}x{cim_cfg.n_group_vcol*8}"
    save_dir = os.path.join(".save", curr_time_str)
    run_op_list(op_list, save_dir, pad_count=pad_count, delay_apply=delay_apply, num_macros=num_macros, enable_weight_rewrite=enable_weight_rewrite)
    # exit()

    # # print(code)
    # # import pdb; pdb.set_trace()
    # for idx, value in enumerate(extract_frame_info(min_compute_op, cim_cfg, different_weight=True)):
    #     timestamp, frame_info = value
    #     print(f"Index: {idx}.    Timestamp: {timestamp}")
    #     frame_info.print()
    #     c = input("continue?(y/n):")
    #     if c=="n":
    #         break
    #     else:
    #         continue

if __name__ == "__main__":
    
    main()
    # test_single_op()

