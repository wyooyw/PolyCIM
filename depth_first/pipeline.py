from multi_level_tiling import enumerate_tiling_factors
import benchmark
from multi_level_tiling import factorize, multi_level_splitting_var_level
import islpy as isl
import utils
import itertools
from tqdm import tqdm
from affine_transform import (
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
from hardware_merge_tiling import (
    get_schedule_from_mapping,
    _get_hardware_tiling_schedule
)
from config import get_config
from draw import extract_frame_info
from base_operator import BasicOperator
import concurrent.futures

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

@timing_decorator
def pre_tiling(operator):
    domain = operator.domain
    tiling_factor = 2

    assert domain.is_box(), f"domain={domain} is not box"
    n_iter = domain.dim(isl.dim_type.set)

    domain_shape = utils.get_static_box_shape(domain)
    dim_factors = []
    for dim_size in domain_shape:
        factors = factorize(dim_size, tiling_factor)
        factors = [factor for factor in factors if factor[0]!=1 and factor[1]!=1]
        factors = [*factors,[dim_size]]
        dim_factors.append(factors)

    combination_list = list(itertools.product(*dim_factors))
    new_combination_list = []
    for combination in combination_list:
        new_operator = multi_level_splitting_var_level(operator, combination)
        new_combination_list.append(new_operator)
    return new_combination_list[len(new_combination_list)//3:]
    # for combination in combination_list:
    #     new_operator = multi_level_splitting_var_level(operator, combination)
    #     yield new_operator

class Base:
    def __init__(self, corrdinate, reuse_array_id):
        self.corrdinate = tuple(corrdinate)
        self.reuse_array_id = reuse_array_id

    def __str__(self):
        return f"Base(corrdinate={self.corrdinate}, reuse_array_id={self.reuse_array_id})"

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
        is_valid = True
        for j, point_b in enumerate(points):
            if i == j:
                continue
            if not nonzero_equal(point_a, point_b):
                continue
                
            # Check if point_b is a scalar multiple of point_a
            ratios = [a / b if b != 0 else None for a, b in zip(point_a, point_b)]
            ratios = [ratio for ratio in ratios if ratio is not None]
            assert len(ratios) > 0
            if all(ratio == ratios[0] for ratio in ratios) and ratios[0] >= 1:
                is_valid = False
                break

        if is_valid:
            filtered_points.append(point_a)
    return filtered_points

def base_construction(operator):
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
        points = filter_cover_points(points)
        # TODO: filter points
        for point in points:
            base = Base(point, array_id)
            bases[base] = None

    for i in range(n_dim):
        point = [0 for d in range(n_dim)]
        point[i] = 1
        base = Base(tuple(point), -1)
        bases[base] = None

    bases = list(bases.keys())
    # filter bases
    # new_bases = []
    # for i,base in enumerate(bases):
    #     can_cover = False
    #     for j,subset_base in enumerate(bases):
    #         if i == j:
    #             continue
    #         if cover(base, subset_base):
    #             can_cover = True
    #             break
    #     if not can_cover:
    #         new_bases.append(base)
    # bases = new_bases

    return bases

    

def select_bases(bases, operator):
    n_dim = operator.domain.dim(isl.dim_type.set)
    selected_bases = []
    
    # Find all combinations of n_dim bases
    base_combinations = itertools.combinations(bases, n_dim)
    
    # Start timing
    start_time = time.time()
    
    for combination in base_combinations:
        if satisfies_constraints(combination, operator):
            selected_bases.append(combination)
    
    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time
    # print(f"Time taken to select bases: {elapsed_time:.4f} seconds")
    # print(f"{len(selected_bases)} selected bases from {len(bases)} bases")

    return selected_bases

def satisfies_constraints(combination, operator):
    # Extract the coordinates from each item in the combination
    matrix = np.array([item.corrdinate for item in combination])

    # check reuse constraint
    reuse_ids = {item.reuse_array_id for item in combination}
    if not (0 in reuse_ids and 1 in reuse_ids):
        return False

    # Check if the matrix is square and invertible
    if matrix.shape[0] != matrix.shape[1]:
        return False
    
    # Calculate the determinant to check if the matrix is invertible
    if np.linalg.det(matrix) == 0:
        return False

    return True

@timing_decorator
def affine_transform(operator):
    # print(f"{operator.domain=}")
    # 1. base construction
    bases = base_construction(operator)  
    # for base in bases:
    #     print(str(base))
    # print("\n")
    
    # 2. base selection
    # select n_dim base from bases that satisfy:
    #  constraint1: for each array, at least one base is selected to reuse it.
    #  constraint2: the selected bases are linear independent
    # find all selection combinations
    selected_bases_list = select_bases(bases, operator)
    # print(f"{len(selected_bases_list)=}")
    # exit()
    
    # 3. affine transform
    new_op_list = []
    for selected_bases in selected_bases_list:
        rev_selected_bases = tuple(list(selected_bases)[::-1])
        matrix = Matrix([base.corrdinate for base in rev_selected_bases])
        # print("")
        # print(f"{matrix=}")
        schedule = base_to_coor_transform_schedule(matrix)
        # print("")
        # print(f"{schedule=}")
        new_op = operator.apply_schedule(schedule)
        new_op = shift_to_positive(new_op)
        # print("")
        # print(f"{new_op.domain=}")
        new_op_list.append((new_op, selected_bases))
    # exit()
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
def coalesce_and_tiling(operator, bases, cim_cfg):
    mapping = get_mapping_from_bases(bases)
    # print(f"{mapping=}")
    coalescing_schedule = get_schedule_from_mapping(mapping, operator)
    # print(f"{coalescing_schedule=}")
    tiling_factor = [
        cim_cfg.n_comp, cim_cfg.n_group_vcol
    ]
    tiling_schedule = _get_hardware_tiling_schedule(coalescing_schedule.range().dim(isl.dim_type.set), tiling_factor)
    # print(f"{tiling_schedule=}")
    new_op = operator.apply_schedule(coalescing_schedule, skip_simplify=True)
    new_op = new_op.apply_schedule(tiling_schedule, skip_simplify=True)
    # exit()
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


@timing_decorator
def count_val(domain):
    return domain.count_val()

class SearchSpace:
    def __init__(self, cim_cfg):
        self.cim_cfg = cim_cfg

    def search(self, op):
        result = []
        min_compute_times = int(1e9)
        min_compute_op = None
        for op1 in tqdm(self.pre_tiling(op), desc="pre_tiling"):            
            for op2,bases in tqdm(self.affine_transform(op1), desc="affine_transform"):
                begin_time = time.time()
                for op3 in self.coalesce_and_tiling(op2, bases):
                    # result.append(op3)
                    n_dim = op3.domain.dim(isl.dim_type.set)
                    outer_domain = op3.domain.project_out(isl.dim_type.set, n_dim - 2, 2)
                    
                    # Use the execute_with_timeout function
                    exe_time = count_val_upto(outer_domain, min_compute_times + 1)
                    
                    if exe_time is not None and exe_time < min_compute_times:
                        min_compute_times = int(str(exe_time))
                        min_compute_op = op3
                        print(f"min_compute_times={min_compute_times}")
                end_time = time.time()
                # print(f"time={end_time - begin_time}")
                # print("\n")
        return min_compute_times, min_compute_op
        

    def pre_tiling(self, op):
        return pre_tiling(op)
        # return [op]

    def affine_transform(self, op):
        return affine_transform(op)

    def coalesce_and_tiling(self, op, bases):
        return coalesce_and_tiling(op, bases, self.cim_cfg)

def show_result(min_compute_times, cim_cfg, flops):
    flops_per_cim_compute = flops / min_compute_times
    peak_flops_per_cim_compute = cim_cfg.n_comp * cim_cfg.n_group_vcol
    use_rate_percent = flops_per_cim_compute / peak_flops_per_cim_compute * 100
    print(f"min_compute_times: {min_compute_times}")
    print(f"cim: {cim_cfg.n_comp} comp, {cim_cfg.n_group_vcol} group_vcol")
    print(f"flops={flops}")
    print(f"flops_per_cim_compute={flops_per_cim_compute}")
    print(f"peak_flops_per_cim_compute={peak_flops_per_cim_compute}")
    print(f"use_rate={use_rate_percent:.2f}%")

def run_op_list(op_list):
    cim_cfg = get_config()
    search_space = SearchSpace(cim_cfg)
    for name,op in op_list.items():
        print(f"{name=}")
        min_compute_times, min_compute_op = search_space.search(op)
        flops = int(str(op.domain.count_val()))
        show_result(min_compute_times, cim_cfg, flops)
        print("\n")

@timing_decorator
def main():
    op_list = OrderedDict()
    # op_list["C1"] = benchmark.get_op_dwconv2d(ic=1, oh=112, ow=112, kh=3, kw=3, stride=1, dilation=1)
    # op_list["C2"] = benchmark.get_op_dwconv2d(ic=1, oh=56, ow=56, kh=3, kw=3, stride=1, dilation=1)
    # op_list["C3"] = benchmark.get_op_dwconv2d(ic=1, oh=28, ow=28, kh=5, kw=5, stride=1, dilation=1)
    # op_list["C4"] = benchmark.get_op_dwconv2d(ic=1, oh=14, ow=14, kh=3, kw=3, stride=1, dilation=1)
    # op_list["C5"] = benchmark.get_op_dwconv2d(ic=1, oh=14, ow=14, kh=5, kw=5, stride=1, dilation=1)
    # op_list["C6"] = benchmark.get_op_dwconv2d(ic=1, oh=7, ow=7, kh=3, kw=3, stride=1, dilation=1)
    # op_list["C7"] = benchmark.get_op_dwconv2d(ic=1, oh=7, ow=7, kh=5, kw=5, stride=1, dilation=1)
    op_list["C8"] = benchmark.get_op_dwconv2d(ic=1, oh=56, ow=56, kh=7, kw=7, stride=1, dilation=1)
    # op_list["C9"] = benchmark.get_op_dwconv2d(ic=1, oh=7, ow=7, kh=7, kw=7, stride=1, dilation=1)
    # op_list["C10"] = benchmark.get_op_dwconv2d(ic=1, oh=56, ow=56, kh=51, kw=51, stride=1, dilation=1)
    # op_list["C11"] = benchmark.get_op_dwconv2d(ic=1, oh=7, ow=7, kh=13, kw=13, stride=1, dilation=1)
    # op_list["C12"] = benchmark.get_op_dwconv2d(ic=1, oh=28, ow=28, kh=3, kw=3, stride=1, dilation=2)
    # op_list["C13"] = benchmark.get_op_dwconv2d(ic=1, oh=28, ow=28, kh=5, kw=5, stride=1, dilation=2)
    # op_list["C14"] = benchmark.get_op_dwconv2d(b=1, oc=1, ic=1, oh=28, ow=28, kh=5, kw=5, stride=1, dilation=2)
    # op_list["C15"] = benchmark.get_op_dwconv3d(ic=1, ox=28, oy=28, oz=28, kx=5, ky=5, kz=5, stride=1)

    # op_list["test"] = benchmark.get_op_dwconv2d(ic=1, oh=16, ow=16, kh=3, kw=3, stride=1, dilation=1)
    run_op_list(op_list)
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
    print("\nCumulative Execution Times:")
    for func_name, total_time in execution_times.items():
        print(f"{func_name}: {total_time:.4f} seconds")