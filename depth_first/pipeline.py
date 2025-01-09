from multi_level_tiling import enumerate_tiling_factors
import benchmark
from multi_level_tiling import (
    factorize, 
    multi_level_splitting_var_level, 
    combine_tilesize_by_symmetry_info
)
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
from draw import (
    draw,
    extract_frame_info
)
from depth_first.count_minimal_macro import count_minimal_needed_macro
from base_operator import BasicOperator
import concurrent.futures
import datetime
import os
from depth_first.timeout import timeout
from functools import reduce
import math
from loop_padding import loop_padding_to_box_all, shift_to_zero

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
    # return tqdm(ls, desc=desc)
    return ls

@timing_decorator
def pre_tiling(operator, symmetry_info=None):
    domain = operator.domain
    tiling_factor = 2

    assert domain.is_box(), f"domain={domain} is not box"
    n_iter = domain.dim(isl.dim_type.set)

    domain_shape = utils.get_static_box_shape(domain)
    dim_factors = []
    for dim_size in domain_shape:
        factors = factorize(dim_size, tiling_factor)
        factors = [tuple(factor) for factor in factors if factor[0]!=1 and factor[1]!=1]
        factors = reversed(factors)
        factors = [*factors,(dim_size,)]
        dim_factors.append(tuple(factors))
    dim_factors = tuple(dim_factors)

    if symmetry_info is None:
        combination_list = list(itertools.product(*dim_factors))
    else:
        combination_list = combine_tilesize_by_symmetry_info(dim_factors, symmetry_info)

    new_combination_list = []
    for idx,combination in enumerate(combination_list):
        new_operator = multi_level_splitting_var_level(operator, combination)
        new_combination_list.append((new_operator, combination))

    return new_combination_list
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
def affine_transform(operator, **kwargs):
    # 1. base construction
    bases = base_construction(operator, **kwargs)  
    
    # 2. base selection
    # select n_dim base from bases that satisfy:
    #  constraint1: for each array, at least one base is selected to reuse it.
    #  constraint2: the selected bases are linear independent
    # find all selection combinations
    selected_bases_list = select_bases(bases, operator)

    # 3. affine transform
    new_op_list = []
    for selected_bases in selected_bases_list:
        rev_selected_bases = tuple(list(selected_bases)[::-1])
        matrix = Matrix([base.corrdinate for base in rev_selected_bases])

        schedule = base_to_coor_transform_schedule(matrix)

        new_op = operator.apply_schedule(schedule, name="affine")
        new_op = shift_to_positive(new_op)
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
    # print(f"{mapping=}")
    coalescing_schedule = get_schedule_from_mapping(mapping, operator)
    # print(f"{coalescing_schedule=}")
    tiling_factor = [
        cim_cfg.n_comp, cim_cfg.n_group_vcol
    ]
    tiling_schedule = _get_hardware_tiling_schedule(coalescing_schedule.range().dim(isl.dim_type.set), tiling_factor)
    # print(f"{tiling_schedule=}")
    if return_schedule:
        return [(coalescing_schedule, tiling_schedule)]
    else:
        new_op = operator.apply_schedule(coalescing_schedule, skip_simplify=True, name="coalescing")
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


@timeout(seconds=100)
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
        self.cim_cfg = cim_cfg
        self.pad_count = pad_count
        self.delay_apply = delay_apply
        self.num_macros = num_macros
        self.enable_weight_rewrite = enable_weight_rewrite

    def search(self, op, **kwargs):
        result = []
        min_compute_times = int(1e9)
        min_compute_op = None
        stats = {"count_val": list()}
        for op1,tile_sizes in get_tqdm(self.pre_tiling(op, symmetry_info=kwargs.get("symmetry_info", None)), desc="pre_tiling"):            
            # print("a")
            begin_time = time.time()
            for op2,bases in get_tqdm(self.affine_transform(op1, tile_sizes=tile_sizes), desc="affine_transform"):

                
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
                        else:
                            assert isinstance(exe_time, int)
                    
                    if exe_time is not None and exe_time < min_compute_times:
                        min_compute_times = exe_time
                        if self.delay_apply:
                            op3 = op2.apply_schedule(coalescing_schedule, skip_simplify=True, name="coalescing")
                            op3 = op3.apply_schedule(tiling_schedule, skip_simplify=True, name="tiling")
                        min_compute_op = op3
                        print(f"min_compute_times={min_compute_times}")
                
            end_time = time.time()
            
            # dump_schedules(min_compute_op)
            # print(f"time={end_time - begin_time}")
            # print("\n")
            # print(f"min_compute_times={min_compute_times}")
            # draw(min_compute_op, self.cim_cfg)
            # exit()
        return min_compute_times, min_compute_op, stats
        

    def pre_tiling(self, op, symmetry_info):
        return pre_tiling(op, symmetry_info=symmetry_info)
        # return [op]

    def affine_transform(self, op, **kwargs):
        return affine_transform(op, **kwargs)

    def coalesce_and_tiling(self, op, bases, return_schedule=False):
        return coalesce_and_tiling(op, bases, self.cim_cfg, return_schedule)

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

dump_index = 0
def dump_schedules(op, **kwargs):
    global dump_index
    schedule_dict = OrderedDict()
    schedule_dict["pre_tiling"] = None
    schedule_dict["affine"] = None
    schedule_dict["shift_to_positive"] = None
    schedule_dict["coalescing"] = None
    schedule_dict["tiling"] = None
    for name_schedule in op.history_schedules:
        if type(name_schedule) == dict and list(name_schedule.keys())[0] in schedule_dict:
            name = list(name_schedule.keys())[0]
            schedule = name_schedule[name]
            schedule_dict[name] = str(schedule)
    
    init_domain = op.history_domains[0]
    dump_code = "\"\"\"\n"
    for key,value in kwargs.items():
        dump_code += f"{key} = {value}\n"
    dump_code += "\"\"\"\n"
    dump_code += f"import islpy as isl\n"
    dump_code += f"import time\n"
    dump_code += f"domain = isl.BasicSet(\"{init_domain}\")\n\n"
    for key, value in schedule_dict.items():
        dump_code += f"schedule_{key} = isl.BasicMap(\"{value}\")\n"
        dump_code += f"domain = schedule_{key}.intersect_domain(domain).range()\n\n"

    dump_code += """
n_dim = domain.dim(isl.dim_type.set)
begin_time = time.time()
outer_domain = domain.project_out(isl.dim_type.set, n_dim - 2, 2)
val = outer_domain.count_val()
dur_time = time.time() - begin_time
print(f"outer_domain.count_val {val=}, {dur_time=}")
    """
    save_dir = "dump_code"
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f"dump_code_{dump_index}.py"), "w") as f:
        f.write(dump_code)
    dump_index += 1
    print("dump_code saved to dump_code.py")
    # exit()
    return dump_code

def run_op_list(op_list, save_dir, pad_count, delay_apply, num_macros, enable_weight_rewrite):
    os.makedirs(save_dir, exist_ok=True)
    cim_cfg = get_config()
    search_space = SearchSpace(cim_cfg, 
                              pad_count=pad_count, 
                              delay_apply=delay_apply,
                              num_macros=num_macros,
                              enable_weight_rewrite=enable_weight_rewrite)
    for name,op in op_list.items():
        if isinstance(op, tuple):
            op, symmetry_info = op
        else:
            symmetry_info = None
        print(f"{name=}")
        min_compute_times, min_compute_op, stats = search_space.search(op, symmetry_info=symmetry_info)
        flops = int(str(op.domain.count_val()))
        show_result(min_compute_times, cim_cfg, flops)
        print("\n")
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

    op_list = OrderedDict()
    op_list["C1"] = benchmark.get_op_dwconv2d(ic=1, oh=112, ow=112, kh=3, kw=3, stride=1, dilation=1)
    op_list["C2"] = benchmark.get_op_dwconv2d(ic=1, oh=56, ow=56, kh=3, kw=3, stride=1, dilation=1)
    op_list["C3"] = benchmark.get_op_dwconv2d(ic=1, oh=28, ow=28, kh=5, kw=5, stride=1, dilation=1)
    op_list["C4"] = benchmark.get_op_dwconv2d(ic=1, oh=14, ow=14, kh=3, kw=3, stride=1, dilation=1)
    op_list["C5"] = benchmark.get_op_dwconv2d(ic=1, oh=14, ow=14, kh=5, kw=5, stride=1, dilation=1)
    op_list["C6"] = benchmark.get_op_dwconv2d(ic=1, oh=7, ow=7, kh=5, kw=5, stride=1, dilation=1)
    op_list["C7"] = benchmark.get_op_dwconv2d(ic=1, oh=7, ow=7, kh=3, kw=3, stride=1, dilation=1)
    op_list["C8"] = benchmark.get_op_dwconv2d(ic=1, oh=56, ow=56, kh=7, kw=7, stride=1, dilation=1)
    op_list["C9"] = benchmark.get_op_dwconv2d(ic=1, oh=7, ow=7, kh=7, kw=7, stride=1, dilation=1)
    # op_list["C10"] = (
    #     benchmark.get_op_dwconv2d(ic=1, oh=56, ow=56, kh=51, kw=51, stride=1, dilation=1, virtual_axis=False),
    #     # symmetry_info
    #     ((1,3),(2,4))
    # )
    # op_list["C11"] = benchmark.get_op_dwconv2d(ic=1, oh=7, ow=7, kh=13, kw=13, stride=1, dilation=1)
    # op_list["C12"] = (
    #     benchmark.get_op_dwconv2d(ic=1, oh=28, ow=28, kh=3, kw=3, stride=1, dilation=2, virtual_axis=False),
    #     # symmetry_info
    #     ((1,3),(2,4))
    # )
    # op_list["C13"] = (
    #     benchmark.get_op_dwconv2d(ic=1, oh=28, ow=28, kh=5, kw=5, stride=1, dilation=2, virtual_axis=False),
    #     # symmetry_info
    #     ((1,3),(2,4))
    # )
    # # op_list["C14"] = benchmark.get_op_dwconv2d(b=1, oc=1, ic=1, oh=28, ow=28, kh=5, kw=5, stride=1, dilation=2)
    # op_list["C15"] = (
    #     benchmark.get_op_dwconv3d(ic=1, ox=28, oy=28, oz=28, kx=5, ky=5, kz=5, stride=1),
    #     # symmetry_info
    #     ((1,4),(2,5),(3,6))
    # )

    # op_list["test"] = (
    #     benchmark.get_op_dwconv2d(ic=1, oh=28, ow=28, kh=5, kw=5, stride=1, dilation=1, virtual_axis=False),
    #     # symmetry_info
    #     ((1,3),(2,4))
    # )
    # op_list["test"] = (
    #     benchmark.get_op_dwconv2d(ic=4, oh=16, ow=16, kh=3, kw=3, stride=1, dilation=1, virtual_axis=False),
    #     # None
    #     # symmetry_info
    #     ((1,3),(2,4))
    # )

    curr_time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
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

