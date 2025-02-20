import json
import islpy as isl
from typing import Optional
from polycim.passes.base import Schedule
from polycim.passes.base import SchedulePass
from polycim.passes.base import SchedulePassResult
from sympy import Matrix
from polycim.passes.loop_padding import loop_padding_to_box_all, shift_to_zero

import polycim.utils.utils as utils
from polycim.passes.multi_level_tiling import (
    remove_all_one_factors,
    combine_tilesize_by_symmetry_info,
    multi_level_splitting_var_level,
)
from polycim.utils.math import factorize
from polycim.passes.affine_transform import (
    parse_operator,
    find_base,
    base_to_coor_transform_schedule,
    shift_to_positive
)
from functools import partial
import itertools
import math
import time
import numpy as np
from functools import reduce
from collections import OrderedDict
from polycim.depth_first.pipeline import Base

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

def cover(corr1, corr2):
    nonzero1 = [int(i!=0) for i in corr1]
    nonzero2 = [int(i!=0) for i in corr2]
    return all(i1 >= i2 for i1,i2 in zip(nonzero1, nonzero2)) and any(i1 > i2 for i1,i2 in zip(nonzero1, nonzero2))


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

def get_rank_key(base_combination):
    num_non_zero = 0
    for base in base_combination:
        if base.reuse_array_id in [0,1]:
            nonzero = [int(i!=0) for i in base.corrdinate]
            num_non_zero += sum(nonzero)
    return num_non_zero

def select_bases(bases, operator):
    n_dim = operator.domain.dim(isl.dim_type.set)
    selected_bases = []
    
    # Find all combinations of n_dim bases
    base_combinations = itertools.combinations(bases, n_dim)
    
    # Start timing
    start_time = time.time()

    for i,combination in enumerate(base_combinations):
        if satisfies_constraints(combination, operator):
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

def satisfies_constraints(combination, operator):
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
    tile_sizes = operator.attr["pre_tile_sizes"]

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

def filter_tile_direction_points(points, tile_sizes):

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

def extend_scalar_dim_for_operator(operator):
    n_dim = operator.domain.dim(isl.dim_type.set)
    origin_dims = [f"i{i}" for i in range(n_dim)]
    new_dims = ["0", *origin_dims]
    origin_dims_str = ",".join(origin_dims)
    new_dims_str = ",".join(new_dims)
    extend_dim_scheduel = isl.BasicMap(f"{{ [{origin_dims_str}] -> [{new_dims_str}] }}")
    new_op = operator.apply_schedule(extend_dim_scheduel, name="extend_dim")
    return new_op

def extend_scalar_dim_for_bases(bases, reuse_array_id):
    assert reuse_array_id in (0,1)
    n_dim = len(bases[0].corrdinate)
    new_bases = [Base([1, *([0] * n_dim)], reuse_array_id, True)]
    for base in bases:
        new_corrdinate = tuple([0] + list(base.corrdinate))
        new_bases.append(Base(new_corrdinate, base.reuse_array_id, base.is_trival))
    return new_bases

def create_scalar_axis_for_reuse(bases, operator):
    reuse_cnt = {
        0:len([base for base in bases if base.reuse_array_id == 0 and not base.is_skewed]), 
        1:len([base for base in bases if base.reuse_array_id == 1 and not base.is_skewed])
    }

    if reuse_cnt[0] == 0:
        operator = extend_scalar_dim_for_operator(operator)
        bases = extend_scalar_dim_for_bases(bases, 0)
        operator.set_attr("pre_tile_sizes", ((1,), *operator.attr["pre_tile_sizes"]), overwrite=True)
        operator.set_attr("dim_types", ["_"] + operator.attr["dim_types"], overwrite=True)

    if reuse_cnt[1] == 0:
        operator = extend_scalar_dim_for_operator(operator)
        bases = extend_scalar_dim_for_bases(bases, 1)
        operator.set_attr("pre_tile_sizes", ((1,), *operator.attr["pre_tile_sizes"]), overwrite=True)
        operator.set_attr("dim_types", ["_"] + operator.attr["dim_types"], overwrite=True)

    return bases, operator

class AffineSchedule(Schedule):
    def __init__(self, bases):
        super().__init__()
        self.bases = bases
    
    def dumps(self):
        result = []
        for base in self.bases:
            result.append({
                "corrdinate":base.corrdinate,
                "reuse_array_id":base.reuse_array_id,
                "is_trival":base.is_trival
            })
        return json.dumps(result)

    def parse(self, data):
        assert isinstance(data, list)
        self.bases = []
        for item in data:
            self.bases.append(Base(item["corrdinate"], item["reuse_array_id"], item["is_trival"]))

class AffinePass(SchedulePass):
    def __init__(self, 
            args,
            fix_schedule: Optional[AffineSchedule]=None, 
            schedule_as_key: bool=False,
            pad: bool=True,
        ):
        super().__init__(
            fix_schedule=fix_schedule, 
            schedule_as_key=schedule_as_key
        )
        self.args = args
        self.pad = pad
        assert self.fix_schedule is None or isinstance(self.fix_schedule, AffineSchedule)

    def base_construction(self, operator):
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
            points = filter_tile_direction_points(points, operator.attr["pre_tile_sizes"])
            points = filter_cover_points(points)
            points = filter_many_direction_points(points)
            # TODO: filter points
            for point in points:
                is_trival = is_base_trival(point, operator.attr["dim_types"])
                base = Base(point, array_id, is_trival)
                bases[base] = None

        for i in range(n_dim):
            point = [0 for d in range(n_dim)]
            point[i] = 1
            base = Base(tuple(point), -1, True)
            bases[base] = None

        bases = list(bases.keys())

        return bases

    def filter_bases(self, bases, operator):
        if self.args.disable_affine:
            new_bases = []
            for base in bases:
                num_nonzero = sum([int(i!=0) for i in base.corrdinate])
                if num_nonzero == 1:
                    new_bases.append(base)

            return new_bases
        return bases

    def apply(self, operator):
        
        if self.fix_schedule is None:
            # 1. base construction
            bases = self.base_construction(operator)  

            bases = self.filter_bases(bases, operator)

            bases, operator = create_scalar_axis_for_reuse(bases, operator)

            # 2. base selection
            # select n_dim base from bases that satisfy:
            #  constraint1: for each array, at least one base is selected to reuse it.
            #  constraint2: the selected bases are linear independent
            # find all selection combinations
            selected_bases_list = select_bases(bases, operator)
            
        else:
            selected_bases_list = [self.fix_schedule.bases]

        # 3. affine transform
        result_list = []
        for selected_bases in selected_bases_list:
            rev_selected_bases = tuple(list(selected_bases)[::-1])
            corrdinates = [base.corrdinate for base in rev_selected_bases]
            matrix = Matrix(corrdinates)

            schedule = base_to_coor_transform_schedule(matrix)

            new_op = operator.apply_schedule(schedule, name="affine")
            # print(new_op.attr.get("affine::bases", "None"))
            # import pdb; pdb.set_trace()
            new_op = shift_to_positive(new_op)
            if self.pad:
                new_op = loop_padding_to_box_all(new_op)

            base_str = ""
            for base in selected_bases:
                base_str += str(base) + "\n"
            new_op.history_schedules.append({"bases":base_str})
            new_op.history_schedules.append({"matrix":matrix})
            
            
            
            new_op.set_attr("affine::bases", selected_bases)
            
            affine_schedule = AffineSchedule(selected_bases)
            result = SchedulePassResult(new_op, affine_schedule)
            result_list.append(result)

        
        return result_list