import islpy as isl
import numpy as np
from queue import Queue
from functools import partial
from dataclasses import dataclass

from sympy import Matrix, lcm, nsimplify
import sympy
import inv
from typing import List
import utils
import time
import cProfile
import pstats


def add_constraints_exclude_null_space(base, exclude_null_space_of):
    if exclude_null_space_of is None:
        return base
    assert isinstance(exclude_null_space_of, sympy.Matrix), f"{type(exclude_null_space_of)=}"
    if exclude_null_space_of.rows==0:
        return base

    assert isinstance(base, isl.Set), f"{type(base)=}"

    row, col = exclude_null_space_of.shape
    new_base = None
    for row_idx in range(exclude_null_space_of.rows):
        cons_pos = isl.Constraint.inequality_alloc(base.get_space())
        cons_neg = isl.Constraint.inequality_alloc(base.get_space())
        for col_idx in range(exclude_null_space_of.cols):
            entry = int(exclude_null_space_of[row_idx, col_idx])
            cons_pos = cons_pos.set_coefficient_val(isl.dim_type.set, col_idx, entry)
            cons_neg = cons_neg.set_coefficient_val(isl.dim_type.set, col_idx, -entry)
        
        cons_pos = cons_pos.set_constant_val(isl.Val(-1))
        cons_neg = cons_neg.set_constant_val(isl.Val(-1))

        base_pos = base.add_constraint(cons_pos)
        base_neg = base.add_constraint(cons_neg)
        

        if new_base is None:
            new_base = base_pos.union(base_neg)
        else:
            new_base = new_base.union(base_pos)
            new_base = new_base.union(base_neg)

    # import pdb; pdb.set_trace()
    

    # row_sums = sympy.ones(1, row) * exclude_null_space_of
    # cons_non_zero = isl.Constraint.inequality_alloc(base.get_space())
    # for col_idx in range(exclude_null_space_of.cols):
    #     entry = int(row_sums[col_idx])
    #     cons_non_zero = cons_non_zero.set_coefficient_val(isl.dim_type.set, col_idx, entry)
    # cons_non_zero = cons_non_zero.set_constant_val(isl.Val(-1))
    # print(f"{row_sums=}")
    # print(f"{cons_non_zero=}")
    # import pdb; pdb.set_trace()
    # base = base.add_constraint(cons_non_zero)

    
    return new_base

def add_constraints_positive_first_nonzero(set_):
    """
    a_0 >= 1 or 
    (a_0=0 and a_1>=1) or
    (a_0=0 and a_1=0 and a_2>=1) or
    ......
    """
    n_dim = set_.dim(isl.dim_type.set)

    cons_eq_zero_list = []
    for j in range(0,n_dim):
        cons = isl.Constraint.equality_alloc(set_.get_space())
        cons = cons.set_coefficient_val(isl.dim_type.set, j, 1)
        cons_eq_zero_list.append(cons)

    new_set = None
    for i in range(n_dim):
        
        cons = isl.Constraint.inequality_alloc(set_.get_space())
        cons = cons.set_coefficient_val(isl.dim_type.set, i, 1)
        cons = cons.set_constant_val(isl.Val(-1))

        current_set = set_.add_constraints([*cons_eq_zero_list[:i], cons])
        if new_set is None:
            new_set = current_set
        else:
            new_set = new_set.union(current_set)

    return new_set

cache_for_find_base = {}
def _find_base_cache(n_dim, dim_sizes, min_reuse_factor, hyperplanes):
    global cache_for_find_base
    key = (n_dim, dim_sizes, min_reuse_factor, hyperplanes)
    if key in cache_for_find_base:
        return cache_for_find_base[key]
    
    dims = [f"j{i}" for i in range(n_dim)]
    dims_str = ",".join(dims)
    base = isl.Set(f"{{ [{dims_str}] }}")
    
    for hyperplane in hyperplanes:
        assert type(hyperplane)==tuple, f"{hyperplane=}"
        assert len(hyperplane)==n_dim, f"{len(hyperplane)=}"
        cons = isl.Constraint.equality_alloc(base.get_space())
        for i,coef in enumerate(hyperplane):
            cons = cons.set_coefficient_val(isl.dim_type.set, i, coef)
        base = base.add_constraint(cons)

    for i in range(n_dim):
        bound = dim_sizes[i] // min_reuse_factor

        cons_lb = isl.Constraint.inequality_alloc(base.get_space())
        cons_lb = cons_lb.set_coefficient_val(isl.dim_type.set, i, 1)
        cons_lb = cons_lb.set_constant_val(isl.Val(bound))
        base = base.add_constraint(cons_lb)

        cons_ub = isl.Constraint.inequality_alloc(base.get_space())
        cons_ub = cons_ub.set_coefficient_val(isl.dim_type.set, i, -1)
        cons_ub = cons_ub.set_constant_val(isl.Val(bound))
        base = base.add_constraint(cons_ub)
    
    base = add_constraints_positive_first_nonzero(base)
    cache_for_find_base[key] = base
    return base

second_array=False
call_find_base_times = 0
def find_base(n_dim, dim_sizes, min_reuse_factor, hyperplanes, exclude_null_space_of):
    global second_array
    global call_find_base_times
    call_find_base_times += 1

    base = _find_base_cache(n_dim, dim_sizes, min_reuse_factor, hyperplanes)
    base = add_constraints_exclude_null_space(base, exclude_null_space_of)

    return base

def empty_or_zero_point(set_):
    return int(str(set_.count_val())) <= 1

def find_base_with_max_reuse(n_dim, dim_sizes, max_reuse_factor, hyperplanes, exclude_null_space_of=None):
    global second_array
    """
    try reuse factor from 1 to max_reuse_factor, 
    """

    # binary search to find the largest reuse factor
    low = 1
    high = max_reuse_factor
    base = None
    
    while low < high:
        mid = (low + high + 1) // 2
        base = find_base(n_dim, dim_sizes, mid, hyperplanes, exclude_null_space_of)
        if empty_or_zero_point(base):
            high = mid - 1
        else:
            low = mid

    base = find_base(n_dim, dim_sizes, low, hyperplanes, exclude_null_space_of)
    # if second_array:
    #     import pdb; pdb.set_trace()

    if base is not None and empty_or_zero_point(base):
        base = None

    return base, low

@dataclass
class SearchStatus:
    bases: Matrix
    max_reuse: int

def find_bases_with_max_reuse(n_dim, dim_sizes, max_reuse_factor, hyperplanes, pre_bases = Matrix([[]])):
    global second_array
    result = []
    queue = Queue()
    
    if pre_bases.rows > 0:
        orth_subspace = orthogonal_sub_space(pre_bases)
    else:
        orth_subspace = None

    bases, reuse = find_base_with_max_reuse(n_dim, dim_sizes, max_reuse_factor, hyperplanes, orth_subspace)

    if bases is None:
        return result

    for base in foreach_nontrival_point(bases):
        search_status = SearchStatus(bases=pre_bases.col_join(Matrix([base])), max_reuse=reuse)
        queue.put(search_status)

    while not queue.empty():
        search_status = queue.get()
        if search_status.bases.rows == n_dim:
            result.append(search_status)
            continue
        
        subspace = orthogonal_sub_space(search_status.bases)
        if subspace.rows == 0:
            result.append(search_status)
            continue
        
        new_bases, new_reuse = find_base_with_max_reuse(
            n_dim, 
            dim_sizes, 
            max_reuse_factor // search_status.max_reuse , 
            hyperplanes,
            exclude_null_space_of=subspace
        )
        if new_bases is None:
            result.append(search_status)
            continue
        
        for new_base in foreach_nontrival_point(new_bases):
            new_search_status = SearchStatus(
                bases=search_status.bases.col_join(Matrix([new_base])),
                max_reuse=search_status.max_reuse * new_reuse
            )
            queue.put(new_search_status)
        

    return result

@dataclass
class MultiArraySearchStatus:
    search_status_per_array: List[SearchStatus]

def concat_bases(ma_search_status):
    search_status_per_array = ma_search_status.search_status_per_array
    base = search_status_per_array[0].bases
    for i in range(1, len(search_status_per_array)):
        base = base.col_join(search_status_per_array[i].bases)
    return base

def find_bases_for_multi_array_reuse(n_dim, n_array, dim_sizes, max_reuse_factor_for_arrays, hyperplanes_for_arrays):
    global second_array
    assert type(max_reuse_factor_for_arrays)==tuple
    assert len(max_reuse_factor_for_arrays)==n_array
    assert type(hyperplanes_for_arrays)==tuple
    assert len(hyperplanes_for_arrays)==n_array
    assert type(dim_sizes)==tuple
    assert len(dim_sizes)==n_dim

    ma_result = []

    array_id = 0
    max_reuse_factor = max_reuse_factor_for_arrays[array_id]
    hyperplanes = hyperplanes_for_arrays[array_id]
    result = find_bases_with_max_reuse(n_dim, dim_sizes, max_reuse_factor, hyperplanes)
    
    for search_status in result:
        ma_search_status = MultiArraySearchStatus(
            search_status_per_array=[search_status]
        )
        ma_result.append(ma_search_status)

    second_array = True
    for array_id in range(1, n_array):
        new_ma_result = []

        max_reuse_factor = max_reuse_factor_for_arrays[array_id]
        hyperplanes = hyperplanes_for_arrays[array_id]
        for ma_search_status in ma_result:
            pre_bases = concat_bases(ma_search_status)
            result = find_bases_with_max_reuse(n_dim, dim_sizes, max_reuse_factor, hyperplanes, pre_bases)
            # import pdb; pdb.set_trace()
            for search_status in result:
                new_ma_search_status = MultiArraySearchStatus(
                    search_status_per_array=[*ma_search_status.search_status_per_array, search_status]
                )
                new_ma_result.append(new_ma_search_status)
        
        ma_result = new_ma_result

    return ma_result



def record_points(point, record):
    multi_val = point.get_multi_val()
    if multi_val.is_zero():
        return
    val = [int(str(multi_val.get_val(i))) for i in range(len(multi_val))] 
    record.append(val)

def foreach_nontrival_point(set_):
    points = []
    record_points_fn = partial(record_points, record=points)
    set_.foreach_point(record_points_fn)
    return points

def orthogonal_sub_space(A):
    n = A.cols
    At = A.transpose()
    B = A * At
    
    B_inv = B.inv()
    C = At * B_inv * A
    D = Matrix.eye(n) - C

    E = inv.scale_to_integer(D)
    F = inv.find_independent_rows(E)

    return F

def base_to_coor_transform_matrix(bases_matrix):
    assert type(bases_matrix)==Matrix, f"{type(bases_matrix)=}"
    bases_matrix = bases_matrix.transpose()
    coor_transform_matrix = bases_matrix.inv()
    coor_transform_matrix = inv.scale_to_integer(coor_transform_matrix)
    return coor_transform_matrix

def matrix_to_schedule(coor_transform_matrix):
    domain_iters_def = ", ".join([f"i{i}" for i in range(coor_transform_matrix.rows)])
    range_iters_def = ", ".join([f"o{i}" for i in range(coor_transform_matrix.rows)])
    schedule = isl.Map(f"{{ [{domain_iters_def}] -> [{range_iters_def}] }}")

    for row_idx in range(coor_transform_matrix.rows):
        cons = isl.Constraint.equality_alloc(schedule.get_space())
        for col_idx in range(coor_transform_matrix.cols):
            coor = coor_transform_matrix[row_idx, col_idx]
            cons = cons.set_coefficient_val(isl.dim_type.in_, col_idx, coor)

        cons = cons.set_coefficient_val(isl.dim_type.out, row_idx, -1)
        schedule = schedule.add_constraint(cons)
    return schedule

def base_to_coor_transform_schedule(bases_matrix):
    # assert bases_matrix.rows==len(iter_names), f"{bases_matrix.rows=}, {len(iter_names)=}"
    # print(f"{bases_matrix=}")
    coor_transform_matrix = base_to_coor_transform_matrix(bases_matrix)
    schedule = matrix_to_schedule(coor_transform_matrix)
    return schedule

def batch_base_to_coor_transform_schedule(bases_matrix_list):
    schedule_list = []
    for bases_matrix in bases_matrix_list:
        schedule = base_to_coor_transform_schedule(bases_matrix)
        schedule_list.append(schedule)
    return schedule_list

def padding_for_base_matrix(base_matrix):
    n_rows = base_matrix.rows
    n_cols = base_matrix.cols
    if n_rows == n_cols:
        return base_matrix
    assert n_rows < n_cols, f"{n_rows=}, {n_cols=}"
    I = Matrix.eye(n_cols)
    base_matrix = base_matrix.col_join(I)
    base_matrix_ = inv.find_independent_rows(base_matrix)
    return base_matrix_

def find_schedules_for_multi_array_reuse(n_dim, n_array, dim_sizes, max_reuse_factor_for_arrays, hyperplanes_for_arrays):
    result = find_bases_for_multi_array_reuse(
        n_dim=n_dim,
        n_array=n_array, 
        dim_sizes=dim_sizes, 
        max_reuse_factor_for_arrays=max_reuse_factor_for_arrays, 
        hyperplanes_for_arrays=hyperplanes_for_arrays
    )
    base_matrixs = [ma_search_status.search_status_per_array[-1].bases for ma_search_status in result]
    base_matrixs = [padding_for_base_matrix(base_matrix) for base_matrix in base_matrixs]
    schedules = batch_base_to_coor_transform_schedule(base_matrixs)
    return schedules

def parse_operator(domain, access_relations, max_reuse_factor_for_arrays):
    """
    n_dim, n_array, dim_sizes, max_reuse_factor_for_arrays, hyperplanes_for_arrays
    """
    n_dim = domain.dim(isl.dim_type.set)
    n_array = len(access_relations)
    dim_sizes = [domain.dim_max_val(i).get_num_si() + 1 for i in range(n_dim)]
    hyperplanes_for_arrays = []
    for access_relation in access_relations:
        assert access_relation.dim(isl.dim_type.in_) == n_dim, f"{access_relation.dim(isl.dim_type.in_)=}, {n_dim=}"
        pma = access_relation.as_pw_multi_aff()
        assert pma.n_piece()==1, f"{len(mas)=}"

        ma = pma.as_multi_aff()
        hyperplanes = []
        print(f"{ma=}, {ma.n_piece()=}")
        for aff_idx in range(ma.get_list().n_aff()):
            aff = ma.get_aff(aff_idx)
            coef = []
            for i in range(aff.dim(isl.dim_type.in_)):
                coef.append(int(str(aff.get_coefficient_val(isl.dim_type.in_, i))))
            hyperplanes.append(tuple(coef))
        hyperplanes_for_arrays.append(tuple(hyperplanes))
    return n_dim, n_array, tuple(dim_sizes), tuple(max_reuse_factor_for_arrays), tuple(hyperplanes_for_arrays)

def find_schedules_for_operator(domain, access_relations, max_reuse_factor_for_arrays):
    n_dim, n_array, dim_sizes, max_reuse_factor_for_arrays, hyperplanes_for_arrays = parse_operator(
        domain=domain,
        access_relations=access_relations,
        max_reuse_factor_for_arrays=max_reuse_factor_for_arrays
    )
    schedules = find_schedules_for_multi_array_reuse(
        n_dim=n_dim,
        n_array=n_array, 
        dim_sizes=dim_sizes, 
        max_reuse_factor_for_arrays=max_reuse_factor_for_arrays, 
        hyperplanes_for_arrays=hyperplanes_for_arrays
    )
    return schedules

def main():

    # conv2d
    domain = isl.BasicSet("{ S[oh,ow,kh,kw]: 0<=oh<64 and 0<=ow<64 and 0<=kh<3 and 0<=kw<3 }")
    access_I = isl.BasicMap("{ S[oh,ow,kh,kw] -> I[oh + kh, ow + kw] }")
    access_O = isl.BasicMap("{ S[oh,ow,kh,kw] -> O[oh, ow] }")
    tile_transform = isl.BasicMap("{ S[oh,ow,kh,kw] -> S[floor(oh/2),oh%2,ow, kh,kw] }")
    domain = tile_transform.intersect_domain(domain).range()
    access_I = tile_transform.reverse().apply_range(access_I)#.intersect_domain(domain)
    access_O = tile_transform.reverse().apply_range(access_O)#.intersect_domain(domain)
    access_I = utils.simplify_basic_map(access_I)
    access_O = utils.simplify_basic_map(access_O)
    access_I = isl.BasicMap("{ S[i0, i1, i2, i3, i4] -> I[o0, o1] : o0 = 2i0 + i1 + i3 and o1 = i2 + i4}")
    access_O = isl.BasicMap("{ S[i0, i1, i2, i3, i4] -> O[o0, o1] : o0 = 2i0 + i1 and o1 = i2}")
    # print(f"{domain=}")
    # print(f"{access_I=}")
    # print(f"{access_O=}")
    # print(f"{access_O.universe(access_O.get_space())=}")
    # exit()
    begin_time = time.time()
    schedules = find_schedules_for_operator(
        domain=domain,
        access_relations=[access_I, access_O],
        max_reuse_factor_for_arrays=(8, 8)
    )
    end_time = time.time()
    print(f"{len(schedules)=}")
    print(f"Duration: {end_time-begin_time} s")
    print(f"{call_find_base_times=}")
    return

    # result = find_bases_for_multi_array_reuse(
    #     n_dim=4,
    #     n_array=2, 
    #     dim_sizes=[64,64,4,4], 
    #     max_reuse_factor_for_arrays=[16, 16], 
    #     hyperplanes_for_arrays=[
    #         [[0,1,0,1],[1,0,1,0]],
    #         [[1,0,0,0],[0,1,0,0]]
    #     ]
    # )
    # print(f"{len(result)=}")
    # for idx,ma_search_status in enumerate(result):
    #     print(f"\nSearch result {idx}")
    #     assert len(ma_search_status.search_status_per_array) <= 2, f"{len(ma_search_status.search_status_per_array)=}"
    #     for array_id, search_result in enumerate(ma_search_status.search_status_per_array):
    #         # assert array_id < 2, 
    #         print(f"- {array_id = }")
    #         print(f"  - {search_result.bases = }")
    #         print(f"  - {search_result.max_reuse = }")

    # exit()

    # result = find_bases_with_max_reuse(
    #     n_dim=4,
    #     dim_sizes=[64,64,4,4],
    #     max_reuse_factor=32,
    #     hyperplanes=[
    #         [0,1,0,1],
    #         [1,0,1,0]
    #     ]
    # )
    
    # for idx,search_status in enumerate(result):
    #     print(f"---------- {idx} ----------")
    #     print(f"{search_status.bases=}")
    #     print(f"{search_status.max_reuse=}")
    
    # print("--------------------")
    # print(len(result))
    # exit()


    bases = find_base(
        n_dim=4,
        min_reuse_factor=2,
        dim_sizes=[64,64,3,3],
        hyperplanes=[
            [0,2,0,1],
            [2,0,1,0]
        ],
        exclude_null_space_of=None
    )
    print(bases)
    bases.foreach_point(print)
    exit()

    bases = foreach_nontrival_point(bases)
    # bases = Matrix(bases)
    # print(f"{bases=}")
    # bases = inv.find_independent_rows(bases)
    # print(f"{bases=}")
    # exit()
    current_bases =  Matrix([[]])
    base = Matrix([[-1, 0, 1, 0]])
    current_bases = current_bases.col_join(base)
    # print(current_bases)

    subspace = orthogonal_sub_space(current_bases)
    print(f"{subspace=}")

    base2 = find_base(
        n_dim=4,
        min_reuse_factor=4,
        dim_sizes=[64,64,4,4],
        hyperplanes=[
            [0,1,0,1],
            [1,0,1,0]
        ],
        exclude_null_space_of=subspace
    )
    # print(f"{foreach_nontrival_point(base2)=}")
    print("-------------------")
    base2.foreach_point(print)
    


if __name__ == "__main__":
    main()