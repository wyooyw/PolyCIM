import cProfile
import pstats
import time
from dataclasses import dataclass
from functools import partial
from queue import Queue
from typing import List

import islpy as isl
import numpy as np
import sympy
from sympy import Matrix, lcm, nsimplify

import polycim.utils.mat_utils as inv
import polycim.utils.utils as utils
from tqdm import tqdm
from polycim.op.base_operator import BasicOperator
import random

from pulp import (
    LpMaximize, LpVariable, getSolver, LpStatus, pulp, value
)
from functools import reduce

def get_areas_by_sizes(lengths):
    volumn = reduce(lambda x, y: x*y, lengths)
    areas = [volumn//size for size in lengths]
    return areas

def get_areas_by_sizes_inner(lengths):
    lengths = [max(l-1,1) for l in lengths]
    volumn = reduce(lambda x, y: x*y, lengths)
    areas = [volumn//size for size in lengths]
    return areas

def get_cross_by_sizes(lengths):
    volumn = reduce(lambda x, y: x*y, lengths)
    cross = []
    for i in range(len(lengths)):
        cross_row = []
        for j in range(len(lengths)):
            if i == j:
                cross_row.append(0)
            else:
                cross_row.append(volumn//lengths[i]//lengths[j])
        assert len(cross_row)==len(lengths)
        cross.append(cross_row)
    assert len(cross)==len(lengths)
    return cross

def make_ilp(n, hyperplanes, lengths, exclude_null_space_of):
    # areas = get_areas_by_sizes(lengths)
    areas = get_areas_by_sizes_inner(lengths)
    cross = get_cross_by_sizes(lengths)

    model = pulp.LpProblem('linear_programming', LpMaximize)
    solver = getSolver('PULP_CBC_CMD')

    x = [LpVariable(f'x{i}', cat = 'Integer') for i in range(n)]
    y = [LpVariable(f'y{i}', cat = 'Integer') for i in range(n)]

    u = [LpVariable(f'u{i}', cat = 'Integer') for i in range(n)]
    v = [LpVariable(f'v{i}', cat = 'Integer') for i in range(n)]

    a = [LpVariable(f'a{i}', cat = 'Integer') for i in range(n)]
    b = [LpVariable(f'b{i}', cat = 'Integer') for i in range(n)]
    c = [LpVariable(f'c{i}', cat = 'Integer') for i in range(n)]
    d = [LpVariable(f'd{i}', cat = 'Integer') for i in range(n)]

    if exclude_null_space_of is not None:
        row = exclude_null_space_of.rows
        z = [LpVariable(f'z{i}', cat = 'Integer') for i in range(row)]
        p = [LpVariable(f'p{i}', cat = 'Integer') for i in range(row)]
        q = [LpVariable(f'q{i}', cat = 'Integer') for i in range(row)]

    # cross term
    e = []
    # for i in range(n):
    #     e_row = []
    #     for j in range(i+1):
    #         e_row.append(None)
    #     for j in range(i+1, n):
    #         e_ij = LpVariable(f'e{i}{j}', cat = 'Integer')
    #         model += e_ij >= 0
    #         model += e_ij <= 1
    #         model += e_ij <= a[i]
    #         model += e_ij <= a[j]
    #         model += e_ij >= a[i] + a[j] - 1
    #         e_row.append(e_ij)
    #     assert len(e_row)==n
    #     e.append(e_row)
    # assert len(e)==n

    # model += sum(a) <= 2

    # declare objective
    model += -(
        sum([areas[i] * y[i] for i in range(n)]) 
        # -sum(cross[i][j] * e[i][j] for i in range(n) for j in range(i+1,n))
    )
    # import pdb; pdb.set_trace()

    # abs constraints
    # y = |x|
    M = max(lengths) + 1
    for i in range(n):
        model += x[i] <= y[i]
        model += -x[i] <= y[i]
        model += y[i] <= x[i] + (1 - u[i]) * M
        model += y[i] <= -x[i] + u[i] * M

    for i in range(n):
        model += u[i] >= 0
        model += u[i] <= 1

    # hyperplane constraints
    # hyperplane * x = 0
    for hyperplane in hyperplanes:
        model += sum([hyperplane[j] * x[j] for j in range(n)]) == 0

    # constraints for a[:]
    # a = 0 iff y = 0
    # a = 1 iff y > 0
    for i in range(n):
        model += y[i] <= a[i] * M
        model += a[i] <= y[i]

    for i in range(n):
        model += a[i] >= 0
        model += a[i] <= 1

    model += sum(a) >= 1

    # Constraints for b[:]
    # b[0] = a[0]
    # b[i] = b[i-1] + a[i]
    model += b[0] == a[0]
    for i in range(1, n):
        model += b[i] == b[i-1] + a[i]

    # Constraints for c[:]
    # c = |b-1|
    M2 = n+1
    for i in range(n):
        model += (b[i] - 1) <= c[i]
        model += -(b[i] - 1) <= c[i]
        model += c[i] <= (b[i]-1) + (1 - v[i]) * M2
        model += c[i] <= -(b[i]-1) + v[i] * M2

        model += v[i] >= 0
        model += v[i] <= 1
    
    # Constraints for d[:]
    # M2 = n + 1
    for i in range(n):
        model += c[i] <= (1 - d[i]) * M2
        model += (1-d[i]) <= c[i]

    # # Constraints between d and x
    for i in range(n):
        model += x[i] >= (d[i]-1)*M

    # Constraint for linear independent of previous solutions
    if exclude_null_space_of is not None:
        row = exclude_null_space_of.rows
        col = exclude_null_space_of.cols
        for r in range(row):
            model += sum(exclude_null_space_of[r, c] * x[c] for c in range(col)) == z[r]
            model += z[r] <= (1-p[r]) * M - 1
            model += -z[r] <= (1-q[r]) * M - 1

        for i in range(row):
            model += p[i] >= 0
            model += p[i] <= 1
            model += q[i] >= 0
            model += q[i] <= 1
        
        model += sum(p) + sum(q) >= 1

        return model, solver, {"x":x, "y":y, "a":a,"b":b,"c":c,"d":d,"u":u,"v":v, "z":z, "p":p, "q":q,"e":e}


    return model, solver, {"x":x, "y":y, "a":a,"b":b,"c":c,"d":d,"u":u,"v":v,"e":e}

def find_base(n_dim, dim_sizes, hyperplanes, exclude_null_space_of):
    model, solver, decision_vars = make_ilp(n_dim, hyperplanes, dim_sizes, exclude_null_space_of)
    # print(model)

    # solve 
    results = model.solve(solver=solver)
    

    # print results
    if LpStatus[results] == 'Optimal': 
        print('The solution is optimal.')
        # print(f'Objective value: z* = {value(model.objective)}')
        # print(f'Solution: ')
        x = decision_vars['x']
        # y = decision_vars['y']
        a = decision_vars['a']
        a = [int(value(xi)) for xi in a]
        e = decision_vars['e']
        print(f"{a=}")
        # for i in range(n_dim):
        #     for j in range(i+1, n_dim):
        #         if e[i][j] is not None: 
        #             print(f"e[{i}][{j}] = {int(value(e[i][j]))}")

        # b = decision_vars['b']
        # c = decision_vars['c']
        # d = decision_vars['d']
        # print(f"  x = {[value(xi) for xi in x]}")
        # print(f"  y = {[value(yi) for yi in y]}")
        # print(f"  a = {[value(ai) for ai in a]}")
        # print(f"  b = {[value(bi) for bi in b]}")
        # print(f"  c = {[value(ci) for ci in c]}")
        # print(f"  d = {[value(di) for di in d]}")

        ret = [int(value(xi)) for xi in x], int(value(model.objective))
        # import pdb; pdb.set_trace()
        return ret
    elif LpStatus[results] == 'Infeasible':
        print('Problem is infeasible - no solution exists')
        return None, 0
    else:
        print(f'Solver status: {LpStatus[results]}')
        return None, 0


def empty_or_zero_point(set_):
    return set_.count_val() == 0

def make_bases_to_matrix(bases):
    base_matrix = []
    for base in foreach_nontrival_point(bases):
        base_matrix.append(base)
    base_matrix = Matrix(base_matrix)
    base_matrix = base_matrix.transpose()
    return base_matrix

def find_base_with_max_reuse(
    n_dim, dim_sizes, max_reuse_factor, hyperplanes, exclude_null_space_of=None, lex_lt_set=None
):
    global second_array
    """
    try reuse factor from 1 to max_reuse_factor, 
    """

    return find_base(n_dim, dim_sizes, hyperplanes, exclude_null_space_of)


@dataclass
class SearchStatus:
    bases: Matrix
    max_reuse: int
    final_row_as_set: isl.Set

def list_to_set(ls):
    ls = [str(i) for i in ls]
    set_ = isl.Set(f"{{ [{','.join(ls)}] }}")
    return set_

def find_bases_with_max_reuse(
    n_dim, dim_sizes, max_reuse_factor, hyperplanes, pre_bases=Matrix([[]])
):
    global second_array
    result = []
    queue = Queue()

    if pre_bases.rows > 0:
        orth_subspace = orthogonal_sub_space(pre_bases)
    else:
        orth_subspace = None
    
    base, reuse = find_base_with_max_reuse(
        n_dim, dim_sizes, max_reuse_factor, hyperplanes, orth_subspace
    )
    # import pdb; pdb.set_trace()
    
    # import pdb; pdb.set_trace()
    if base is None:
        return result

    # for base in foreach_nontrival_point(bases):
    search_status = SearchStatus(
        bases=pre_bases.col_join(Matrix([base])), 
        max_reuse=reuse,
        final_row_as_set=list_to_set(base)
    )
    queue.put(search_status)

    while not queue.empty():
        search_status = queue.get()
        result.append(search_status)
        
        if search_status.bases.rows == n_dim:
            continue

        subspace = orthogonal_sub_space(search_status.bases)
        if subspace.rows == 0:
            continue

        new_base, new_reuse = find_base_with_max_reuse(
            n_dim,
            dim_sizes,
            max_reuse_factor, #// search_status.max_reuse,
            hyperplanes,
            subspace,
            # lex_lt_set=search_status.final_row_as_set,
        )
        if new_base is None:
            continue

        # for new_base in foreach_nontrival_point(new_bases):
        new_search_status = SearchStatus(
            bases=search_status.bases.col_join(Matrix([new_base])),
            max_reuse=search_status.max_reuse * new_reuse,
            final_row_as_set=list_to_set(new_base),
        )
        queue.put(new_search_status)
    # import pdb; pdb.set_trace()
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


def find_bases_for_multi_array_reuse(
    n_dim, n_array, dim_sizes, max_reuse_factor_for_arrays, hyperplanes_for_arrays
):
    global second_array
    assert type(max_reuse_factor_for_arrays) == tuple
    assert len(max_reuse_factor_for_arrays) == n_array
    assert type(hyperplanes_for_arrays) == tuple
    assert len(hyperplanes_for_arrays) == n_array
    assert type(dim_sizes) == tuple
    assert len(dim_sizes) == n_dim

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
            result = find_bases_with_max_reuse(
                n_dim, dim_sizes, max_reuse_factor, hyperplanes, pre_bases
            )
            # import pdb; pdb.set_trace()
            for search_status in result:
                new_ma_search_status = MultiArraySearchStatus(
                    search_status_per_array=[
                        *ma_search_status.search_status_per_array,
                        search_status,
                    ]
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


def foreach_nontrival_point(set_, return_isl_obj=False):
    points = []
    record_points_fn = partial(record_points, record=points)
    set_.foreach_point(record_points_fn)

    nonzero_count = []
    for point in points:
        nonzero_count.append(sum([1 for i in point if i != 0]))
    min_nonzero_count = min(nonzero_count)
    
    new_points = []
    for i in range(len(points)):
        if nonzero_count[i] == min_nonzero_count:
            new_points.append(points[i])
    # new_points = points

    return new_points


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
    assert type(bases_matrix) == Matrix, f"{type(bases_matrix)=}"
    bases_matrix = bases_matrix.transpose()
    coor_transform_matrix = bases_matrix.inv()
    coor_transform_matrix = inv.scale_to_integer_per_row(coor_transform_matrix)
    return coor_transform_matrix


def matrix_to_schedule(coor_transform_matrix):
    """
    Each row of coor_transform_matrix is a coordinate transformation

    Rows top->down corresponds to the inner->outer of the domain iterators

    So we need to reverse the order of rows
    """
    domain_iters_def = ", ".join([f"i{i}" for i in range(coor_transform_matrix.rows)])
    range_iters_def = ", ".join([f"o{i}" for i in range(coor_transform_matrix.rows)])
    schedule = isl.BasicMap(f"{{ [{domain_iters_def}] -> [{range_iters_def}] }}")

    for row_idx in range(coor_transform_matrix.rows):
        cons = isl.Constraint.equality_alloc(schedule.get_space())
        for col_idx in range(coor_transform_matrix.cols):
            coor = coor_transform_matrix[row_idx, col_idx]
            cons = cons.set_coefficient_val(isl.dim_type.in_, col_idx, coor)

        cons = cons.set_coefficient_val(isl.dim_type.out, coor_transform_matrix.rows - row_idx - 1, -1)
        schedule = schedule.add_constraint(cons)
    return schedule


def base_to_coor_transform_schedule(bases_matrix):
    # assert bases_matrix.rows==len(iter_names), f"{bases_matrix.rows=}, {len(iter_names)=}"
    # print(f"{bases_matrix=}")
    coor_transform_matrix = base_to_coor_transform_matrix(bases_matrix)
    # print(f"{coor_transform_matrix=}")
    schedule = matrix_to_schedule(coor_transform_matrix)
    # print(f"{schedule=}")
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


def find_schedules_for_multi_array_reuse(
    n_dim, n_array, dim_sizes, max_reuse_factor_for_arrays, hyperplanes_for_arrays,
    return_detail=False
):
    result = find_bases_for_multi_array_reuse(
        n_dim=n_dim,
        n_array=n_array,
        dim_sizes=dim_sizes,
        max_reuse_factor_for_arrays=max_reuse_factor_for_arrays,
        hyperplanes_for_arrays=hyperplanes_for_arrays,
    )
    base_matrixs = [
        ma_search_status.search_status_per_array[-1].bases
        for ma_search_status in result
    ]
    base_matrixs = [
        padding_for_base_matrix(base_matrix) for base_matrix in base_matrixs
    ]
    schedules = batch_base_to_coor_transform_schedule(base_matrixs)

    if return_detail:
        return schedules, base_matrixs

    return schedules


def parse_operator(domain, access_relations):
    """
    n_dim, n_array, dim_sizes, max_reuse_factor_for_arrays, hyperplanes_for_arrays
    """
    n_dim = domain.dim(isl.dim_type.set)
    n_array = len(access_relations)
    dim_sizes = [domain.dim_max_val(i).get_num_si() + 1 for i in range(n_dim)]
    
    max_reuse_factor_for_arrays = (max(dim_sizes), max(dim_sizes))

    hyperplanes_for_arrays = []
    for access_relation in access_relations:
        assert (
            access_relation.dim(isl.dim_type.in_) == n_dim
        ), f"{access_relation.dim(isl.dim_type.in_)=}, {n_dim=}"
        pma = access_relation.as_pw_multi_aff()
        assert pma.n_piece() == 1, f"{len(mas)=}"

        mas = []
        pma.foreach_piece(lambda x,y: mas.append(y))
        assert len(mas)==1
        ma = mas[0]
        hyperplanes = []
        # print(f"{ma=}, {ma.n_piece()=}")
        for aff_idx in range(ma.get_list().n_aff()):
            aff = ma.get_aff(aff_idx)
            coef = []
            for i in range(aff.dim(isl.dim_type.in_)):
                coef.append(int(str(aff.get_coefficient_val(isl.dim_type.in_, i))))
            hyperplanes.append(tuple(coef))
        hyperplanes_for_arrays.append(tuple(hyperplanes))
    return (
        n_dim,
        n_array,
        tuple(dim_sizes),
        tuple(max_reuse_factor_for_arrays),
        tuple(hyperplanes_for_arrays),
    )


def find_schedules_for_operator(domain, access_relations, 
    return_detail=False):
    n_dim, n_array, dim_sizes, max_reuse_factor_for_arrays, hyperplanes_for_arrays = (
        parse_operator(
            domain=domain,
            access_relations=access_relations,
        )
    )
    # print(f"{n_dim=}")
    # print(f"{n_array=}")
    # print(f"{dim_sizes=}")
    # print(f"{hyperplanes_for_arrays=}")
    # exit()
    # if return_detail is True, return (schedules, base_matrixs)
    schedules = find_schedules_for_multi_array_reuse(
        n_dim=n_dim,
        n_array=n_array,
        dim_sizes=dim_sizes,
        max_reuse_factor_for_arrays=max_reuse_factor_for_arrays,
        hyperplanes_for_arrays=hyperplanes_for_arrays,
        return_detail=return_detail,
    )
    return schedules

def shift_to_positive(op):
    domain = op.domain
    min_val = [domain.dim_min_val(i).get_num_si() for i in range(domain.dim(isl.dim_type.set))]
    shift = [0 if val >= 0 else -val for val in min_val]
    # shift = [str(val) for val in shift]
    # shift = ",".join(shift)

    shift_domain = ",".join([f"i{i}" for i in range(domain.dim(isl.dim_type.set))])
    shift_range = ",".join([f"i{i} + {shift[i]}" for i in range(domain.dim(isl.dim_type.set))])
    shift = isl.BasicMap(f"{{ [{shift_domain}] -> [{shift_range}] }}")
    new_op = op.apply_schedule(shift)

    return new_op

def auto_skewing_pass(op_list, return_detail=False):
    ori_op_list = []
    schedule_list = []
    base_matrix_list = []
    new_op_list = []
    for op in tqdm(op_list):
        schedules = find_schedules_for_operator(
            domain=op.domain,
            access_relations=[op.access_I, op.access_O],
            return_detail=return_detail
        )
        if return_detail:
            schedules, base_matrixs = schedules

        for idx,schedule in enumerate(schedules):
            new_op = op.apply_schedule(schedule)
            new_op_list.append(new_op)
            if return_detail:
                new_op.history_schedules.append(base_matrixs[idx])
                ori_op_list.append(op)
                schedule_list.append(schedule)
                base_matrix_list.append(base_matrixs[idx])

    for idx in range(len(new_op_list)):
        new_op = new_op_list[idx]
        new_op = shift_to_positive(new_op)
        new_op_list[idx] = new_op
    
    if return_detail:
        return new_op_list, ori_op_list, schedule_list, base_matrix_list
    else:
        return new_op_list

def main():

    # conv2d
    operator = BasicOperator(
        domain = isl.BasicSet(
            f"{{ [oh,ow,kh,kw]: 0<=oh<4 and 0<=ow<4 and 0<=kh<3 and 0<=kw<3 }}"
        ),
        access_I = isl.BasicMap("{ [oh,ow,kh,kw] -> I[oh + kh, ow + kw] }"),
        access_O = isl.BasicMap("{ [oh,ow,kh,kw] -> O[oh, ow] }"),
        access_W = isl.BasicMap("{ [oh,ow,kh,kw] -> W[kh, kw] }"),
    )
    tiling_schedule = isl.BasicMap("{ [i0, i1, i2, i3] -> [o0, o1, o2, o3, o4, o5, o6, o7] : (i0 + o4) mod 2 = 0 and (i1 + o5) mod 2 = 0 and (-i2 + o6) mod 3 = 0 and (-i3 + o7) mod 3 = 0 and 0 <= i0 <= 3 and 0 <= i1 <= 3 and 0 <= i2 <= 2 and 0 <= i3 <= 2 and -1 + i0 <= 2o0 <= i0 and -1 + i1 <= 2o1 <= i1 and -2 + i2 <= 3o2 <= i2 and -2 + i3 <= 3o3 <= i3 and 0 <= o4 <= 1 and 0 <= o5 <= 1 and 0 <= o6 <= 2 and 0 <= o7 <= 2 }")
    operator = operator.apply_schedule(tiling_schedule)
    new_ops = auto_skewing_pass([operator], max_reuse_factor_for_arrays=(16,4), return_detail=False)
    for new_op in new_ops:
        domain_min_per_iter = [new_op.domain.dim_min_val(i).get_num_si() for i in range(new_op.domain.dim(isl.dim_type.set))]
        domain_max_per_iter = [new_op.domain.dim_max_val(i).get_num_si() for i in range(new_op.domain.dim(isl.dim_type.set))]
        # print(f"{domain_min_per_iter=}")
        # print(f"{domain_max_per_iter=}")
        # print("-----------------------")
    # exit()
    domain = isl.BasicSet(
        "{ S[oh,ow,kh,kw]: 0<=oh<64 and 0<=ow<64 and 0<=kh<3 and 0<=kw<3 }"
    )
    access_I = isl.BasicMap("{ S[oh,ow,kh,kw] -> I[oh + kh, ow + kw] }")
    access_O = isl.BasicMap("{ S[oh,ow,kh,kw] -> O[oh, ow] }")
    # tile_transform = isl.BasicMap("{ S[oh,ow,kh,kw] -> S[floor(oh/4), oh%4, floor(ow/4), ow%4 , kh,kw] }")
    # domain = tile_transform.intersect_domain(domain).range()
    # access_I = tile_transform.reverse().apply_range(
    #     access_I
    # )
    # access_O = tile_transform.reverse().apply_range(
    #     access_O
    # )
    # access_I = utils.simplify_basic_map(access_I)
    # access_O = utils.simplify_basic_map(access_O)


    # domain = isl.BasicSet("{ [i0, i1, i2] : 0 <= i0 <= 1 and 0 <= i1 <= 3 and 0 <= i2 <= 2 }")
    # access_I = isl.BasicMap("{ [i0, i1, i2] -> I[o0] : o0 = 4i0 + i1 + i2 and i1 >= 0 and -4i0 <= i1 <= 7 - 4i0 and i1 <= 3 and 0 <= i2 <= 2 }")
    # access_O = isl.BasicMap("{ [i0, i1, i2] -> O[o0] : o0 = 4i0 + i1 and i1 >= 0 and -4i0 <= i1 <= 7 - 4i0 and i1 <= 3 and 0 <= i2 <= 2 }")

    begin_time = time.time()
    schedules, matrixs = find_schedules_for_operator(
        domain=domain,
        access_relations=[access_I],
        max_reuse_factor_for_arrays=(9,),
        return_detail=True
    )
    end_time = time.time()
    print(f"{len(schedules)=}")
    print(f"Duration: {end_time-begin_time} s")
    print(f"{call_find_base_times=}")
    # for idx , (schedule, matrix) in enumerate(zip(schedules, matrixs)):
    #     print(f"{idx}")
    #     print(f"- {schedule=}")
    #     print(f"- {matrix=}")
        
    return

    """
    BasicSet("{ [i0, i1, i2] : 0 <= i0 <= 1 and 0 <= i1 <= 3 and 0 <= i2 <= 2 }")
    ori_op.access_I = BasicMap("{ [i0, i1, i2] -> I[o0] : o0 = 4i0 + i1 + i2 and i1 >= 0 and -4i0 <= i1 <= 7 - 4i0 and i1 <= 3 and 0 <= i2 <= 2 }")
    ori_op.access_O = BasicMap("{ [i0, i1, i2] -> O[o0] : o0 = 4i0 + i1 and i1 >= 0 and -4i0 <= i1 <= 7 - 4i0 and i1 <= 3 and 0 <= i2 <= 2 }")
    """

    # result = find_bases_for_multi_array_reuse(
    #     n_dim=3,
    #     n_array=2,
    #     max_reuse_factor_for_arrays=(3,3),
    #     dim_sizes=(2, 4, 3),
    #     hyperplanes_for_arrays=(
    #         ((4,1,1),),
    #         ((4,1,0),),
    #     ),
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

    # new_bases, new_reuse = find_base_with_max_reuse(
    #     n_dim=6,
    #     max_reuse_factor=4,
    #     dim_sizes=(32, 2, 32, 2, 3, 3),
    #     hyperplanes=((2,1,0,0,1,0),(0,0,2,1,0,1)),
    # )
    # print(f"{new_bases.count_val() = }")
    # print(new_bases.foreach_point(print))
    # print(f"{new_reuse = }")
    # exit()

    # for idx,search_status in enumerate(result):
    #     print(f"---------- {idx} ----------")
    #     print(f"{search_status.bases=}")
    #     print(f"{search_status.max_reuse=}")

    # print("--------------------")
    # print(len(result))
    # exit()

    bases = find_base(
        n_dim=2,
        min_reuse_factor=1,
        dim_sizes=(8, 3),
        hyperplanes=((1,1),),
        exclude_null_space_of=None,
    )
    # print(bases)
    # bases.foreach_point(print)
    # exit()

    bases = foreach_nontrival_point(bases)
    # bases = Matrix(bases)
    # print(f"{bases=}")
    # bases = inv.find_independent_rows(bases)
    # print(f"{bases=}")
    # exit()
    current_bases = Matrix([[]])
    base = Matrix([[-1, 0, 1, 0]])
    current_bases = current_bases.col_join(base)
    # print(current_bases)

    subspace = orthogonal_sub_space(current_bases)
    # print(f"{subspace=}")

    base2 = find_base(
        n_dim=4,
        min_reuse_factor=4,
        dim_sizes=[64, 64, 4, 4],
        hyperplanes=[[0, 1, 0, 1], [1, 0, 1, 0]],
        exclude_null_space_of=subspace,
    )
    # print(f"{foreach_nontrival_point(base2)=}")
    print("-------------------")
    # base2.foreach_point(print)


if __name__ == "__main__":
    main()
