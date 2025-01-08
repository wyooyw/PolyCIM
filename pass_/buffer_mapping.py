import itertools
import math
from dataclasses import dataclass
from functools import reduce

import islpy as isl
import numpy as np
from tqdm import tqdm

import utils.utils as utils
from base_operator import (AccessRelation, BasicOperator, DataMovement,
                           DataMovementOperator)
from config import get_memory_sizes
from utils.utils import (get_box_hull_shape, rename_all_dims_for_basic_map,
                         rename_all_dims_for_basic_set,
                         rename_out_dims_for_basic_map)
from utils.logger import get_logger, debug_tqdm

logger = get_logger(__name__)

def find_domain_iters_exist_in_range(aff, return_name=True):

    n_iters_domain = aff.dim(isl.dim_type.in_)
    coefs = [
        aff.get_coefficient_val(isl.dim_type.in_, i) for i in range(n_iters_domain)
    ]
    domain_iter_index_exist_in_range = []
    for i, coef in enumerate(coefs):
        if not coef.is_zero():
            domain_iter_index_exist_in_range.append(i)

    if return_name:
        domain_iter_names_exist_in_range = [
            aff.get_dim_name(isl.dim_type.in_, i)
            for i in domain_iter_index_exist_in_range
        ]
        return domain_iter_names_exist_in_range

    return domain_iter_index_exist_in_range


def find_domain_iters_exist_in_range_list_of_aff(affs):
    domain_iter_names_exist_in_range = []
    for aff in affs:
        domain_iter_names_exist_in_range.extend(find_domain_iters_exist_in_range(aff))
    domain_iter_names_exist_in_range = list(set(domain_iter_names_exist_in_range))
    return domain_iter_names_exist_in_range


def pw_aff_to_aff(pw_aff):
    affs = pw_aff.get_pieces()
    assert len(affs) == 1, f"{affs=}"
    aff = affs[0][1]
    assert type(aff) == isl.Aff
    return aff


def build_domain_aligned_buffer_exclude_iters(
    domain, buffer_name, exclude_iter_names=[]
):
    """
    args:
        domain: [i,j,k]
        exclude_iter_names: [j]
    return:
        access relation: [i,j,k] -> A[i,k]
    """

    shape = get_box_hull_shape(domain)
    iter_names = domain.get_var_names(isl.dim_type.set)
    n_iter = len(iter_names)

    # assert exclude_iter_names is subset of iter_names
    assert set(exclude_iter_names).issubset(
        set(iter_names)
    ), f"{exclude_iter_names=}, {iter_names=}"

    iter_in_array_names = [
        iter_name for iter_name in iter_names if iter_name not in exclude_iter_names
    ]
    access_relation = isl.BasicMap(
        f"{{ [{','.join(iter_names)}] -> {buffer_name}[{','.join(iter_in_array_names)}] }}"
    )
    access_relation = access_relation.intersect_domain(domain)
    access_relation = rename_out_dims_for_basic_map(access_relation)
    return access_relation


def map_domain_aligned_buffer_to_origin_buffer_v2(domain, acc_rel):
    """
    1.get domain aligned buffer
    2.map this aligned buffer to origin buffer
    if a domain's iter not exist in buffer's iter, then no need to map it to buffer.
    """
    buffer_name = acc_rel.get_tuple_name(isl.dim_type.out)
    align_buffer_name = f"{buffer_name}_aligned"

    n_buf_dim = acc_rel.dim(isl.dim_type.out)
    n_domain_dim = acc_rel.dim(isl.dim_type.in_)

    domain_iter_names = acc_rel.get_var_names(isl.dim_type.in_)
    involve_dims = get_dominate_iters_of_pw_multi_aff(
        acc_rel.as_pw_multi_aff(), return_name=True
    )
    domain_iter_names_not_exist_in_lb_ub = list(
        set(domain_iter_names) - set(involve_dims)
    )

    aligned_acc_rel = build_domain_aligned_buffer_exclude_iters(
        domain, align_buffer_name, domain_iter_names_not_exist_in_lb_ub
    )
    # one to many
    # buffer_mapping = acc_rel.reverse().apply_range(aligned_acc_rel)

    # many to one
    buffer_mapping = aligned_acc_rel.reverse().apply_range(acc_rel)
    assert buffer_mapping.is_single_valued()
    return buffer_mapping, aligned_acc_rel


def map_domain_aligned_buffer_to_origin_buffer_for_weight(
    domain, acc_rel, force_inner_level=4
):
    """
    1.get domain aligned buffer
    2.map this aligned buffer to origin buffer
    if a domain's iter not exist in buffer's iter, then no need to map it to buffer.
    """
    buffer_name = acc_rel.get_tuple_name(isl.dim_type.out)
    align_buffer_name = f"{buffer_name}_aligned"

    n_buf_dim = acc_rel.dim(isl.dim_type.out)
    n_domain_dim = acc_rel.dim(isl.dim_type.in_)

    domain_iter_names = acc_rel.get_var_names(isl.dim_type.in_)
    involve_dims = get_dominate_iters_of_pw_multi_aff(
        acc_rel.as_pw_multi_aff(), return_name=True
    )
    force_involve_dims = domain_iter_names[-force_inner_level:]
    involve_dims = involve_dims | set(force_involve_dims)

    domain_iter_names_not_exist_in_lb_ub = list(
        set(domain_iter_names) - set(involve_dims)
    )

    aligned_acc_rel = build_domain_aligned_buffer_exclude_iters(
        domain, align_buffer_name, domain_iter_names_not_exist_in_lb_ub
    )

    # one to many
    # buffer_mapping = acc_rel.reverse().apply_range(aligned_acc_rel)

    # many to one
    buffer_mapping = aligned_acc_rel.reverse().apply_range(acc_rel)
    assert buffer_mapping.is_single_valued()
    return buffer_mapping, aligned_acc_rel


# def is_continuous(isl_obj, ):


def make_affs_to_aff_list(affs):
    aff_list = isl.AffList.alloc(affs[0].get_ctx(), 0)
    for aff in affs:
        aff_list = aff_list.add(aff)
    return aff_list


def make_pw_affs_to_aff_list(pw_affs):
    pw_aff_list = isl.PwAffList.alloc(pw_affs[0].get_ctx(), 0)
    for pw_aff in pw_affs:
        pw_aff_list = pw_aff_list.add(pw_aff)
    return pw_aff_list


def get_dominate_iters_of_pw_aff(pw_aff):
    """
    {[i0,i1,..,ik] -> [f(i1,i2)]}
    return {i1,i2}
    """
    dim_names = [
        pw_aff.get_dim_name(isl.dim_type.in_, i)
        for i in range(pw_aff.dim(isl.dim_type.in_))
    ]

    dominate_dim = set()
    for cond, aff in pw_aff.get_pieces():
        for i in range(aff.dim(isl.dim_type.in_)):
            coef = aff.get_coefficient_val(isl.dim_type.in_, i)
            if not coef == 0:
                dominate_dim.add(dim_names[i])
    return dominate_dim


def get_pieces_from_pw_multi_aff(pw_multi_aff):
    record = []
    pw_multi_aff.foreach_piece(lambda x, y: record.append((x, y)))
    return record


def get_dominate_iters_of_pw_multi_aff_per_out(pw_multi_aff, return_name=True):
    """
    {[i0,i1,..,ik] -> [f(i1,i2)]}
    return {i1,i2}
    """
    dim_names = [
        pw_multi_aff.get_dim_name(isl.dim_type.in_, i)
        for i in range(pw_multi_aff.dim(isl.dim_type.in_))
    ]
    n_dim_range = pw_multi_aff.dim(isl.dim_type.out)

    dominate_dims = []
    for i in range(n_dim_range):
        dominate_dims.append(set())

    for cond, multi_aff in get_pieces_from_pw_multi_aff(pw_multi_aff):
        for dim in range(n_dim_range):
            aff = multi_aff.get_at(dim)
            for i in range(aff.dim(isl.dim_type.in_)):
                if aff.involves_dims(isl.dim_type.in_, i, 1):
                    dominate_dims[dim].add(dim_names[i] if return_name else i)

    return dominate_dims


def get_dominate_iters_of_pw_multi_aff(pw_multi_aff, return_name=True):
    """
    {[i0,i1,..,ik] -> [f(i1,i2)]}
    return {i1,i2}
    """
    dim_names = [
        pw_multi_aff.get_dim_name(isl.dim_type.in_, i)
        for i in range(pw_multi_aff.dim(isl.dim_type.in_))
    ]
    n_dim_range = pw_multi_aff.dim(isl.dim_type.out)

    dominate_dims = set()

    for cond, multi_aff in get_pieces_from_pw_multi_aff(pw_multi_aff):
        for dim in range(n_dim_range):
            aff = multi_aff.get_at(dim)
            for i in range(aff.dim(isl.dim_type.in_)):
                # coef = aff.get_coefficient_val(isl.dim_type.in_, i)
                if aff.involves_dims(isl.dim_type.in_, i, 1):
                    dominate_dims.add(dim_names[i] if return_name else i)

    return dominate_dims


def get_local_buffer_axis_mapping(domain, acc_rel, level):
    """
    local_to_global_buf_axis_mapping = [1,2] means local buffer axis [0,1] map to global buffer axis [1,2]
    """

    # step 1: get dominate iter of each dim in acc_rel's range
    # dominate: iter exist in dim's lowerbound / upperbound affine function

    n_buf_dim = acc_rel.dim(isl.dim_type.out)
    n_iter_dim = acc_rel.dim(isl.dim_type.in_)

    # lb_per_dim = [acc_rel.dim_min(i) for i in range(n_buf_dim)]
    # ub_per_dim = [acc_rel.dim_max(i) for i in range(n_buf_dim)]

    # dominate_iters_per_buffer_axis = []
    # for buf_axix,(lb,ub) in enumerate(zip(lb_per_dim, ub_per_dim)):
    #     lb_dominate_iters = get_dominate_iters_of_pw_aff(lb)
    #     ub_dominate_iters = get_dominate_iters_of_pw_aff(ub)
    #     dominate_iters = lb_dominate_iters.union(ub_dominate_iters)
    #     dominate_iters_per_buffer_axis.append(dominate_iters)
    dominate_iters_per_buffer_axis = get_dominate_iters_of_pw_multi_aff_per_out(
        acc_rel.as_pw_multi_aff()
    )

    # step 2: get inner level iter names
    iter_names = acc_rel.get_var_names(isl.dim_type.in_)
    inner_iter_names = iter_names[level:]

    # step 3: find buffer axis that contain inner level iter
    local_to_global_buf_axis_mapping = []
    for buffer_axis, dominate_iters in enumerate(dominate_iters_per_buffer_axis):
        if dominate_iters.intersection(inner_iter_names):
            local_to_global_buf_axis_mapping.append(buffer_axis)

    return local_to_global_buf_axis_mapping


def map_prefix_domain_aligned_buffer_to_aligned_buffer_v2(domain, acc_rel, level):
    local_to_global_buf_axis_mapping = get_local_buffer_axis_mapping(
        domain, acc_rel, level
    )

    n_buf_dim = acc_rel.dim(isl.dim_type.out)
    n_iter_dim = acc_rel.dim(isl.dim_type.in_)
    # assert n_buf_dim==n_iter_dim, f"{n_buf_dim=}, {n_iter_dim=}"

    iter_names = domain.get_var_names(isl.dim_type.set)
    prefix_acc_rel = acc_rel.project_out_except(iter_names[:level], [isl.dim_type.in_])
    used_global_buffer_dynamic_shape = utils.get_dynamic_shape_from_dynamic_map(
        prefix_acc_rel
    )  # [level:]
    local_buffer_dynamic_shape = [
        used_global_buffer_dynamic_shape[local_to_global_buf_axis_mapping[i]]
        for i in range(len(local_to_global_buf_axis_mapping))
    ]
    # logger.debug(f"{local_buffer_dynamic_shape = }")
    # check prefix_acc_rel is continous on given dim

    lower_bound_per_dim = [prefix_acc_rel.dim_min(i) for i in range(n_buf_dim)]
    lb_aff_per_dim = lower_bound_per_dim

    n_local_buf_dim = len(local_to_global_buf_axis_mapping)  # - level

    param_names = acc_rel.get_var_names(isl.dim_type.param)
    domain_names = acc_rel.get_var_names(isl.dim_type.in_)
    range_names = acc_rel.get_var_names(isl.dim_type.out)

    local_buffer_iters = [utils.get_unique_name() for i in range(n_local_buf_dim)]

    # insert local buffer dim into lb_aff_per_dim's domain
    # For example: {[i0,i1] -> Buffer[j0,j1,j2]}  -> {[i0,i1,a_,b_] -> Buffer[j0,j1,j2]}
    # a_,b_ is local buffer dim
    for i in range(len(lb_aff_per_dim)):

        lb_aff = lb_aff_per_dim[i]
        lb_aff = lb_aff.insert_dims(isl.dim_type.in_, level, n_local_buf_dim)
        for j in range(n_local_buf_dim):
            lb_aff = lb_aff.set_dim_id(
                isl.dim_type.in_, level + j, isl.Id(local_buffer_iters[j])
            )
        lb_aff_per_dim[i] = lb_aff

    # build buffer's access relation
    affs = []
    aff_domain_iters = lb_aff_per_dim[0].get_var_names(isl.dim_type.in_)
    aff_domain_def = ",".join(aff_domain_iters)
    for i in range(n_buf_dim):
        affs.append(lb_aff_per_dim[i])

    assert n_local_buf_dim <= n_buf_dim, f"{n_local_buf_dim=}, {n_buf_dim=}"
    for i in range(n_local_buf_dim):
        # i is local buffer dim, use local_to_global_buf_axis_mapping to get global buffer dim
        global_buffer_axis = local_to_global_buf_axis_mapping[i]
        aff_lb = affs[global_buffer_axis]
        aff_i = isl.Aff(f"{{ [{aff_domain_def}] -> [({local_buffer_iters[i]})] }}")
        aff = aff_lb.add(aff_i)
        affs[global_buffer_axis] = aff

    pw_aff_list = utils.make_pw_affs_to_aff_list(affs)

    assign_buffer_acc_rel = isl.MultiPwAff.from_pw_aff_list(
        affs[0].space.insert_dims(isl.dim_type.out, 0, len(pw_aff_list) - 1),
        pw_aff_list,
    )
    assign_buffer_acc_rel = isl.PwMultiAff.from_multi_pw_aff(assign_buffer_acc_rel)
    assign_buffer_acc_rel = isl.Map.from_pw_multi_aff(assign_buffer_acc_rel)

    # basic_maps = assign_buffer_acc_rel.get_basic_maps()
    # assert len(basic_maps)==1, f"{len(basic_maps)=}"
    # assign_buffer_acc_rel = basic_maps[0]

    # build local buffer's access relation
    affs = []
    for i in range(n_local_buf_dim):
        aff = isl.Aff(f"{{ [{aff_domain_def}] -> [({local_buffer_iters[i]})] }}")
        affs.append(aff)
    aff_list = make_affs_to_aff_list(affs)
    local_buffer_acc_rel = isl.BasicMap.from_aff_list(affs[0].domain().space, aff_list)

    # build assign domain
    assign_domain = domain.project_out_except(iter_names[:level], [isl.dim_type.set])
    assign_domain = assign_domain.add_dims(isl.dim_type.set, n_local_buf_dim)
    for i in range(n_local_buf_dim):
        assign_domain = assign_domain.set_dim_name(
            isl.dim_type.set, level + i, local_buffer_iters[i]
        )

    ub_mpf = utils.multi_pw_aff_from_pw_affs(local_buffer_dynamic_shape)
    ub_mpf = ub_mpf.add_constant_val(isl.Val.int_from_si(ub_mpf.get_ctx(), -1))

    # if level==2:
    #     import pdb; pdb.set_trace()
    assign_domain = utils.mpf_upper_bound_for_basic_set(
        ub_mpf, assign_domain, n_local_buf_dim
    )
    assign_domain = utils.zero_lower_bound_for_basic_set(assign_domain, n_local_buf_dim)

    # set tuple name
    buffer_name = acc_rel.get_tuple_name(isl.dim_type.out)
    assign_buffer_acc_rel = assign_buffer_acc_rel.set_tuple_name(
        isl.dim_type.out, buffer_name
    )
    local_buffer_acc_rel = local_buffer_acc_rel.set_tuple_name(
        isl.dim_type.out, f"{buffer_name}_{level}"
    )

    return assign_domain, local_buffer_acc_rel, assign_buffer_acc_rel


def get_local_buffer_axis_mapping_for_weight(domain, acc_rel, level, force_inner_level):
    """
    local_to_global_buf_axis_mapping = [1,2] means local buffer axis [0,1] map to global buffer axis [1,2]
    """

    # step 1: get dominate iter of each dim in acc_rel's range
    # dominate: iter exist in dim's lowerbound / upperbound affine function

    n_buf_dim = acc_rel.dim(isl.dim_type.out)
    n_iter_dim = acc_rel.dim(isl.dim_type.in_)
    iter_names = acc_rel.get_var_names(isl.dim_type.in_)
    dominate_iters_per_buffer_axis = get_dominate_iters_of_pw_multi_aff_per_out(
        acc_rel.as_pw_multi_aff()
    )

    for i in range(force_inner_level):
        dominate_iters = dominate_iters_per_buffer_axis[
            n_buf_dim - force_inner_level + i
        ]
        dominate_iters.add(iter_names[n_iter_dim - force_inner_level + i])
    # import pdb; pdb.set_trace()
    # step 2: get inner level iter names
    # iter_names = acc_rel.get_var_names(isl.dim_type.in_)
    inner_iter_names = iter_names[level:]

    # step 3: find buffer axis that contain inner level iter
    local_to_global_buf_axis_mapping = []
    for buffer_axis, dominate_iters in enumerate(dominate_iters_per_buffer_axis):
        if dominate_iters.intersection(inner_iter_names):
            local_to_global_buf_axis_mapping.append(buffer_axis)

    return local_to_global_buf_axis_mapping


def map_prefix_domain_aligned_buffer_to_aligned_buffer_for_weight(
    domain, acc_rel, level, force_inner_level
):
    local_to_global_buf_axis_mapping = get_local_buffer_axis_mapping_for_weight(
        domain, acc_rel, level, force_inner_level
    )

    n_buf_dim = acc_rel.dim(isl.dim_type.out)
    n_iter_dim = acc_rel.dim(isl.dim_type.in_)
    # assert n_buf_dim==n_iter_dim, f"{n_buf_dim=}, {n_iter_dim=}"
    # import pdb; pdb.set_trace()
    iter_names = domain.get_var_names(isl.dim_type.set)
    prefix_acc_rel = acc_rel.project_out_except(iter_names[:level], [isl.dim_type.in_])
    used_global_buffer_dynamic_shape = utils.get_dynamic_shape_from_dynamic_map(
        prefix_acc_rel
    )  # [level:]
    local_buffer_dynamic_shape = [
        used_global_buffer_dynamic_shape[local_to_global_buf_axis_mapping[i]]
        for i in range(len(local_to_global_buf_axis_mapping))
    ]
    # logger.debug(f"{local_buffer_dynamic_shape = }")
    # check prefix_acc_rel is continous on given dim

    lower_bound_per_dim = [prefix_acc_rel.dim_min(i) for i in range(n_buf_dim)]
    lb_aff_per_dim = lower_bound_per_dim

    n_local_buf_dim = len(local_to_global_buf_axis_mapping)  # - level

    param_names = acc_rel.get_var_names(isl.dim_type.param)
    domain_names = acc_rel.get_var_names(isl.dim_type.in_)
    range_names = acc_rel.get_var_names(isl.dim_type.out)

    local_buffer_iters = [utils.get_unique_name() for i in range(n_local_buf_dim)]

    # insert local buffer dim into lb_aff_per_dim's domain
    # For example: {[i0,i1] -> Buffer[j0,j1,j2]}  -> {[i0,i1,a_,b_] -> Buffer[j0,j1,j2]}
    # a_,b_ is local buffer dim
    for i in range(len(lb_aff_per_dim)):

        lb_aff = lb_aff_per_dim[i]
        lb_aff = lb_aff.insert_dims(isl.dim_type.in_, level, n_local_buf_dim)
        for j in range(n_local_buf_dim):
            lb_aff = lb_aff.set_dim_id(
                isl.dim_type.in_, level + j, isl.Id(local_buffer_iters[j])
            )
        lb_aff_per_dim[i] = lb_aff

    # build buffer's access relation
    affs = []
    aff_domain_iters = lb_aff_per_dim[0].get_var_names(isl.dim_type.in_)
    aff_domain_def = ",".join(aff_domain_iters)
    for i in range(n_buf_dim):
        affs.append(lb_aff_per_dim[i])

    assert n_local_buf_dim <= n_buf_dim, f"{n_local_buf_dim=}, {n_buf_dim=}"
    for i in range(n_local_buf_dim):
        # i is local buffer dim, use local_to_global_buf_axis_mapping to get global buffer dim
        global_buffer_axis = local_to_global_buf_axis_mapping[i]
        aff_lb = affs[global_buffer_axis]
        aff_i = isl.Aff(f"{{ [{aff_domain_def}] -> [({local_buffer_iters[i]})] }}")
        aff = aff_lb.add(aff_i)
        affs[global_buffer_axis] = aff

    pw_aff_list = utils.make_pw_affs_to_aff_list(affs)

    assign_buffer_acc_rel = isl.MultiPwAff.from_pw_aff_list(
        affs[0].space.insert_dims(isl.dim_type.out, 0, len(pw_aff_list) - 1),
        pw_aff_list,
    )
    assign_buffer_acc_rel = isl.PwMultiAff.from_multi_pw_aff(assign_buffer_acc_rel)
    assign_buffer_acc_rel = isl.Map.from_pw_multi_aff(assign_buffer_acc_rel)

    # basic_maps = assign_buffer_acc_rel.get_basic_maps()
    # assert len(basic_maps)==1, f"{len(basic_maps)=}"
    # assign_buffer_acc_rel = basic_maps[0]

    # build local buffer's access relation
    affs = []
    for i in range(n_local_buf_dim):
        aff = isl.Aff(f"{{ [{aff_domain_def}] -> [({local_buffer_iters[i]})] }}")
        affs.append(aff)
    aff_list = make_affs_to_aff_list(affs)
    local_buffer_acc_rel = isl.BasicMap.from_aff_list(affs[0].domain().space, aff_list)

    # build assign domain
    assign_domain = domain.project_out_except(iter_names[:level], [isl.dim_type.set])
    assign_domain = assign_domain.add_dims(isl.dim_type.set, n_local_buf_dim)
    for i in range(n_local_buf_dim):
        assign_domain = assign_domain.set_dim_name(
            isl.dim_type.set, level + i, local_buffer_iters[i]
        )

    ub_mpf = utils.multi_pw_aff_from_pw_affs(local_buffer_dynamic_shape)
    ub_mpf = ub_mpf.add_constant_val(isl.Val.int_from_si(ub_mpf.get_ctx(), -1))

    # if level==2:
    #     import pdb; pdb.set_trace()
    assign_domain = utils.mpf_upper_bound_for_basic_set(
        ub_mpf, assign_domain, n_local_buf_dim
    )
    assign_domain = utils.zero_lower_bound_for_basic_set(assign_domain, n_local_buf_dim)

    # set tuple name
    buffer_name = acc_rel.get_tuple_name(isl.dim_type.out)
    assign_buffer_acc_rel = assign_buffer_acc_rel.set_tuple_name(
        isl.dim_type.out, buffer_name
    )
    local_buffer_acc_rel = local_buffer_acc_rel.set_tuple_name(
        isl.dim_type.out, f"{buffer_name}_{level}"
    )

    return assign_domain, local_buffer_acc_rel, assign_buffer_acc_rel


def apply_skew(domain, acc_rel):
    skew_map = isl.BasicMap("{ [i,j,k] -> [i,j+k,k] }")
    new_domain = skew_map.intersect_domain(domain).range()
    new_acc_rel = skew_map.reverse().apply_range(acc_rel)

    new_acc_rel = rename_all_dims_for_basic_map(new_acc_rel)
    new_domain = rename_all_dims_for_basic_set(new_domain)
    return new_domain, new_acc_rel


def apply_tile(domain, acc_rel):
    tile_map = isl.BasicMap(
        "{ [i,j,k] -> [floor(i/2),floor(j/2),floor(k/2),i%2,j%2,k%2] }"
    )
    new_domain = tile_map.intersect_domain(domain).range()
    new_acc_rel = tile_map.reverse().apply_range(acc_rel)

    new_acc_rel = rename_all_dims_for_basic_map(new_acc_rel)
    new_domain = rename_all_dims_for_basic_set(new_domain)
    return new_domain, new_acc_rel


def get_range_dim_size(acc_rel, pos, return_int=True):
    assert type(acc_rel) == isl.BasicMap
    dim_min_pw_aff = acc_rel.dim_min(pos)
    dim_max_pw_aff = acc_rel.dim_max(pos)

    dim_len = dim_max_pw_aff.sub(dim_min_pw_aff)
    dim_len = dim_len.add_constant_val(isl.Val.one(dim_len.get_ctx()))

    dim_len_ub = dim_len.max_val()
    assert dim_len_ub.is_int()
    if return_int:
        dim_len_ub = int(str(dim_len_ub))

    return dim_len_ub


def get_static_shape_from_dynamic_map(isl_map, return_list_int=True):
    # num dim in out
    n_out = isl_map.dim(isl.dim_type.out)
    shape = [get_range_dim_size(isl_map, pos, return_list_int) for pos in range(n_out)]
    if return_list_int:
        shape = [shape_i for shape_i in shape]
    return shape


def insert_const_dim_in_range(map_, pos, si):
    map_ = map_.insert_dims(isl.dim_type.out, pos, 1)
    val = isl.Val.int_from_si(map_.get_ctx(), si)
    map_ = map_.upper_bound_val(isl.dim_type.out, pos, val)
    map_ = map_.lower_bound_val(isl.dim_type.out, pos, val)
    return map_


def insert_many_const_dim_in_range(map_, pos, size, si):
    map_ = map_.insert_dims(isl.dim_type.out, pos, size)
    val = isl.Val.int_from_si(map_.get_ctx(), si)
    for i in range(size):
        map_ = map_.upper_bound_val(isl.dim_type.out, pos + i, val)
        map_ = map_.lower_bound_val(isl.dim_type.out, pos + i, val)
    return map_


def align_compute_and_assign_schedules(compute_schedule, assign_schedules, levels):
    level_to_assign_schedule = dict()
    assign_schedule_to_level = dict()
    for assign_schedule, level in zip(assign_schedules, levels):
        if level in level_to_assign_schedule:
            level_to_assign_schedule[level].append(assign_schedule)
        else:
            level_to_assign_schedule[level] = [assign_schedule]

        assign_schedule_to_level[assign_schedule] = level

    sorted_levels = sorted(list(level_to_assign_schedule.keys()))
    logger.debug(f"{sorted_levels=}")

    # insert dims
    for level in levels:
        assign_schedules_at_level = level_to_assign_schedule[level]

        # insert constant dims for assign schedules at current level
        for i in range(len(assign_schedules_at_level)):
            assign_schedules_at_level[i] = insert_const_dim_in_range(
                assign_schedules_at_level[i], level, i
            )

        # insert dims for other schedule
        const = len(assign_schedules_at_level)
        for other_level in levels:
            if other_level == level:
                continue
            assign_schedules_at_level[i] = insert_const_dim_in_range(
                assign_schedules_at_level[i], level, const
            )
        compute_schedule = insert_const_dim_in_range(compute_schedule, level, const)

    # padding schedule at end
    max_range_size = compute_schedule.dim(isl.dim_type.out)
    for level in levels:
        assign_schedules_at_level = level_to_assign_schedule[level]
        for assign_schedule in assign_schedules_at_level:
            range_size = assign_schedule.dim(isl.dim_type.out)
            max_range_size = max(max_range_size, range_size)

    cur_range_size = compute_schedule.dim(isl.dim_type.out)
    compute_schedule = insert_many_const_dim_in_range(
        compute_schedule, cur_range_size, max_range_size - cur_range_size, 0
    )
    for level in levels:
        assign_schedules_at_level = level_to_assign_schedule[level]
        for i in range(len(assign_schedules_at_level)):
            assign_schedule = assign_schedules_at_level[i]
            cur_range_size = assign_schedule.dim(isl.dim_type.out)
            assign_schedule = insert_many_const_dim_in_range(
                assign_schedule, cur_range_size, max_range_size - cur_range_size, 0
            )
            assign_schedules_at_level[i] = assign_schedule

    union_schedule = compute_schedule
    for level in levels:
        assign_schedules_at_level = level_to_assign_schedule[level]
        for assign_schedule in assign_schedules_at_level:
            union_schedule = union_schedule.add_map(assign_schedule)

    return union_schedule


def map_access_to_buffer(
    origin_access, assign_buffer_input_access, assign_buffer_output_access, level
):
    """
    origin_access: compute domain -> global array
    assign_buffer_input_access: assign domain -> global array
    assign_buffer_output_access: assign domain -> local buffer

    return: compute domain -> local buffer
    """

    assign_buffer_input_access = assign_buffer_input_access.move_dims(
        isl.dim_type.param, 0, isl.dim_type.in_, 0, level
    )
    assign_buffer_output_access = assign_buffer_output_access.move_dims(
        isl.dim_type.param, 0, isl.dim_type.in_, 0, level
    )

    global_local_buffer_mapping = assign_buffer_input_access.reverse().apply_range(
        assign_buffer_output_access
    )

    origin_access = origin_access.move_dims(
        isl.dim_type.param, 0, isl.dim_type.in_, 0, level
    )
    local_buffer_access = origin_access.apply_range(global_local_buffer_mapping)
    local_buffer_access = local_buffer_access.move_dims(
        isl.dim_type.in_, 0, isl.dim_type.param, 0, level
    )
    return local_buffer_access


def insert_single_buffer_single_level(op, buffer_name, buffer_level):
    map_buf_align_to_ori, aligned_acc_rel = (
        map_domain_aligned_buffer_to_origin_buffer_v2(
            op.domain, op.get_access_by_name(buffer_name)
        )
    )
    assign_domain, assign_local_buffer_acc_rel, assign_global_buffer_acc_rel = (
        map_prefix_domain_aligned_buffer_to_aligned_buffer_v2(
            op.domain, aligned_acc_rel, buffer_level
        )
    )
    compute_local_buffer_acc_rel = map_access_to_buffer(
        aligned_acc_rel,
        assign_global_buffer_acc_rel,
        assign_local_buffer_acc_rel,
        buffer_level,
    )

    accesses = {"I": op.access_I, "O": op.access_O, "W": op.access_W}
    accesses[buffer_name] = compute_local_buffer_acc_rel
    new_op = DataMovementOperator(
        domain=op.domain,
        access_I=accesses["I"],
        access_O=accesses["O"],
        access_W=accesses["W"],
    )
    datamove = DataMovement(
        domain=assign_domain,
        access_I=assign_global_buffer_acc_rel.intersect_domain(assign_domain),
        access_O=assign_local_buffer_acc_rel.intersect_domain(assign_domain),
        level=buffer_level,
    )
    new_op.insert_buffer(buffer_name, datamove)

    # compute_schedule = utils.identity_map_from_set(op.domain)
    # assign_schedule = utils.identity_map_from_set(assign_domain)

    # compute_domain = op.domain.set_tuple_name("S")
    # compute_schedule = compute_schedule.set_tuple_name(isl.dim_type.in_, "S")

    # assign_domain = assign_domain.set_tuple_name("T")
    # assign_schedule = assign_schedule.set_tuple_name(isl.dim_type.in_, "T")

    # union_domain = compute_domain.add_set(assign_domain) #.add_set(assign_domain2)
    # union_schedule = align_compute_and_assign_schedules(compute_schedule, [assign_schedule], [buffer_level])

    # ast = utils.gen_ast(union_domain,union_schedule,None)
    # code = utils.gen_code(union_domain,union_schedule,None)
    return new_op


def insert_single_buffer_single_level_pass(op_list, buffer_name, buffer_level):
    new_codes = []
    for op in debug_tqdm(op_list):
        new_op = insert_single_buffer_single_level(op, buffer_name, buffer_level)
        new_codes.append(new_op)
    return new_codes


def parse_buffer_levels(op, buffer_levels):

    n_domain_dim = op.domain.dim(isl.dim_type.set)
    new_buffer_levels = []
    for buffer_level in buffer_levels:
        if buffer_level < 0:
            buffer_level = n_domain_dim + buffer_level + 1
        new_buffer_levels.append(buffer_level)
    # check increase
    for i in range(1, len(new_buffer_levels)):
        assert new_buffer_levels[i] > new_buffer_levels[i - 1], f"{new_buffer_levels=}"
    return new_buffer_levels


def insert_single_buffer_multi_level(
    op, buffer_name, buffer_levels, memory_types, force_inner_level=5
):
    buffer_levels = parse_buffer_levels(op, buffer_levels)

    assert isinstance(buffer_levels, list)
    buffer_levels = sorted(buffer_levels)

    if "W" in buffer_name:
        map_buf_align_to_ori, aligned_acc_rel = (
            map_domain_aligned_buffer_to_origin_buffer_for_weight(
                op.domain,
                op.get_access_by_name(buffer_name),
                force_inner_level=force_inner_level,
            )
        )

    else:
        map_buf_align_to_ori, aligned_acc_rel = (
            map_domain_aligned_buffer_to_origin_buffer_v2(
                op.domain, op.get_access_by_name(buffer_name)
            )
        )

    compute_acc_rel = aligned_acc_rel
    data_movement_list = []

    assert (
        len(memory_types) == len(buffer_levels) + 1
    ), f"{memory_types=}, {buffer_levels=}"
    memory_types = [*memory_types]
    # memory_types.insert(0, "__INPUT_MEMORY__")

    for idx, buffer_level in enumerate(buffer_levels):
        if "W" in buffer_name:
            assign_domain, assign_local_buffer_acc_rel, assign_global_buffer_acc_rel = (
                map_prefix_domain_aligned_buffer_to_aligned_buffer_for_weight(
                    op.domain,
                    compute_acc_rel,
                    buffer_level,
                    force_inner_level=force_inner_level,
                )
            )
            # import pdb; pdb.set_trace()
        else:
            assign_domain, assign_local_buffer_acc_rel, assign_global_buffer_acc_rel = (
                map_prefix_domain_aligned_buffer_to_aligned_buffer_v2(
                    op.domain, compute_acc_rel, buffer_level
                )
            )
        compute_acc_rel = map_access_to_buffer(
            compute_acc_rel,
            assign_global_buffer_acc_rel,
            assign_local_buffer_acc_rel,
            buffer_level,
        )
        datamove = DataMovement(
            domain=assign_domain,
            access_I=AccessRelation(
                assign_global_buffer_acc_rel.intersect_domain(assign_domain),
                memory_types[idx],
            ),
            access_O=AccessRelation(
                assign_local_buffer_acc_rel.intersect_domain(assign_domain),
                memory_types[idx + 1],
            ),
            level=buffer_level,
        )
        data_movement_list.append(datamove)

    accesses = {"I": op.access_I, "O": op.access_O, "W": op.access_W}
    accesses[buffer_name] = AccessRelation(compute_acc_rel, memory_types[-1])
    new_op = DataMovementOperator(
        domain=op.domain,
        access_I=accesses["I"],
        access_O=accesses["O"],
        access_W=accesses["W"],
        history_domains=op.history_domains,
        history_schedules=op.history_schedules,
        data_movement=op.data_movement if hasattr(op, "data_movement") else None,
        attr={key: value for key, value in op.attr.items()},
    )

    # logger.debug(f"domain: {op.domain}\n")
    # logger.debug(f"access_I: {op.access_I}\n")
    # logger.debug(f"access_O: {op.access_O}\n")
    # logger.debug(f"access_W: {op.access_W}\n")

    for idx, data_movement in enumerate(data_movement_list):
        new_op.insert_buffer(buffer_name, data_movement)
    #     logger.debug(f"{idx}. {data_movement.domain=}\n")
    # logger.debug("\n-----------------------\n")

    return new_op


def insert_single_buffer_multi_level_pass(op_list, buffer_name, buffer_levels):
    """
    new_ops = insert_single_buffer_multi_level_pass(new_ops, buffer_name="W", buffer_levels=[0, 2])
    """
    new_codes = []
    for op in debug_tqdm(op_list):
        new_op = insert_single_buffer_multi_level(op, buffer_name, buffer_levels)
        new_codes.append(new_op)
    return new_codes


"""
Buffer Searching
"""


def get_valid_buffer_positions(acc_rel):
    dominate_iters_per_dim = get_dominate_iters_of_pw_multi_aff_per_out(
        acc_rel.as_pw_multi_aff(), return_name=False
    )
    # logger.debug(f"{dominate_iters_per_dim=}")
    dominate_iters = reduce(lambda x, y: x.union(y), dominate_iters_per_dim)
    assert type(dominate_iters) == set
    dominate_iters = sorted(list(dominate_iters))
    valid_buffer_positions = [i + 1 for i in dominate_iters]
    return valid_buffer_positions


def buffer_level_serching(
    op,
    buffer_name,
    num_buffer_level,
    level_min=None,
    level_max=None,
):
    """
    num_buffer_level = 1
    names_of_buffer_level = ["__MACRO__"]
    level_min = None
    level_max = None

    num_buffer_level = 2
    names_of_buffer_level = ["__INPUT_MEMORY__", "__PIM_INPUT_REG_BUFFER__"]
    level_min = 1
    level_max = -5

    return buffer_level_list
    """
    if level_min is None:
        level_min = 0
    if level_max is None:
        level_max = -1

    level_min, level_max = parse_buffer_levels(op, (level_min, level_max))

    acc_rel = op.get_access_by_name(buffer_name)

    # find dominate iters
    valid_buffer_positions = get_valid_buffer_positions(acc_rel)
    valid_buffer_positions = [0] + valid_buffer_positions
    valid_buffer_positions = [
        i for i in valid_buffer_positions if i >= level_min and i <= level_max
    ]

    assert num_buffer_level <= len(
        valid_buffer_positions
    ), f"{num_buffer_level=}, {len(valid_buffer_positions)=}"
    buffer_level_combinations = list(
        itertools.combinations(valid_buffer_positions, num_buffer_level)
    )

    return buffer_level_combinations


def get_macro_level(op, buffer_name, buffer_compute_level):
    acc_rel = op.get_access_by_name(buffer_name)
    valid_buffer_positions = get_valid_buffer_positions(acc_rel)
    valid_buffer_positions = [0] + valid_buffer_positions

    buffer_compute_level = parse_buffer_levels(op, (buffer_compute_level,))[0]
    # find biggest level in valid_buffer_positions smaller or equal with buffer_compute_level
    max_level = None
    for idx, level in enumerate(valid_buffer_positions):
        if level <= buffer_compute_level:
            max_level = level
        else:
            break

    assert max_level is not None
    assert (
        max_level in valid_buffer_positions and max_level <= buffer_compute_level
    ), f"{max_level=}, {valid_buffer_positions=}, {buffer_compute_level=}"

    return max_level


def multi_level_buffer_insersion_pass(op_list, macro_compute_level):
    num_input_buffer_level = 2
    input_memory_names = ["__INPUT_MEMORY__", "__PIM_INPUT_REG_BUFFER__"]
    weight_memory_names = ["__MACRO__"]

    new_ops = []
    for op in debug_tqdm(op_list):
        input_buffer_level_combinations = buffer_level_serching(
            op, "I", num_buffer_level=1, level_min=0, level_max=macro_compute_level + 1
        )
        # input_buffer_level_combinations = input_buffer_level_combinations[2:]
        weight_buffer_level = get_macro_level(op, "W", macro_compute_level)
        for buffer_levels in input_buffer_level_combinations:
            op = op.convex_hull()  # Is this safe?
            new_op = insert_single_buffer_multi_level(
                op, "I", buffer_levels, input_memory_names
            )
            new_op = insert_single_buffer_multi_level(
                new_op, "W", [], weight_memory_names
            )
            new_op = new_op.convex_hull()
            # import pdb; pdb.set_trace()
            new_ops.append(new_op)
    return new_ops


def memory_access_cost(op):
    bandwidth_factor = {
        # ("__GLOBAL__", "__INPUT_MEMORY__"): 4,
        ("__INPUT_MEMORY__", "__PIM_INPUT_REG_BUFFER__"): 1024,
        # ("__PIM_INPUT_REG_BUFFER__"): 1,
        # ("__GLOBAL__", "__MACRO__"): 4
    }
    total_cost = 0
    # cost of moving I and W
    for buffer_name in ["I", "W"]:
        for datamove in op.data_movement[buffer_name]:
            # TODO: consider data type, int8 / int32
            # data_volumn = datamove.domain.count_val()

            domain = datamove.domain
            domain_size = domain.dim(isl.dim_type.set)

            n_access_O_dim = datamove.access_O.offsets.dim(isl.dim_type.out)
            n_access_I_dim = datamove.access_I.offsets.dim(isl.dim_type.out)
            n_inner_level = n_access_O_dim

            outer_domain = domain.project_out(
                isl.dim_type.set, domain_size - n_inner_level, n_inner_level
            )
            outer_domain_time = outer_domain.count_val()

            access_O_sizes = [
                int(str(datamove.access_O.offsets.range().dim_max_val(i)))
                for i in range(n_access_O_dim)
            ]
            access_I_sizes = [
                int(str(datamove.access_I.offsets.range().dim_max_val(i)))
                for i in range(n_access_I_dim)
            ]
            assert access_O_sizes == access_I_sizes[n_access_I_dim - n_inner_level :]
            tensorize_data_volumn = reduce(lambda x, y: x * y, access_O_sizes)

            src_memory_type = datamove.access_I.memory_type
            dst_memory_type = datamove.access_O.memory_type
            bandwidth = bandwidth_factor[(src_memory_type, dst_memory_type)]
            tensorize_time = math.ceil(tensorize_data_volumn / bandwidth)

            total_cost += tensorize_time * outer_domain_time

    # TODO: cost of moving O

    return total_cost


def get_name_and_shape(access):
    assert type(access) == AccessRelation
    offsets = access.offsets.range()
    sizes = [offsets.dim_max_val(i) + 1 for i in range(offsets.dim(isl.dim_type.set))]

    name = access.offsets.get_tuple_name(isl.dim_type.out)
    return name, sizes


@dataclass
class BufferInfo:
    name: str
    shape: list
    memory_type: str


def extract_buffer_defines(op):

    buffer_to_size = dict()
    buffer_to_memory_type = dict()

    def _update_shape(name, shape):
        if name not in buffer_to_size:
            buffer_to_size[name] = shape
        else:
            old_shape = buffer_to_size[name]
            assert len(old_shape) == len(shape), f"{old_shape=}, {shape=}"
            max_shape = [max(old_shape[i], shape[i]) for i in range(len(shape))]
            buffer_to_size[name] = max_shape

    def _update_memory_type(name, memory_type):
        if name not in buffer_to_memory_type:
            buffer_to_memory_type[name] = memory_type
        else:
            assert (
                buffer_to_memory_type[name] == memory_type
            ), f"{buffer_to_memory_type[name]=}, {memory_type=}"

    # import pdb; pdb.set_trace()
    I_name, I_shape = get_name_and_shape(op.access_I)  # this maybe incorrect.
    W_name, W_shape = get_name_and_shape(op.access_W)
    O_name, O_shape = get_name_and_shape(op.access_O)

    _update_shape(I_name, I_shape)
    _update_shape(W_name, W_shape)
    _update_shape(O_name, O_shape)

    _update_memory_type(I_name, op.access_I.memory_type)
    _update_memory_type(W_name, op.access_W.memory_type)
    _update_memory_type(O_name, op.access_O.memory_type)

    for buffer in ["I", "W"]:
        for data_movement in op.data_movement[buffer]:
            assert type(data_movement) == DataMovement

            name, shape = get_name_and_shape(data_movement.access_I)
            _update_shape(name, shape)
            _update_memory_type(name, data_movement.access_I.memory_type)

            name, shape = get_name_and_shape(data_movement.access_O)
            _update_shape(name, shape)
            _update_memory_type(name, data_movement.access_O.memory_type)

    buffer_name_to_info = dict()
    for name in buffer_to_size.keys():
        shape = buffer_to_size[name]
        memory_type = buffer_to_memory_type[name]
        buffer_name_to_info[name] = BufferInfo(
            name=name, shape=shape, memory_type=memory_type
        )
    return buffer_name_to_info


def memory_access_satisfy_constraint(op):
    buffer_name_to_info = extract_buffer_defines(op)

    buffer_type_to_use_size = dict()
    # get each memory type's use size
    for buffer_info in buffer_name_to_info.values():
        memory_type = buffer_info.memory_type
        buffer_type_to_use_size[memory_type] = buffer_type_to_use_size.get(
            memory_type, 0
        ) + reduce(lambda x, y: x * y, buffer_info.shape)

    buffer_type_to_size = get_memory_sizes()

    satisfy = True
    for memory_type, use_size in buffer_type_to_use_size.items():
        size_limit = buffer_type_to_size[memory_type]
        if memory_type == "__PIM_INPUT_REG_BUFFER__" and use_size > size_limit:
            satisfy = False
            logger.debug(f"Memory not satisfy! {memory_type=}, {use_size=}, {size_limit=}")
            break
    return satisfy


def filter_op_by_memory_access_cost_pass(op_list):
    op_list = [op for op in op_list if memory_access_satisfy_constraint(op)]
    memory_access_cost_list = [memory_access_cost(op) for op in op_list]
    # logger.debug(f"{memory_access_cost_list=}")
    # exit()
    memory_access_cost_list = np.array(memory_access_cost_list)
    sorted_indices = np.argsort(memory_access_cost_list)

    new_op_list = []
    new_op_list_memory_access = []
    min_value = memory_access_cost_list[sorted_indices[0]]
    num_ops = len(op_list)
    for i, index in enumerate(sorted_indices):
        if i < 3 or memory_access_cost_list[index] == min_value:
            new_op_list.append(op_list[index])
            new_op_list_memory_access.append(memory_access_cost_list[index])
            logger.debug(f"{i}. {memory_access_cost_list[index]=}")

    return new_op_list


if __name__ == "__main__":

    """
    domain: { [i0, i1, i2, i3, i4, i5] : i1 >= -2 + i0 and 0 <= i1 <= 63 and i1 <= i0 and 0 <= i4 <= 3 and 0 <= i5 <= 3 and 4*floor((i5)/4) >= -63 + 4i3 + i5 and 4*floor((i5)/4) <= 4i3 + i5 and -2 - 66i0 + 4i2 - 4i3 + i4 - i5 + 4*floor((i5)/4) <= 4*floor((i4)/4) <= -66i0 + 4i2 - 4i3 + i4 - i5 + 4*floor((i5)/4) }

    access_I: { [i0, i1, i2, i3, i4, i5] -> I[o0, o1] : o1 = i0 and (i4 + o0) mod 2 = 0 and (2i0 - i4 + o0) mod 4 = 0 and i1 >= -2 + i0 and 0 <= i1 <= 63 and i1 <= i0 and 0 <= i4 <= 3 and 0 <= i5 <= 3 and -66i0 + 4i2 <= o0 <= 3 - 66i0 + 4i2 and 4*floor((i5)/4) >= -63 + 4i3 + i5 and 4*floor((i5)/4) >= 4i3 + i5 - o0 and 4*floor((i5)/4) <= 2 + 4i3 + i5 - o0 and 4*floor((i5)/4) <= 4i3 + i5 }

    access_O: { [i0, i1, i2, i3, i4, i5] -> O[o0, o1] : o1 = i1 and (-i5 + o0) mod 4 = 0 and i1 >= -2 + i0 and 0 <= i1 <= 63 and i1 <= i0 and 0 <= i4 <= 3 and 0 <= i5 <= 3 and o0 >= 4i3 and 0 <= o0 <= 63 and o0 <= 3 + 4i3 and -2 - 66i0 + 4i2 + i4 - o0 <= 4*floor((i4)/4) <= -66i0 + 4i2 + i4 - o0 }

    access_W: { [i0, i1, i2, i3, i4, i5] -> W[o0, o1] : o1 = i0 - i1 and (i4 + i5 + o0) mod 2 = 0 and (2i0 - i4 + i5 + o0) mod 4 = 0 and i1 >= -2 + i0 and 0 <= i1 <= 63 and i1 <= i0 and 0 <= i4 <= 3 and 0 <= i5 <= 3 and 0 <= o0 <= 2 and 4*floor((i4)/4) >= -63 - 66i0 + 4i2 + i4 - o0 and -3 - 66i0 + 4i2 - 4i3 + i4 - o0 <= 4*floor((i4)/4) <= -66i0 + 4i2 - 4i3 + i4 - o0 and 4*floor((i4)/4) <= -66i0 + 4i2 + i4 - o0 }
    """
    operator = BasicOperator(
        domain=isl.BasicSet(
            "{ [i0, i1, i2, i3, i4, i5] : i1 >= -2 + i0 and 0 <= i1 <= 63 and i1 <= i0 and 0 <= i4 <= 3 and 0 <= i5 <= 3 and 4*floor((i5)/4) >= -63 + 4i3 + i5 and 4*floor((i5)/4) <= 4i3 + i5 and -2 - 66i0 + 4i2 - 4i3 + i4 - i5 + 4*floor((i5)/4) <= 4*floor((i4)/4) <= -66i0 + 4i2 - 4i3 + i4 - i5 + 4*floor((i5)/4) }"
        ),
        access_I=isl.BasicMap(
            "{ [i0, i1, i2, i3, i4, i5] -> I[o0, o1] : o1 = i0 and (i4 + o0) mod 2 = 0 and (2i0 - i4 + o0) mod 4 = 0 and i1 >= -2 + i0 and 0 <= i1 <= 63 and i1 <= i0 and 0 <= i4 <= 3 and 0 <= i5 <= 3 and -66i0 + 4i2 <= o0 <= 3 - 66i0 + 4i2 and 4*floor((i5)/4) >= -63 + 4i3 + i5 and 4*floor((i5)/4) >= 4i3 + i5 - o0 and 4*floor((i5)/4) <= 2 + 4i3 + i5 - o0 and 4*floor((i5)/4) <= 4i3 + i5 }"
        ),
        access_O=isl.BasicMap(
            "{ [i0, i1, i2, i3, i4, i5] -> O[o0, o1] : o1 = i1 and (-i5 + o0) mod 4 = 0 and i1 >= -2 + i0 and 0 <= i1 <= 63 and i1 <= i0 and 0 <= i4 <= 3 and 0 <= i5 <= 3 and o0 >= 4i3 and 0 <= o0 <= 63 and o0 <= 3 + 4i3 and -2 - 66i0 + 4i2 + i4 - o0 <= 4*floor((i4)/4) <= -66i0 + 4i2 + i4 - o0 }"
        ),
        access_W=isl.BasicMap(
            "{ [i0, i1, i2, i3, i4, i5] -> W[o0, o1] : o1 = i0 - i1 and (i4 + i5 + o0) mod 2 = 0 and (2i0 - i4 + i5 + o0) mod 4 = 0 and i1 >= -2 + i0 and 0 <= i1 <= 63 and i1 <= i0 and 0 <= i4 <= 3 and 0 <= i5 <= 3 and 0 <= o0 <= 2 and 4*floor((i4)/4) >= -63 - 66i0 + 4i2 + i4 - o0 and -3 - 66i0 + 4i2 - 4i3 + i4 - o0 <= 4*floor((i4)/4) <= -66i0 + 4i2 - 4i3 + i4 - o0 and 4*floor((i4)/4) <= -66i0 + 4i2 + i4 - o0 }"
        ),
    )
    new_op = insert_single_buffer_multi_level(operator, "I", [4])
    # logger.debug(code)


def test():
    domain = isl.BasicSet("{ [i,j,k]: 0 <= i < 4 and 0 <= j < 4 and 0 <= k < 4}")
    acc_rel = isl.BasicMap("{ [i,j,k] -> A[j,k] }")
    # acc_rel2=isl.BasicMap("{ [i,j,k] -> B[i,j] }")
    domain, acc_rel = apply_skew(domain, acc_rel)
    domain, acc_rel = apply_tile(domain, acc_rel)
    print(f"{domain = }")
    print(f"{acc_rel = }")
    print("----------------------------")
    # acc_rel=isl.BasicMap("{ [i,j,k] -> A[i * 2, k] }")
    map_buf_align_to_ori, aligned_acc_rel = (
        map_domain_aligned_buffer_to_origin_buffer_v2(domain, acc_rel)
    )
    # map_buf_align_to_ori2, aligned_acc_rel2 = map_domain_aligned_buffer_to_origin_buffer_v2(domain, acc_rel2)
    print(f"{aligned_acc_rel = }")
    print("----------------------------")
    assign_domain, local_buffer_acc_rel, assign_buffer_acc_rel = (
        map_prefix_domain_aligned_buffer_to_aligned_buffer_v2(
            domain, aligned_acc_rel, 4
        )
    )
    # assign_domain2, local_buffer_acc_rel2, assign_buffer_acc_rel2 = map_prefix_domain_aligned_buffer_to_aligned_buffer_v2(domain, aligned_acc_rel2, 4)
    print(f"{assign_domain = }")
    print(f"{local_buffer_acc_rel = }")
    print(f"{assign_buffer_acc_rel = }")

    print("----------------------------")
    print(f"{domain=}")
    print(f"{assign_domain=}")
    compute_schedule = utils.identity_map_from_set(domain)
    print(f"{compute_schedule=}")
    assign_schedule = utils.identity_map_from_set(assign_domain)
    # assign_schedule2 = utils.identity_map_from_set(assign_domain2)
    print(f"{assign_schedule=}")

    compute_domain = domain.set_tuple_name("S")
    compute_schedule = compute_schedule.set_tuple_name(isl.dim_type.in_, "S")

    assign_domain = assign_domain.set_tuple_name("T")
    assign_schedule = assign_schedule.set_tuple_name(isl.dim_type.in_, "T")

    # assign_domain2 = assign_domain2.set_tuple_name("P")
    # assign_schedule2 = assign_schedule2.set_tuple_name(isl.dim_type.in_, "P")

    union_domain = compute_domain.add_set(assign_domain)  # .add_set(assign_domain2)
    # assign_schedule2
    union_schedule = align_compute_and_assign_schedules(
        compute_schedule, [assign_schedule], [4]
    )
    print("--------------------------------------------")
    print(f"{type(union_domain)}, {union_domain=}\n")
    print(f"{type(union_schedule)}, {union_schedule=}\n")
    ast = utils.gen_ast(union_domain, union_schedule, None)
    code = utils.gen_code(union_domain, union_schedule, None)
    print(ast, "\n")
    print(code)
    print(type(ast), ast.get_type())
    print("\n-------------------------------------\n")

    from ast_ import codegen_str

    print(codegen_str(ast))

    exit()
    print(f"{assign_buffer_acc_rel = }")
    print("----------------------------------")
    pma = assign_buffer_acc_rel.as_pw_multi_aff()

    def show(cond, ma):
        print(f"- {cond = }")
        print(f"- {ma = }")
        filter_acc_rel = assign_buffer_acc_rel.intersect_domain(cond)
        print(f"- {filter_acc_rel = }")
        print("")

    pma.foreach_piece(show)
