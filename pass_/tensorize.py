import islpy as isl
from tqdm import tqdm

import utils.utils as utils
from base_operator import (AccessRelation, BasicOperator, DataMovement,
                           DataMovementOperator, TensorAccessRelation)


def pwaffs_to_map(affs):
    pw_aff_list = utils.make_pw_affs_to_aff_list(affs)

    assign_buffer_acc_rel = isl.MultiPwAff.from_pw_aff_list(
        affs[0].space.insert_dims(isl.dim_type.out, 0, len(pw_aff_list) - 1),
        pw_aff_list,
    )
    assign_buffer_acc_rel = isl.PwMultiAff.from_multi_pw_aff(assign_buffer_acc_rel)
    assign_buffer_acc_rel = isl.Map.from_pw_multi_aff(assign_buffer_acc_rel)
    return assign_buffer_acc_rel


def transform_access(ori_access, n_inner_level):
    if isinstance(ori_access, AccessRelation):
        access = ori_access.offsets
    else:
        access = ori_access
    assert type(access) in (isl.BasicMap, isl.Map)

    access_domain_size = access.dim(isl.dim_type.in_)
    access_range_size = access.dim(isl.dim_type.out)
    outer_access = access.project_out(
        isl.dim_type.in_, access_domain_size - n_inner_level, n_inner_level
    )
    # if not outer_access.range().is_bounded():
    #     import pdb; pdb.set_trace()
    pwaff_access_sizes = utils.get_dynamic_shape_from_dynamic_map(
        outer_access
    )  # [level:]
    pwaff_access_offsets = [outer_access.dim_min(i) for i in range(access_range_size)]

    access_sizes = pwaffs_to_map(pwaff_access_sizes)
    access_offsets = pwaffs_to_map(pwaff_access_offsets)

    buffer_name = access.get_tuple_name(isl.dim_type.out)
    access_sizes = access_sizes.set_tuple_name(isl.dim_type.out, buffer_name)
    access_offsets = access_offsets.set_tuple_name(isl.dim_type.out, buffer_name)

    return TensorAccessRelation(
        offsets=access_offsets, sizes=access_sizes, memory_type=ori_access.memory_type
    )


def tensorize_cim_compute(op):
    domain = op.domain
    domain_size = domain.dim(isl.dim_type.set)
    n_inner_level = 4

    outer_domain = domain.project_out(
        isl.dim_type.set, domain_size - n_inner_level, n_inner_level
    )
    access_I = transform_access(op.access_I, n_inner_level)
    access_O = transform_access(op.access_O, n_inner_level)
    access_W = transform_access(op.access_W, n_inner_level)

    return DataMovementOperator(
        domain=outer_domain,
        access_I=access_I,
        access_O=access_O,
        access_W=access_W,
        history_domains=[*op.history_domains, outer_domain],
        history_schedules=[
            *op.history_schedules,
            {"tensorize_cim_compute": n_inner_level},
        ],
        data_movement=op.data_movement,
        attr={key: value for key, value in op.attr.items()},
    )


def vectorize_data_movement(data_movement):
    domain = data_movement.domain
    domain_size = domain.dim(isl.dim_type.set)

    n_access_O_dim = data_movement.access_O.offsets.dim(isl.dim_type.out)
    n_access_I_dim = data_movement.access_I.offsets.dim(isl.dim_type.out)
    n_inner_level = n_access_O_dim

    outer_domain = domain.project_out(
        isl.dim_type.set, domain_size - n_inner_level, n_inner_level
    )

    access_O_sizes = [
        data_movement.access_O.offsets.range().dim_max_val(i)
        for i in range(n_access_O_dim)
    ]
    access_I_sizes = [
        data_movement.access_I.offsets.range().dim_max_val(i)
        for i in range(n_access_I_dim)
    ]
    assert access_O_sizes == access_I_sizes[n_access_I_dim - n_inner_level :]

    access_I = transform_access(data_movement.access_I, n_inner_level)
    access_O = transform_access(data_movement.access_O, n_inner_level)

    return DataMovement(
        domain=outer_domain,
        access_I=access_I,
        access_O=access_O,
        level=data_movement.level,
    )


def vectorize_data_movement_for_op(op):
    origin_data_movement = op.data_movement

    new_data_movement_dict = dict()
    for array_name, data_movement_list in origin_data_movement.items():
        new_data_movement_list = []
        for data_movement in data_movement_list:
            new_data_movement = vectorize_data_movement(data_movement)
            new_data_movement_list.append(new_data_movement)
        new_data_movement_dict[array_name] = new_data_movement_list

    return DataMovementOperator(
        domain=op.domain,
        access_I=op.access_I,
        access_O=op.access_O,
        access_W=op.access_W,
        history_domains=op.history_domains,
        history_schedules=op.history_schedules,
        data_movement=new_data_movement_dict,
        attr={key: value for key, value in op.attr.items()},
    )


def tensorize_pass(op_list):
    new_op_list = []
    for op in tqdm(op_list):

        new_op = tensorize_cim_compute(op)
        new_op = vectorize_data_movement_for_op(new_op)
        new_op_list.append(new_op)
    return new_op_list
