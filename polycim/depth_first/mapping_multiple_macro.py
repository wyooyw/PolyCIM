import islpy as isl
from polycim.depth_first.count_minimal_macro import (
    get_non_dominate_iters_of_pw_multi_aff,
    get_dominate_iters_of_pw_multi_aff,
)
import polycim.utils.utils as utils
from polycim.passes.buffer_mapping import insert_single_buffer_multi_level
from polycim.codegen_.codegen_cimdsl import codegen_pass
from polycim.passes.tensorize import tensorize_pass
from polycim.passes.backend import backend_compile_and_profile_pass
from polycim.utils.math import get_factors
import os
from functools import reduce

def get_scalar_iters(domain):
    shape = utils.get_box_hull_shape(domain)
    n_dim = domain.dim(isl.dim_type.set)
    scalar_iters = {i for i in range(n_dim) if shape[i] == 1}
    return scalar_iters

def mapping_multiple_macro(op, cim_cfg, **kwargs):
    assert "enable_weight_rewrite" in kwargs
    if kwargs["enable_weight_rewrite"]:
        return mapping_multiple_macro_enable_weight_rewrite(op, cim_cfg, **kwargs)
    else:
        return mapping_multiple_macro_disable_weight_rewrite(op, cim_cfg, **kwargs)

def get_candidate_iters(op):
    share_input_iters = get_non_dominate_iters_of_pw_multi_aff(op.access_I.as_pw_multi_aff(), return_name=False)
    share_output_iters = get_non_dominate_iters_of_pw_multi_aff(op.access_O.as_pw_multi_aff(), return_name=False)
    share_weight_iters = get_non_dominate_iters_of_pw_multi_aff(op.access_W.as_pw_multi_aff(), return_name=False)

    n_dim = op.domain.dim(isl.dim_type.set)
    macro_iters = {n_dim - 1, n_dim - 2}
    scalar_iters = get_scalar_iters(op.domain)
    ignore_iters = scalar_iters | macro_iters
    
    candidate_share_weight_iters = share_weight_iters - ignore_iters
    candidate_share_weight_iters = list(candidate_share_weight_iters)
    candidate_share_weight_iters = sorted(candidate_share_weight_iters, key=lambda x: -x)

    candidate_iters = candidate_share_weight_iters
    
    return candidate_iters

def make_group_schedule(op, candidate_iters, cim_cfg):
    
    shape = utils.get_box_hull_shape(op.domain)
    n_dim = op.domain.dim(isl.dim_type.set)
    n_group = cim_cfg.n_group
    remain_group_factor = n_group

    in_group_iters = []
    for candidate_iter in candidate_iters:
        iter_size = shape[candidate_iter]

        # find max factor <= remain_group_factor
        factors = get_factors(iter_size)
        factors = sorted(factors, key=lambda x: -x)
        for factor in factors:
            if factor <= remain_group_factor:
                break
        
        if factor == 1:
            break
        elif factor == iter_size:
            in_group_iters.append(candidate_iter)
            remain_group_factor //= factor
        elif factor > 1 and factor < iter_size:
            break
            remain_group_factor //= factor
        else:
            raise ValueError(f"factor={factor} is invalid")

    in_group_iters = in_group_iters[::-1]
    if len(in_group_iters) == 0:
        in_group_iters = ["0"]
    row_iter = ["0"]
    comp_iter = [n_dim - 2]
    col_iter = [n_dim - 1]
    other_iters = [i for i in range(n_dim) if i not in (in_group_iters + comp_iter + col_iter)]

    name = lambda ls: [f"i{i}" if type(i) == int else i for i in ls]

    old_iter_names = [f"i{i}" for i in range(n_dim)]
    new_iter_names = name(other_iters) + row_iter + name(comp_iter) + name(in_group_iters) + name(col_iter)
    # import pdb; pdb.set_trace()
    reorder_schedule = isl.BasicMap(f"{{ [{','.join(old_iter_names)}] -> [{','.join(new_iter_names)}] }}")
    
    n_macro_iters = len(row_iter) + len(in_group_iters) + len(comp_iter) + len(col_iter)

    if len(in_group_iters) == 1 and in_group_iters[0] == "0":
        n_use_group = 1
    else:
        assert len(in_group_iters) >= 1
        assert all([type(i) == int for i in in_group_iters])
        n_use_group = reduce(lambda x, y: x * y, [shape[i] for i in in_group_iters], 1)

    return reorder_schedule, n_macro_iters, n_use_group

def mapping_multiple_macro_enable_weight_rewrite(op, cim_cfg, **kwargs):
    """
    for outer_iters_dominate_weight:
        Load weights
        for outer_iters:
            for i in [0, n_macro_share_output):
                Move inputs
            for macro_i in [0, n_macro_share_input):
                for macro_j in [0, n_macro_share_output):
                    Macro compute
            for j in [0, n_macro_share_input):
                Add partial sums
                Write back
    """
    domain = op.domain
    shape = utils.get_box_hull_shape(domain)
    n_dim = domain.dim(isl.dim_type.set)
    inner_iters = {n_dim - i for i in range(1,5)}
    # import pdb; pdb.set_trace()
    
    # set_attr
    # n_use_group = shape[n_dim - 3] * shape[n_dim - 4]
    # op.set_attr("n_use_group", n_use_group)
    
    # reorder
    # step 1: get candidate iters mapping to groups
    candidate_iters = get_candidate_iters(op)
    
    # step 2: try add candidate iters to group
    reorder_schedule, n_macro_iters, n_use_group = make_group_schedule(op, candidate_iters, cim_cfg)
    op.set_attr("n_use_group", n_use_group)

    # import pdb; pdb.set_trace()
    # dominate_weight_iters = get_dominate_iters_of_pw_multi_aff(op.access_W.as_pw_multi_aff(), return_name=False)
    # assert len(dominate_weight_iters)==2
    # assert (n_dim - 1) in dominate_weight_iters
    # assert (n_dim - 2) in dominate_weight_iters
    # def make_reorder_schedule():
    #     keep_iters = [f"i{i}" for i in range(0, n_dim - 4)]
    #     group_iters = [f"i{i}" for i in [n_dim - 4, n_dim - 3]]
    #     comp_iter = [f"i{n_dim - 2}"]
    #     col_iter = [f"i{n_dim - 1}"]
    #     row_iter = ["0"]
    #     old_order = keep_iters + group_iters + comp_iter + col_iter
    #     new_order = keep_iters + row_iter + comp_iter + group_iters + col_iter
    #     reorder_schedule = isl.BasicMap(f"{{ [{','.join(old_order)}] -> [{','.join(new_order)}] }}")
    #     return reorder_schedule
    # reorder_schedule = make_reorder_schedule()
    op = op.apply_schedule(reorder_schedule)
    # import pdb; pdb.set_trace()

    # n_dim - 1: share input, one row
    # n_dim - 2: share output, one column
    # n_dim - 3: share input, macros
    # n_dim - 4: share output, macros

    new_op = multi_level_buffer_insersion_pass(op, n_macro_iters)
    return new_op

def multi_level_buffer_insersion_pass(op, n_macro_iters):
    n_dim = op.domain.dim(isl.dim_type.set)

    input_memory_names = ["global", "input_memory", "pim_input_reg_buffer"]
    output_memory_names = ["global", "output_memory", "pim_output_reg_buffer"]
    weight_memory_names = ["global", "macro"]

    """
    example for 'level':
    // level 0
    for i0
        // level 1
        for i_1
        ...
            // level n-1
            for i_{n-1}
                // level n
    """
    
    input_buffer_level = (0, n_dim - (n_macro_iters - 1)) # minus 1 is the row dimension
    output_buffer_level = (0, n_dim - (n_macro_iters - 1))
    weight_buffer_level = (0,)

    # n_macro_iters: [row, comp, group0,...,groupk, col]
    n_group_iters = n_macro_iters - 3
    group_iters = [n_dim - n_macro_iters + 2 + i for i in range(n_group_iters)]
    comp_iter = [n_dim - n_macro_iters + 1]
    input_layout_inner_dims = group_iters + comp_iter

    shape = utils.get_box_hull_shape(op.domain)
    n_dim = op.domain.dim(isl.dim_type.set)
    import pdb; pdb.set_trace()
    new_op = op.convex_hull()  # Is this safe?
    new_op, layout_convert_code_I = insert_single_buffer_multi_level(
        new_op, "I", input_buffer_level, input_memory_names, 
        # force_dominate_iters=[n_dim-2],
        force_nondominate_iters = [n_dim-1],
        force_layout_inner_iters = input_layout_inner_dims
    )
    new_op, layout_convert_code_O = insert_single_buffer_multi_level(
        new_op, "O", output_buffer_level, output_memory_names
    )
    new_op, layout_convert_code_W = insert_single_buffer_multi_level(
        new_op, "W", weight_buffer_level, weight_memory_names,
        force_inner_level=n_macro_iters
    )
    new_op = new_op.convex_hull()
    
    new_op.attr["n_tensorize_cim_compute_level"] = n_macro_iters - 1

    # print("weight:")
    # for data_movement in new_op.data_movement["W"]:
    #     print(f"{data_movement.level=}")
    #     print(f"{data_movement.access_O=}")
    #     print(f"{data_movement.access_I=}\n")

    # print("input:")
    # for data_movement in new_op.data_movement["I"]:
    #     print(f"{data_movement.level=}")
    #     print(f"{data_movement.access_O=}")
    #     print(f"{data_movement.access_I=}\n")

    # print("output:")
    # for data_movement in new_op.data_movement["O"]:
    #     print(f"{data_movement.level=}")
    #     print(f"{data_movement.access_O=}")
    #     print(f"{data_movement.access_I=}\n")
    
    # import pdb; pdb.set_trace()
    data_layout_convert_code = {
        "I": layout_convert_code_I,
        "O": layout_convert_code_O,
        "W": layout_convert_code_W
    }
    new_op.attr["data_layout_convert_code"] = data_layout_convert_code

    return new_op

def mapping_multiple_macro_disable_weight_rewrite(op, cim_cfg, **kwargs):
    pass