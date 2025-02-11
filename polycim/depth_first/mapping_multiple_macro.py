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
import os

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
    n_use_group = shape[n_dim - 3] * shape[n_dim - 4]
    op.set_attr("n_use_group", n_use_group)
    
    dominate_input_iters = get_dominate_iters_of_pw_multi_aff(op.access_I.as_pw_multi_aff(), return_name=False)
    # import pdb; pdb.set_trace()
    # share_input_iters = get_non_dominate_iters_of_pw_multi_aff(op.access_I.as_pw_multi_aff(), return_name=False)
    # share_output_iters = get_non_dominate_iters_of_pw_multi_aff(op.access_O.as_pw_multi_aff(), return_name=False)
    
    dominate_weight_iters = get_dominate_iters_of_pw_multi_aff(op.access_W.as_pw_multi_aff(), return_name=False)
    outer_dominate_weight_iters = dominate_weight_iters - inner_iters
    assert len(outer_dominate_weight_iters)==0, "Currently not support weight movement multiple times, this will be supported in the future"
    # import pdb; pdb.set_trace()

    # reorder
    # reorder_schedule = make_reorder_schedule(op)
    assert len(dominate_weight_iters)==2
    assert (n_dim - 1) in dominate_weight_iters
    assert (n_dim - 2) in dominate_weight_iters
    def make_reorder_schedule():
        keep_iters = [f"i{i}" for i in range(0, n_dim - 4)]
        group_iters = [f"i{i}" for i in [n_dim - 4, n_dim - 3]]
        comp_iter = [f"i{n_dim - 2}"]
        col_iter = [f"i{n_dim - 1}"]
        row_iter = ["0"]
        old_order = keep_iters + group_iters + comp_iter + col_iter
        new_order = keep_iters + row_iter + comp_iter + group_iters + col_iter
        reorder_schedule = isl.BasicMap(f"{{ [{','.join(old_order)}] -> [{','.join(new_order)}] }}")
        return reorder_schedule
    reorder_schedule = make_reorder_schedule()
    op = op.apply_schedule(reorder_schedule)
    # import pdb; pdb.set_trace()

    # n_dim - 1: share input, one row
    # n_dim - 2: share output, one column
    # n_dim - 3: share input, macros
    # n_dim - 4: share output, macros

    new_op = multi_level_buffer_insersion_pass(op)
    return new_op

def multi_level_buffer_insersion_pass(op):
    n_dim = op.domain.dim(isl.dim_type.set)

    num_input_buffer_level = 2
    input_memory_names = ["global", "input_memory", "pim_input_reg_buffer"]
    output_memory_names = ["global", "output_memory", "pim_output_reg_buffer"]
    weight_memory_names = ["global", "macro"]

    input_buffer_level = (0, n_dim-4)
    output_buffer_level = (0, n_dim-4)
    weight_buffer_level = (0,)
    new_op = op.convex_hull()  # Is this safe?
    new_op, layout_convert_code_I = insert_single_buffer_multi_level(
        new_op, "I", input_buffer_level, input_memory_names, 
        force_dominate_iters=[n_dim-2, n_dim-4],
        force_nondominate_iters=[n_dim-1, n_dim-3]
    )
    new_op, layout_convert_code_O = insert_single_buffer_multi_level(
        new_op, "O", output_buffer_level, output_memory_names
    )
    new_op, layout_convert_code_W = insert_single_buffer_multi_level(
        new_op, "W", weight_buffer_level, weight_memory_names
    )
    new_op = new_op.convex_hull()

    data_layout_convert_code = {
        "I": layout_convert_code_I,
        "O": layout_convert_code_O,
        "W": layout_convert_code_W
    }
    new_op.attr["data_layout_convert_code"] = data_layout_convert_code

    return new_op
    
# def mapping_multiple_macro_enable_weight_rewrite(op, cim_cfg, **kwargs):
    # import pdb; pdb.set_trace()
    # domain = op.domain
    # shape = utils.get_box_hull_shape(domain)
    # n_dim = domain.dim(isl.dim_type.set)
    
    # dominate_weight_iters = get_dominate_iters_of_pw_multi_aff(op.access_W.as_pw_multi_aff(), return_name=False)
    # dominate_input_iters = get_dominate_iters_of_pw_multi_aff(op.access_I.as_pw_multi_aff(), return_name=False)
    # dominate_output_iters = get_dominate_iters_of_pw_multi_aff(op.access_O.as_pw_multi_aff(), return_name=False)
    # share_weight_iters = get_non_dominate_iters_of_pw_multi_aff(op.access_W.as_pw_multi_aff(), return_name=False)
    # share_input_iters = get_non_dominate_iters_of_pw_multi_aff(op.access_I.as_pw_multi_aff(), return_name=False)
    # share_output_iters = get_non_dominate_iters_of_pw_multi_aff(op.access_O.as_pw_multi_aff(), return_name=False)
    
    # cim_iters = {n_dim - 2, n_dim - 1}
    # scalar_iters = get_scalar_iters(domain)
    
    # dominate_weight_iters = dominate_weight_iters - scalar_iters - cim_iters
    # dominate_input_iters = dominate_input_iters - scalar_iters - cim_iters
    # dominate_output_iters = dominate_output_iters - scalar_iters - cim_iters
    # share_weight_iters = share_weight_iters - scalar_iters - cim_iters
    # share_input_iters = share_input_iters - scalar_iters - cim_iters
    # share_output_iters = share_output_iters - scalar_iters - cim_iters

    # share_output_and_not_dominate_weight_iters = share_output_iters - dominate_weight_iters
    # share_input_and_not_dominate_weight_iters = share_input_iters - dominate_weight_iters

    # order:
    # for {dominate_weight_iters}
    #   for {share_weight_iters}
    #     for {share_input_and_not_dominate_weight_iters}
    #       for {share_output_and_not_dominate_weight_iters}
    # map as much as possible loop iters into inter-macro dimension
    # the remaining loop iters are time dimensions
    
    

    # num_macros = kwargs["num_macros"]

    # import pdb; pdb.set_trace()
    # pass


def mapping_multiple_macro_disable_weight_rewrite(op, cim_cfg, **kwargs):
    pass