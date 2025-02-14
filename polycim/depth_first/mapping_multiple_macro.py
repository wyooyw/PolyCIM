import islpy as isl
from polycim.utils.dominate import (
    get_dominate_iters_of_pw_multi_aff,
    get_non_dominate_iters_of_pw_multi_aff
)
import polycim.utils.utils as utils
from polycim.passes.buffer_mapping import insert_single_buffer_multi_level
from polycim.codegen_.codegen_cimdsl import codegen_pass
from polycim.passes.tensorize import tensorize_pass
from polycim.passes.backend import backend_compile_and_profile_pass
from polycim.utils.math import get_factors
import os
from functools import reduce
from polycim.passes.loop_padding import loop_padding_dim
import math
from polycim.utils.logger import get_logger

logger = get_logger(__name__)

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

from dataclasses import dataclass

@dataclass
class ScalarIter:
    pass

@dataclass
class Iter:
    iter_id: int
    iter_size: int

@dataclass
class TiledIter(Iter):
    tile_size: int
    is_inner: bool
    
def make_group_schedule(op, candidate_iters, cim_cfg):
    
    shape = utils.get_box_hull_shape(op.domain)
    n_dim = op.domain.dim(isl.dim_type.set)
    n_group = cim_cfg.n_group
    remain_group_factor = n_group

    in_group_iters = []
    n_use_group = 1

    candidate_iters = [Iter(iter_id=i, iter_size=shape[i]) for i in candidate_iters]
    
    for candidate_iter in candidate_iters:
        iter_size = candidate_iter.iter_size

        if remain_group_factor==1:
            break

        factors = get_factors(remain_group_factor)
        factors = sorted(factors) # small to big
        for factor in factors:
            if factor >= iter_size:
                break
        
        if factor >= iter_size:
            padded_iter_size = factor
            in_group_iters.append(Iter(
                iter_id=candidate_iter.iter_id, 
                iter_size=padded_iter_size
            ))
            remain_group_factor //= factor
            n_use_group *= factor
        elif factor < iter_size:
            assert factor == remain_group_factor, f"factor={factor} is invalid"
            in_group_iters.append(TiledIter(
                iter_id=candidate_iter.iter_id, 
                iter_size=factor,
                tile_size=factor,
                is_inner=True
            ))
            remain_group_factor //= factor
            n_use_group *= factor
            break
        else:
            raise ValueError(f"factor={factor} is invalid")
    # import pdb; pdb.set_trace()
    assert remain_group_factor==1, f"Currently, only support use all groups. When meet the situation that remain some group, it should be fixed."

    in_group_iters = in_group_iters[::-1]
    if len(in_group_iters) == 0:
        in_group_iters = [ScalarIter()]
    row_iter = [ScalarIter()]
    comp_iter = [Iter(iter_id=n_dim - 2, iter_size=shape[n_dim - 2])]
    col_iter = [Iter(iter_id=n_dim - 1, iter_size=shape[n_dim - 1])]

    id_to_iter = {iter_.iter_id:iter_ for iter_ in (in_group_iters + comp_iter + col_iter) if not isinstance(iter_, ScalarIter)}
    other_iters = []
    for i in range(n_dim):
        if i not in id_to_iter:
            other_iters.append(Iter(iter_id=i, iter_size=shape[i]))
        elif isinstance(id_to_iter[i], TiledIter):
            ori_iter_size = shape[id_to_iter[i].iter_id]
            outer_iter_size = int(math.ceil(ori_iter_size / id_to_iter[i].tile_size))
            other_iters.append(TiledIter(
                iter_id=id_to_iter[i].iter_id,
                iter_size=outer_iter_size,
                tile_size=id_to_iter[i].tile_size,
                is_inner=False
            ))

    def iter_to_str(iter_):
        if isinstance(iter_, TiledIter):
            if iter_.is_inner:
                return f"(i{iter_.iter_id}%{iter_.tile_size})"
            else:
                return f"floor(i{iter_.iter_id}/{iter_.tile_size})"
        elif isinstance(iter_, Iter):
            return f"i{iter_.iter_id}"
        elif isinstance(iter_, ScalarIter):
            return "0"
        else:
            raise ValueError(f"iter_={iter_} is invalid")

    new_iters = other_iters + row_iter + comp_iter + in_group_iters + col_iter
    old_iter_names = [f"i{i}" for i in range(n_dim)]
    new_iter_names = [iter_to_str(i) for i in new_iters]
    
    reorder_schedule = isl.BasicMap(f"{{ [{','.join(old_iter_names)}] -> [{','.join(new_iter_names)}] }}")
    
    n_macro_iters = len(row_iter) + len(in_group_iters) + len(comp_iter) + len(col_iter)
    n_use_comp = shape[comp_iter[0].iter_id]

    # padding in-group dims
    for iter_ in in_group_iters:

        if isinstance(iter_, ScalarIter):
            continue
        elif isinstance(iter_, TiledIter):
            ori_size = shape[iter_.iter_id]
            padded_size = int(math.ceil(ori_size / iter_.tile_size) * iter_.tile_size)
            assert padded_size >= ori_size, f"{iter_.iter_id} padded_size={padded_size} should be greater than ori_size={ori_size}"
            if padded_size > ori_size:
                op = loop_padding_dim(op, iter_.iter_id, padded_size)
                logger.debug(f"padding {iter_.iter_id} from {ori_size} to {padded_size}")
        elif isinstance(iter_, Iter):
            ori_size = shape[iter_.iter_id]
            padded_size = iter_.iter_size
            logger.debug(f"{iter_.iter_id} padded_size={padded_size} ori_size={ori_size}")
            assert padded_size >= ori_size, f"{iter_.iter_id} padded_size={padded_size} should be greater than ori_size={ori_size}"
            if padded_size > ori_size:
                op = loop_padding_dim(op, iter_.iter_id, padded_size)
                logger.debug(f"padding {iter_.iter_id} from {ori_size} to {padded_size}")

    

    op = op.apply_schedule(reorder_schedule, skip_simplify=True)

    new_shape = utils.get_box_hull_shape(op.domain)
    logger.debug(f"{shape=}")
    logger.debug(f"{new_shape=}")

    return op, n_macro_iters, n_use_group, n_use_comp

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
    op, n_macro_iters, n_use_group, n_use_comp = make_group_schedule(op, candidate_iters, cim_cfg)
    op.set_attr("n_use_group", n_use_group)
    op.set_attr("n_use_comp", n_use_comp)

    # padding macro dimensions
    n_dim = op.domain.dim(isl.dim_type.set)
    iter_col = n_dim - 1
    op = loop_padding_dim(op, iter_col, cim_cfg.n_group_vcol)

    # insert buffer access
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
    # import pdb; pdb.set_trace()
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