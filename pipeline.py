import json
import tempfile

import islpy as isl

import op_define
from base_operator import BasicOperator
from config import get_config
from pass_.affine_transform import auto_skewing_pass
from pass_.backend import backend_compile_and_profile_pass
from pass_.buffer_mapping import (filter_op_by_memory_access_cost_pass,
                                  insert_single_buffer_multi_level_pass,
                                  insert_single_buffer_single_level_pass,
                                  multi_level_buffer_insersion_pass)
from pass_.codegen import codegen_pass
# from hardware_merge_tiling import hardware_merge_tiling_pass, filter_op_by_execution_time_pass
from pass_.hardware_merge_tiling_4d_macro import (
    filter_op_by_execution_time_pass, hardware_merge_tiling_pass)
from pass_.loop_padding import loop_padding_pass
from pass_.multi_level_tiling import memory_tiling_pass, pre_tiling_pass
from pass_.tensorize import tensorize_pass


def run_pipeline(op, skew, cim_cfg, save_dir):
    new_ops = [op]

    if skew:
        new_ops = pre_tiling_pass(new_ops)
        new_ops = auto_skewing_pass(
            new_ops,
            max_reuse_factor_for_arrays=(cim_cfg.n_group_vcol, cim_cfg.n_comp),
            return_detail=False,
        )
        print(f"after auto_skewing_pass, {len(new_ops)=}")

    new_ops = hardware_merge_tiling_pass(new_ops)
    new_ops = filter_op_by_execution_time_pass(new_ops)
    new_ops = loop_padding_pass(new_ops, padding_inner_size=None)
    new_ops = memory_tiling_pass(new_ops)

    new_ops = multi_level_buffer_insersion_pass(new_ops, macro_compute_level=-6)
    new_ops = filter_op_by_memory_access_cost_pass(new_ops)
    new_ops = new_ops[:1] if len(new_ops) >= 1 else []
    new_ops = tensorize_pass(new_ops)
    new_ops = codegen_pass(new_ops)
    result_list = backend_compile_and_profile_pass(new_ops, save_dir)

    return result_list


if __name__ == "__main__":
    # operator = BasicOperator(
    #     domain = isl.BasicSet(
    #         f"{{ [v0,oh,ow,kh,kw]: v0=0 and 0<=oh<4 and 0<=ow<4 and 0<=kh<3 and 0<=kw<3 }}"
    #     ),
    #     access_I = isl.BasicMap("{ [v0,oh,ow,kh,kw] -> I[oh + kh, ow + kw] }"),
    #     access_O = isl.BasicMap("{ [v0,oh,ow,kh,kw] -> O[v0,oh, ow] }"),
    #     access_W = isl.BasicMap("{ [v0,oh,ow,kh,kw] -> W[v0,kh, kw] }"),
    # )
    skew = False
    virtual_axis = not skew
    operator = op_define.get_op_conv2d(
        b=1,
        oc=64,
        ic=64,
        oh=32,
        ow=32,
        kh=3,
        kw=3,
        stride=1,
        virtual_axis=virtual_axis,
    )
    # print(operator.access_W)
    # print(operator.access_W.intersect_domain(operator.domain))
    print(operator.domain.count_val())
    cim_cfg = get_config()
    temp_dir = tempfile.mkdtemp()
    run_pipeline(operator, skew=skew, cim_cfg=cim_cfg, save_dir=temp_dir)
