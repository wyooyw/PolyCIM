from base_operator import BasicOperator
from affine_transform import auto_skewing_pass
from hardware_merge_tiling import hardware_merge_tiling_pass, filter_op_by_execution_time_pass
# from hardware_merge_tiling_4d_macro import hardware_merge_tiling_pass, filter_op_by_execution_time_pass
import islpy as isl
from buffer_mapping import (
    insert_single_buffer_single_level_pass,
    insert_single_buffer_multi_level_pass,
    multi_level_buffer_insersion_pass,
    filter_op_by_memory_access_cost_pass
)
from codegen import codegen_pass
from loop_padding import loop_padding_pass
from tensorize import tensorize_pass
from backend import backend_compile_and_profile_pass
from multi_level_tiling import pre_tiling_pass, memory_tiling_pass
import benchmark
import json
from config import get_config
from draw import extract_frame_info

def run_pipeline(op, skew, cim_cfg, save_dir):
    new_ops = [op]

    if skew:
        new_ops = pre_tiling_pass(new_ops)
        new_ops = auto_skewing_pass(new_ops, return_detail=False)
    # new_ops = new_ops[:min(len(new_ops), 8)]
    new_ops = hardware_merge_tiling_pass(new_ops, macro_row=cim_cfg.n_comp, macro_col=cim_cfg.n_group_vcol)
    # new_ops = new_ops[:min(len(new_ops), 8)]
    new_ops, execution_times = filter_op_by_execution_time_pass(new_ops)
    min_compute_op = new_ops[0]
    return new_ops[0], execution_times[0]

    # N_COMP, N_GROUP, N_GROUP_VCOL
    new_ops = loop_padding_pass(new_ops, padding_inner_size=None)
    new_ops = memory_tiling_pass(new_ops)
    
    new_ops = multi_level_buffer_insersion_pass(new_ops, macro_compute_level=-6)
    new_ops = filter_op_by_memory_access_cost_pass(new_ops)
    new_ops = new_ops[:1] if len(new_ops) >= 1 else []
    new_ops = tensorize_pass(new_ops)
    new_ops = codegen_pass(new_ops)
    result_list = backend_compile_and_profile_pass(new_ops, save_dir)
    
    return result_list

if __name__=="__main__":
    # operator = BasicOperator(
    #     domain = isl.BasicSet(
    #         f"{{ [v0,oh,ow,kh,kw]: v0=0 and 0<=oh<4 and 0<=ow<4 and 0<=kh<3 and 0<=kw<3 }}"
    #     ),
    #     access_I = isl.BasicMap("{ [v0,oh,ow,kh,kw] -> I[oh + kh, ow + kw] }"),
    #     access_O = isl.BasicMap("{ [v0,oh,ow,kh,kw] -> O[v0,oh, ow] }"),
    #     access_W = isl.BasicMap("{ [v0,oh,ow,kh,kw] -> W[v0,kh, kw] }"),
    # )
    skew = True
    virtual_axis = not skew
    # operator = benchmark.get_op_dwconv3d(
    #     ic=1, ox=4, oy=4, oz=4, kx=3, ky=3, kz=3, stride=1
    # )
    operator = benchmark.get_op_conv2d(b=1, oc=1, ic=1, oh=8, ow=8, kh=3, kw=3, stride=1, virtual_axis=virtual_axis)
    # operator = benchmark.get_op_conv2d(b=1, oc=2, ic=1, oh=8, ow=8, kh=3, kw=3, stride=1, virtual_axis=virtual_axis)
    cim_cfg = get_config()
    # print(operator.domain)
    # exit()
    min_compute_op, _ = run_pipeline(operator, skew=skew, cim_cfg=cim_cfg, save_dir=".temp_save")


    for idx, value in enumerate(extract_frame_info(min_compute_op, cim_cfg)):
        timestamp, frame_info = value
        print(f"Index: {idx}.    Timestamp: {timestamp}")
        frame_info.print()
        c = input("continue?(y/n):")
        if c=="n":
            break
        else:
            continue