from base_operator import BasicOperator
import os
if int(os.environ.get("NEW_ALGO", 0))==1:
    from affine_transform_new import auto_skewing_pass
else:
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
import utils
from collections import OrderedDict
import time

def run_pipeline(op, skew, cim_cfg, save_dir):

    print(f"run_pipeline {skew=}, {op.domain=}")

    new_ops = [op]

    if skew:
        new_ops = pre_tiling_pass(new_ops)
        # tile_schedule = isl.BasicMap(
        #     "{ [i0, i1, i2, i3, i4] -> [o0, o1, o2, o3, o4, o5, o6] : o0 = i0 and o1 = floor(i1/2) and o2 = i1%2 and o3 = floor(i2/4) and o4 = i2%4 and o5 = i3 and o6 = i4 }"
        # )
        # op = op.apply_schedule(tile_schedule)
        # new_ops = [op]
        # new_ops = new_ops[1:2]
        # print(len(new_ops))
        new_ops,_,_,_ = auto_skewing_pass(new_ops, return_detail=True)
        # for idx,op in enumerate(new_ops):
        #     print(idx)
        #     print(op.history_schedules)
        # exit()
    # new_ops = new_ops[:min(len(new_ops), 8)]
    new_ops = hardware_merge_tiling_pass(new_ops, macro_row=cim_cfg.n_comp, macro_col=cim_cfg.n_group_vcol)
    # new_ops = new_ops[:min(len(new_ops), 8)]
    new_ops, execution_times = filter_op_by_execution_time_pass(new_ops, macro_row=cim_cfg.n_comp, macro_col=cim_cfg.n_group_vcol)
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
    op_list = OrderedDict()
    # op_list["C1"] = (
    #     benchmark.get_op_dwconv2d(ic=1, oh=112, ow=112, kh=3, kw=3, stride=1, dilation=1, virtual_axis=False),
    #     # symmetry_info
    #     ((1,3),(2,4))
    # )
    # op_list["C2"] = (
    #     benchmark.get_op_dwconv2d(ic=1, oh=56, ow=56, kh=3, kw=3, stride=1, dilation=1, virtual_axis=False),
    #     # symmetry_info
    #     ((1,3),(2,4))
    # )
    # op_list["C3"] = (
    #     benchmark.get_op_dwconv2d(ic=1, oh=28, ow=28, kh=5, kw=5, stride=1, dilation=1, virtual_axis=False),
    #     # symmetry_info
    #     ((1,3),(2,4))
    # )
    # op_list["C4"] = (
    #     benchmark.get_op_dwconv2d(ic=1, oh=14, ow=14, kh=3, kw=3, stride=1, dilation=1, virtual_axis=False),
    #     # symmetry_info
    #     ((1,3),(2,4))
    # )
    # op_list["C5"]  = (
    #     benchmark.get_op_dwconv2d(ic=1, oh=14, ow=14, kh=5, kw=5, stride=1, dilation=1, virtual_axis=False),
    #     # symmetry_info
    #     ((1,3),(2,4))
    # )
    # op_list["C6"] = (
    #     benchmark.get_op_dwconv2d(ic=1, oh=7, ow=7, kh=5, kw=5, stride=1, dilation=1, virtual_axis=False),
    #     # symmetry_info
    #     ((1,3),(2,4))
    # )
    # op_list["C7"] = (
    #     benchmark.get_op_dwconv2d(ic=1, oh=7, ow=7, kh=3, kw=3, stride=1, dilation=1, virtual_axis=False),
    #     # symmetry_info
    #     ((1,3),(2,4))
    # )
    # op_list["C8"] = (
    #     benchmark.get_op_dwconv2d(ic=1, oh=56, ow=56, kh=7, kw=7, stride=1, dilation=1, virtual_axis=False),
    #     # symmetry_info
    #     ((1,3),(2,4))
    # )
    # op_list["C9"] = (
    #     benchmark.get_op_dwconv2d(ic=1, oh=7, ow=7, kh=7, kw=7, stride=1, dilation=1, virtual_axis=False),
    #     # symmetry_info
    #     ((1,3),(2,4))
    # )
    # op_list["C10"] = (
    #     benchmark.get_op_dwconv2d(ic=1, oh=56, ow=56, kh=51, kw=51, stride=1, dilation=1, virtual_axis=False),
    #     # symmetry_info
    #     ((1,3),(2,4))
    # )
    # op_list["C11"] = (
    #     benchmark.get_op_dwconv2d(ic=1, oh=7, ow=7, kh=13, kw=13, stride=1, dilation=1, virtual_axis=False),
    #     # symmetry_info
    #     ((1,3),(2,4))
    # )
    op_list["C12"] = (
        benchmark.get_op_dwconv2d(ic=1, oh=28, ow=28, kh=3, kw=3, stride=1, dilation=2, virtual_axis=False),
        # symmetry_info
        ((1,3),(2,4))
    )
    # op_list["C13"] = (
    #     benchmark.get_op_dwconv2d(ic=1, oh=28, ow=28, kh=5, kw=5, stride=1, dilation=2, virtual_axis=False),
    #     # symmetry_info
    #     ((1,3),(2,4))
    # )
    # # op_list["C14"] = benchmark.get_op_dwconv2d(b=1, oc=1, ic=1, oh=28, ow=28, kh=5, kw=5, stride=1, dilation=2)
    # op_list["C15"] = (
    #     benchmark.get_op_dwconv3d(ic=1, ox=28, oy=28, oz=28, kx=5, ky=5, kz=5, stride=1),
    #     # symmetry_info
    #     ((1,4),(2,5),(3,6))
    # )
    cim_cfg = get_config()
    for name, (op, symmetry_info) in op_list.items():
        operator = op
        skew = True
        virtual_axis = not skew
        print(f"{name=}")
        begin_time = time.time()
        min_compute_op, _ = run_pipeline(operator, skew=skew, cim_cfg=cim_cfg, save_dir=".temp_save")
        end_time = time.time()
        print(f"Time cost: {end_time - begin_time}")
        print("\n")
    exit()
    operator = benchmark.get_op_dwconv2d(ic=1, oh=28, ow=28, kh=3, kw=3, stride=1, dilation=2)
    skew = True
    virtual_axis = not skew
    # operator = benchmark.get_op_dwconv3d(
    #     ic=1, ox=4, oy=4, oz=4, kx=3, ky=3, kz=3, stride=1
    # )
    # operator = benchmark.get_op_dwconv3d(
    #     ic=1, ox=4, oy=4, oz=4, kx=3, ky=3, kz=3, stride=1
    # )
    # operator = benchmark.get_op_conv2d(b=1, oc=1, ic=1, oh=16, ow=16, kh=3, kw=3, stride=2, virtual_axis=virtual_axis)
    # operator = benchmark.get_op_conv2d(b=1, oc=2, ic=1, oh=8, ow=8, kh=3, kw=3, stride=1, virtual_axis=virtual_axis)
    cim_cfg = get_config()
    # print(operator.domain)
    # exit()
    min_compute_op, _ = run_pipeline(operator, skew=skew, cim_cfg=cim_cfg, save_dir=".temp_save")
    print(min_compute_op)

    for schedule in min_compute_op.history_schedules:
        print(schedule)
    exit()

    union_domain = min_compute_op.domain
    union_schedule = union_domain.identity()
    code = utils.gen_code(union_domain,union_schedule,None)
    # print(code)
    # import pdb; pdb.set_trace()
    for idx, value in enumerate(extract_frame_info(min_compute_op, cim_cfg, different_weight=True)):
        timestamp, frame_info = value
        print(f"Index: {idx}.    Timestamp: {timestamp}")
        frame_info.print()
        c = input("continue?(y/n):")
        if c=="n":
            break
        else:
            continue