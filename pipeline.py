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
    
    # i2#2, i4
    # i1#2, i3
    # tiling = isl.BasicMap("{ [i0, i1, i2, i3, i4] -> [o0, o1, o2, o3, o4, o5, o6] : o0 = i0 and o1 = floor(i1/2) and o2=floor(i2/2) and o3=i3 and o4=i4 and o5=i1%2 and o6=i2%2 }")
    # tiling = isl.BasicMap("{ [j0, j1, i0, i1, i2, i3, i4] -> [j0, j1, o0, o1, o2, o3, o4, o5, o6] : o0 = i0 and o1 = floor(i1/2) and o2=floor(i2/2) and o3=i3 and o4=i4 and o5=i1%2 and o6=i2%2 }")
    # tiling = isl.BasicMap("{ [j0, j1, i0, i1, i2, i3, i4] -> [j0, j1, o0, o1, o2, o3, o4, o5, o6] : o0 = i0 and o1 = floor(i1/2) and o2=floor(i2/2) and o3=i1%2 and o4=i2%2 and o5=i3 and o6=i4 }")
    # tiling = isl.BasicMap("{ [i0,i1,i2,i3,i4,i5,i6] -> [floor(i0/1),floor(i1/1),floor(i2/1),floor(i3/2),floor(i4/2),floor(i5/1),floor(i6/1),i0%1,i1%1,i2%1,i3%2,i4%2,i5%1,i6%1] }")
    # tiling = isl.BasicMap("{ [i0,i1,i2,i3,i4,i5,i6] -> [i0,i1,i2,floor(i3/2),floor(i4/2),i5,i6,i3%2,i4%2] }")
    # op = op.apply_schedule(tiling)
    # new_ops = [op]

    if skew:
        new_ops = pre_tiling_pass(new_ops)
        new_ops,_,_,_ = auto_skewing_pass(new_ops, max_reuse_factor_for_arrays=(cim_cfg.n_group_vcol, cim_cfg.n_comp), return_detail=True)
        
        # exit()
    # print(len(new_ops))
    # for idx,op in enumerate(new_ops):
    #     print(f"{idx=}")
    #     matrix = base_matrix_list[idx]
    #     for row in range(matrix.rows):
    #         for col in range(matrix.cols):
    #             print(f"{matrix[row,col]}",end=" ")
    #         print("")
    #     print("affine schedule:", op.history_schedules[1])
    #     print("")
    # exit()
    # return
    # new_ops = new_ops[3:4]
    new_ops = hardware_merge_tiling_pass(new_ops, macro_row=cim_cfg.n_comp, macro_col=cim_cfg.n_group_vcol)
    # exit()
    # new_ops = hardware_merge_tiling_pass(new_ops)
    print(f"after hardware_merge_tiling_pass, {len(new_ops)=}")
    # exit()
    new_ops = filter_op_by_execution_time_pass(new_ops)

    min_compute_op = new_ops[0]
    for idx,op in enumerate(new_ops):
        print(f"{idx=}")
        n_dim = op.domain.dim(isl.dim_type.set)
        outer_domain = op.domain.project_out(isl.dim_type.set, n_dim - 2, 2)
        print(f"{outer_domain.count_val()=}")
        print("pretiling: ", op.history_schedules[0])
        print("pretiling-factor: ", op.history_schedules[1])
    exit()
    # print("affine: ", min_compute_op.history_schedules[1])
    # print("shift: ", min_compute_op.history_schedules[2])
    for idx, value in enumerate(extract_frame_info(min_compute_op, cim_cfg)):
        timestamp, frame_info = value
        print(f"Index: {idx}.    Timestamp: {timestamp}")
        frame_info.print()
        c = input("continue?(y/n):")
        if c=="n":
            break
        else:
            continue
    exit()
    # N_COMP, N_GROUP, N_GROUP_VCOL
    new_ops = loop_padding_pass(new_ops, padding_inner_size=None)
    new_ops = memory_tiling_pass(new_ops)
    
    new_ops = multi_level_buffer_insersion_pass(new_ops, macro_compute_level=-6)
    # exit()
    # import pdb; pdb.set_trace()
    new_ops = filter_op_by_memory_access_cost_pass(new_ops)
    new_ops = new_ops[:1] if len(new_ops) >= 1 else []
    new_ops = tensorize_pass(new_ops)
    new_ops = codegen_pass(new_ops)
    # print(len(new_ops))
    # print(new_ops[0].dsl)
    result_list = backend_compile_and_profile_pass(new_ops, save_dir)
    
    # for result in result_list:
    #     print(f"{result.save_path=}")
    #     print(json.dumps(result.stats,indent=2))
    #     print("")
    # exit()
    # exit()
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
    operator = benchmark.get_op_dwconv2d(ic=1, oh=64, ow=64, kh=7, kw=7, stride=1, virtual_axis=virtual_axis)
    # operator = benchmark.get_op_conv2d(b=1, oc=1, ic=1, oh=8, ow=8, kh=3, kw=3, stride=1, virtual_axis=virtual_axis)
    cim_cfg = get_config()
    # print(operator.domain)
    # exit()
    run_pipeline(operator, skew=skew, cim_cfg=cim_cfg, save_dir=".temp_save")