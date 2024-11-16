from base_operator import BasicOperator
from pass_.affine_transform import auto_skewing_pass
# from hardware_merge_tiling import hardware_merge_tiling_pass, filter_op_by_execution_time_pass
from pass_.hardware_merge_tiling_4d_macro import hardware_merge_tiling_pass, filter_op_by_execution_time_pass
import islpy as isl
from pass_.buffer_mapping import (
    insert_single_buffer_single_level_pass,
    insert_single_buffer_multi_level_pass,
    multi_level_buffer_insersion_pass,
    filter_op_by_memory_access_cost_pass
)
from pass_.codegen import codegen_pass
from pass_.loop_padding import loop_padding_pass
from pass_.tensorize import tensorize_pass
from pass_.backend import backend_compile_and_profile_pass
from pass_.multi_level_tiling import pre_tiling_pass, memory_tiling_pass
import op_define
import json
from config import get_config
import tempfile

def run_pipeline(op, skew, cim_cfg, save_dir):
    new_ops = [op]
    
    # i2#2, i4
    # i1#2, i3
    # tiling = isl.BasicMap("{ [i0, i1, i2, i3, i4] -> [o0, o1, o2, o3, o4, o5, o6, o7] : o0 = i3 and o1 = i4 and (i0 + o5) mod 2 = 0 and (i1 + o6) mod 2 = 0 and (i2 + o7) mod 2 = 0 and 0 <= i0 <= 7 and 0 <= i1 <= 7 and 0 <= i2 <= 7 and 0 <= i3 <= 2 and 0 <= i4 <= 2 and -1 + i0 <= 2o2 <= i0 and -1 + i1 <= 2o3 <= i1 and -1 + i2 <= 2o4 <= i2 and 0 <= o5 <= 1 and 0 <= o6 <= 1 and 0 <= o7 <= 1 }")
    # skewing = isl.BasicMap("{ [i0, i1, i2, i3, i4, i5, i6, i7] -> [o0, o1, o2, o3, o4, o5, o6, o7] : o0 = i6 + i0 and o1 = i7 + i1 and o2=i2 and o3=i3 and o4=i4 and o5=i5 and o6=i6 and o7=i7 }")
    # schedule = tiling #.apply_range(skewing)
    # op = op.apply_schedule(schedule)
    # new_ops = [op]
    if skew:
        new_ops = pre_tiling_pass(new_ops)
        # for idx,op in enumerate(new_ops):
        #     assert op.domain.is_box()
        new_ops = auto_skewing_pass(new_ops, max_reuse_factor_for_arrays=(cim_cfg.n_group_vcol, cim_cfg.n_comp), return_detail=False)
        print(f"after auto_skewing_pass, {len(new_ops)=}")
        # exit()

    # new_ops = hardware_merge_tiling_pass(new_ops, macro_row, macro_col)
    new_ops = hardware_merge_tiling_pass(new_ops)
    print(f"after hardware_merge_tiling_pass, {len(new_ops)=}")
    # exit()
    new_ops = filter_op_by_execution_time_pass(new_ops)
    # for op in new_ops[:1]:
    #     print("")
    #     print(f"tiling : \n    domain: {op.history_domains[0]=}\n    schedule:{op.history_schedules[0]=}")
    #     print(f"skewing : \n    domain: {op.history_domains[1]=}\n    schedule:{op.history_schedules[1]=}")
        
    # print(len(new_ops))
    # new_ops = new_ops[0:1]
    # 
    # new_ops = new_ops[0:2]
    
    # operator = BasicOperator(
    #     domain = isl.BasicSet(
    #         "{ [i0, i1, i2, i3, i4, i5] : i1 >= -2 + i0 and 0 <= i1 <= 63 and i1 <= i0 and 0 <= i4 <= 3 and 0 <= i5 <= 3 and 4*floor((i5)/4) >= -63 + 4i3 + i5 and 4*floor((i5)/4) <= 4i3 + i5 and -2 - 66i0 + 4i2 - 4i3 + i4 - i5 + 4*floor((i5)/4) <= 4*floor((i4)/4) <= -66i0 + 4i2 - 4i3 + i4 - i5 + 4*floor((i5)/4) }"
    #     ),
    #     access_I = isl.BasicMap("{ [i0, i1, i2, i3, i4, i5] -> I[o0, o1] : o1 = i0 and (i4 + o0) mod 2 = 0 and (2i0 - i4 + o0) mod 4 = 0 and i1 >= -2 + i0 and 0 <= i1 <= 63 and i1 <= i0 and 0 <= i4 <= 3 and 0 <= i5 <= 3 and -66i0 + 4i2 <= o0 <= 3 - 66i0 + 4i2 and 4*floor((i5)/4) >= -63 + 4i3 + i5 and 4*floor((i5)/4) >= 4i3 + i5 - o0 and 4*floor((i5)/4) <= 2 + 4i3 + i5 - o0 and 4*floor((i5)/4) <= 4i3 + i5 }"),
    #     access_O = isl.BasicMap("{ [i0, i1, i2, i3, i4, i5] -> O[o0, o1] : o1 = i1 and (-i5 + o0) mod 4 = 0 and i1 >= -2 + i0 and 0 <= i1 <= 63 and i1 <= i0 and 0 <= i4 <= 3 and 0 <= i5 <= 3 and o0 >= 4i3 and 0 <= o0 <= 63 and o0 <= 3 + 4i3 and -2 - 66i0 + 4i2 + i4 - o0 <= 4*floor((i4)/4) <= -66i0 + 4i2 + i4 - o0 }"),
    #     access_W = isl.BasicMap("{ [i0, i1, i2, i3, i4, i5] -> W[o0, o1] : o1 = i0 - i1 and (i4 + i5 + o0) mod 2 = 0 and (2i0 - i4 + i5 + o0) mod 4 = 0 and i1 >= -2 + i0 and 0 <= i1 <= 63 and i1 <= i0 and 0 <= i4 <= 3 and 0 <= i5 <= 3 and 0 <= o0 <= 2 and 4*floor((i4)/4) >= -63 - 66i0 + 4i2 + i4 - o0 and -3 - 66i0 + 4i2 - 4i3 + i4 - o0 <= 4*floor((i4)/4) <= -66i0 + 4i2 - 4i3 + i4 - o0 and 4*floor((i4)/4) <= -66i0 + 4i2 + i4 - o0 }"),
    # )
    # new_ops = [operator]
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
    skew = False
    virtual_axis = not skew
    operator = op_define.get_op_conv2d(b=1, oc=64, ic=256, oh=16, ow=16, kh=3, kw=3, stride=2, virtual_axis=virtual_axis)
    # print(operator.access_W)
    # print(operator.access_W.intersect_domain(operator.domain))
    print(operator.domain.count_val())
    cim_cfg = get_config()
    temp_dir = tempfile.mkdtemp()
    run_pipeline(operator, skew=skew, cim_cfg=cim_cfg, save_dir=temp_dir)