from base_operator import BasicOperator
from affine_transform import auto_skewing_pass
from hardware_merge_tiling import hardware_merge_tiling_pass, filter_op_by_execution_time_pass
import islpy as isl
from buffer_mapping import (
    insert_single_buffer_single_level_pass,
    insert_single_buffer_multi_level_pass
)
from codegen import codegen_pass
from loop_padding import loop_padding_pass
from tensorize import tensorize_pass
from backend import backend_compile_and_profile_pass
def run_pipeline(op):
    new_ops = [op]
    # new_ops = auto_skewing_pass([op], max_reuse_factor_for_arrays=(16,16), return_detail=False)
    new_ops = hardware_merge_tiling_pass(new_ops)
    new_ops = filter_op_by_execution_time_pass(new_ops)
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

    new_ops = loop_padding_pass(new_ops, padding_inner_size=[
        [0, 3], # begin, size
        [0, 3]
    ])
    new_ops = insert_single_buffer_multi_level_pass(new_ops, buffer_name="W", buffer_levels=[-3])
    new_ops = tensorize_pass(new_ops)
    new_ops = new_ops[0:1]
    new_ops = codegen_pass(new_ops)
    print(new_ops[0].dsl)
    # backend_compile_and_profile_pass(new_ops, ".temp_save")
    # print("\nAfter insert_single_buffer_single_pass: \n")
    # for idx, code in enumerate(new_codes):
    #     print(idx)
    #     print(code)
    #     print("----------------------------------------")

if __name__=="__main__":
    operator = BasicOperator(
        domain = isl.BasicSet(
            "{ [oh,ow,kh,kw]: 0<=oh<64 and 0<=ow<64 and 0<=kh<3 and 0<=kw<3 }"
        ),
        access_I = isl.BasicMap("{ [oh,ow,kh,kw] -> I[oh + kh, ow + kw] }"),
        access_O = isl.BasicMap("{ [oh,ow,kh,kw] -> O[oh, ow] }"),
        access_W = isl.BasicMap("{ [oh,ow,kh,kw] -> W[kh, kw] }"),
    )
    operator = BasicOperator(
        domain = isl.BasicSet(
            "{ [oc, oh,ow,kh,kw]: 0<=oc<4 and 0<=oh<8 and 0<=ow<8 and 0<=kh<3 and 0<=kw<3 }"
        ),
        access_I = isl.BasicMap("{ [oc, oh,ow,kh,kw] -> I[oh + kh, ow + kw] }"),
        access_O = isl.BasicMap("{ [oc, oh,ow,kh,kw] -> O[oc, oh, ow] }"),
        access_W = isl.BasicMap("{ [oc, oh,ow,kh,kw] -> W[oc, kh, kw] }"),
    )
    run_pipeline(operator)