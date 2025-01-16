import islpy as isl
import numpy as np
from codegen_.codegen_data_layout_convert import data_layout_convert
import benchmark
import utils
import pytest

def test_codegen_data_layout_convert_transpose():
    accrel_lhs = isl.Map("{ [i, j] -> A[i, j] : 0 <= i < 4 and 0 <= j < 4 }")
    accrel_rhs = isl.Map("{ [i, j] -> B[j, i] : 0 <= i < 4 and 0 <= j < 4 }")
    input_data = np.arange(16).astype(np.int32).reshape(4, 4)
    golden_data = input_data.T
    output_data = data_layout_convert(accrel_lhs, accrel_rhs, input_data)
    assert np.allclose(output_data, golden_data), f"{output_data=}, {golden_data=}"

def test_codegen_data_layout_convert_copy():
    accrel_output = isl.Map("{ [i, j] -> A[i, j] : 0 <= i < 4 and 0 <= j < 4 }")
    accrel_input = isl.Map("{ [i, j] -> B[i] : 0 <= i < 4 and 0 <= j < 4 }")
    input_data = np.arange(4).astype(np.int32) #.reshape(4, 4)
    golden_data = np.repeat(input_data.reshape(1,-1), 4, axis=1).reshape(4, 4)
    output_data = data_layout_convert(accrel_input, accrel_output, input_data)
    assert np.allclose(output_data, golden_data), f"{output_data=}, {golden_data=}"

def test_codegen_data_layout_convert_flatten():
    accrel_output = isl.Map("{ [i] -> A[i] : 0 <= i < 16 }")
    accrel_input = isl.Map("{ [i] -> B[p,q] : p=i//4 and q=i%4 and 0 <= i < 16 }")
    input_data = np.arange(16).astype(np.int32).reshape(4, 4)
    golden_data = input_data.reshape(-1)
    output_data = data_layout_convert(accrel_input, accrel_output, input_data)
    assert np.allclose(output_data, golden_data), f"{output_data=}, {golden_data=}"

@pytest.mark.parametrize(
    "batch, out_channel, in_channel, out_size, ker_size", 
    [
        (2, 3, 4, 5, 6),
        (6, 5, 4, 3, 2),
        (8, 8, 8, 32, 3),
    ]
)
def test_codegen_data_layout_convert_im2col(
    batch, out_channel, in_channel, out_size, ker_size
):
    op = benchmark.get_op_conv2d(b=batch, oc=out_channel, ic=in_channel, oh=out_size, ow=out_size, kh=ker_size, kw=ker_size, stride=1, virtual_axis=False)

    # do the im2col
    im2col_spatial_size = out_size * out_size
    im2col_kernel_size = in_channel * ker_size * ker_size
    coalescing_schedule = isl.BasicMap(f"{{ [b,oc,ic,oh,ow,kh,kw] -> [b,oc,s,r] : s = oh * {out_size} + ow and r = ic * {ker_size * ker_size} + kh * {ker_size} + kw }}")
    reverse_schedule = isl.BasicMap(
        f"{{ [b, oc, s, r] -> [b, oc, ic, oh, ow, kh, kw] : oh = floor(s/{out_size}) and ow = s%{out_size} and ic = floor(r/{ker_size *ker_size}) and kh = floor((r%{ker_size *ker_size})/{ker_size}) and kw = r%{ker_size} }}"
    )
    op = op.apply_schedule(coalescing_schedule, reverse_schedule=reverse_schedule, skip_simplify=True)

    in_size = out_size + (ker_size - 1)
    input_data = np.arange(batch * in_channel * in_size * in_size).astype(np.int32).reshape(batch, in_channel, in_size, in_size)
    im2col_input_golden_data = np.zeros((batch, im2col_spatial_size, im2col_kernel_size), dtype=np.int32)
    # get golden im2col input
    for b in range(batch):
        for oh in range(out_size):
            for ow in range(out_size):
                input_kernel = input_data[b, :, oh:oh+ker_size, ow:ow+ker_size]
                im2col_input_golden_data[b, oh * out_size + ow, :] = input_kernel.reshape(-1)
    
    accrel_input = op.access_I
    accrel_output = isl.Map(f"{{ [b,oc,s,r] -> Inew[b,s,r] }}")
    accrel_output = accrel_output.intersect_domain(op.domain)
    output_data = data_layout_convert(accrel_input, accrel_output, input_data)
    assert np.allclose(output_data, im2col_input_golden_data), f"{output_data=}, {im2col_input_golden_data=}"

if __name__ == "__main__":
    test_codegen_data_layout_convert_im2col(1, 1, 1, 8, 3)