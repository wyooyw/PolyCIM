import pytest
from multi_level_tiling import multi_level_tiling, multi_level_splitting_var_level
from utils import (
    get_box_hull_shape
)
import islpy as isl
from base_operator import BasicOperator

def get_op_simple_conv2d(oh,ow,kh,kw):
    op = BasicOperator(
        domain = isl.BasicSet(
            f"{{ [oh,ow,kh,kw]: 0<=oh<{oh} and 0<=ow<{ow} and 0<=kh<{kh} and 0<=kw<{kw} }}"
        ),
        access_I = isl.BasicMap(f"{{ [oh,ow,kh,kw] -> I[oh + kh, ow + kw] }}"),
        access_O = isl.BasicMap("{ [oh,ow,kh,kw] -> O[oh, ow] }"),
        access_W = isl.BasicMap("{ [oh,ow,kh,kw] -> W[kh, kw] }"),
    )
    return op

def test_multi_level_tiling_basic():
    op = get_op_simple_conv2d(8,8,3,3)
    tiling_factors = [
        [2,4],
        [2,4],
        [1,3],
        [1,3],
    ]
    new_op = multi_level_tiling(op, 2, tiling_factors)
    domain_shape = get_box_hull_shape(new_op.domain)
    golden_domain_shape = [2,2,1,1,4,4,3,3]
    assert domain_shape == golden_domain_shape

@pytest.mark.parametrize(
    "tiling_factors, golden_domain_shape",
    [
        ([[2,4], [2,2,2], [3], [3]], [2,4,2,2,2,3,3]),
        ([[2,4], [2,2,2], [1,3], [3,1]], [2,4,2,2,2,1,3,3,1]),
        ([[8], [8], [3], [3]], [8,8,3,3]),
    ]
)
def test_multi_level_tiling_var_level(tiling_factors, golden_domain_shape):
    op = get_op_simple_conv2d(8,8,3,3)
    new_op = multi_level_splitting_var_level(op, tiling_factors)
    domain_shape = get_box_hull_shape(new_op.domain)
    assert domain_shape == golden_domain_shape
    
if __name__ == "__main__":
    test_multi_level_tiling_var_level()