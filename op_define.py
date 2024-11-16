from base_operator import BasicOperator
import islpy as isl

def get_op_dwconv2d(
    ic, oh, ow, kh, kw, virtual_axis=True
    ):
    if virtual_axis:
        operator = BasicOperator(
            domain = isl.BasicSet(
                f"{{ [v0,ic,oh,ow,kh,kw]: 0 <= v0 < 2 and 0<=ic<{ic} and 0<=oh<{oh} and 0<=ow<{ow} and 0<=kh<{kh} and 0<=kw<{kw} }}"
            ),
            access_I = isl.BasicMap("{ [v0, ic,oh,ow,kh,kw] -> I[ic, oh + kh, ow + kw] }"),
            access_O = isl.BasicMap("{ [v0, ic,oh,ow,kh,kw] -> O[v0, ic, oh, ow] }"),
            access_W = isl.BasicMap("{ [v0, ic,oh,ow,kh,kw] -> W[v0, ic, kh, kw] }"),
        )
    else:
        operator = BasicOperator(
            domain = isl.BasicSet(
                f"{{ [ic,oh,ow,kh,kw]: 0<=ic<{ic} and 0<=oh<{oh} and 0<=ow<{ow} and 0<=kh<{kh} and 0<=kw<{kw} }}"
            ),
            access_I = isl.BasicMap("{ [ic,oh,ow,kh,kw] -> I[ic, oh + kh, ow + kw] }"),
            access_O = isl.BasicMap("{ [ic,oh,ow,kh,kw] -> O[ic, oh, ow] }"),
            access_W = isl.BasicMap("{ [ic,oh,ow,kh,kw] -> W[ic, kh, kw] }"),
        )
    return operator

def get_op_conv2d(
    b, oc, ic, oh, ow, kh, kw, stride=1, virtual_axis=True
    ):
    operator = BasicOperator(
        domain = isl.BasicSet(
            f"{{ [b,oc,ic,oh,ow,kh,kw]: 0<=b<{b} and 0<=oc<{oc} and 0<=ic<{ic} and 0<=oh<{oh} and 0<=ow<{ow} and 0<=kh<{kh} and 0<=kw<{kw} }}"
        ),
        access_I = isl.BasicMap(f"{{ [b,oc,ic,oh,ow,kh,kw] -> I[b, ic, oh * {stride} + kh, ow * {stride} + kw] }}"),
        access_O = isl.BasicMap("{ [b,oc,ic,oh,ow,kh,kw] -> O[b, oc, oh, ow] }"),
        access_W = isl.BasicMap("{ [b,oc,ic,oh,ow,kh,kw] -> W[oc, ic, kh, kw] }"),
    )
    return operator