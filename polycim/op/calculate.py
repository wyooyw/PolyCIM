import numpy as np

def depth_wise_conv2d(input, weight, dilation=1, stride=1):
    assert len(input.shape) == 3
    assert len(weight.shape) == 3
    ic, ih, iw = input.shape
    kc, kh, kw = weight.shape
    assert ic == kc
    
    effective_kh = kh + (kh - 1) * (dilation - 1)
    effective_kw = kw + (kw - 1) * (dilation - 1)
    oh = (ih - effective_kh) // stride + 1
    ow = (iw - effective_kw) // stride + 1
    oc = ic
    output = np.zeros((oc, oh, ow), dtype=input.dtype)
    # for _oc in range(oc):
    for _oh in range(oh):
        for _ow in range(ow):
            input_window = input[:, _oh: _oh + kh * dilation: dilation, _ow : _ow + kw * dilation : dilation]
            weight_window = weight[:, :, :]
            output[:, _oh, _ow] += np.sum(input_window * weight_window, axis=(1,2))
    return output

if __name__ == "__main__":
    input = np.arange(64).reshape(1, 8, 8).astype(np.int32)
    weight = np.arange(9).reshape(1, 3, 3).astype(np.int32)
    output = depth_wise_conv2d(input, weight, dilation=2)
    print(output)
