import numpy as np

def depth_wise_conv2d(input, weight):
    assert len(input.shape) == 3
    assert len(weight.shape) == 3
    ic, ih, iw = input.shape
    kc, kh, kw = weight.shape
    assert ic == kc
    oh = ih - kh + 1
    ow = iw - kw + 1
    oc = ic
    output = np.zeros((oc, oh, ow), dtype=input.dtype)
    # for _oc in range(oc):
    for _oh in range(oh):
        for _ow in range(ow):
            input_window = input[:, _oh: _oh + kh, _ow: _ow + kw]
            weight_window = weight[:, :, :]
            output[:, _oh, _ow] += np.sum(input_window * weight_window, axis=(1,2))
    return output