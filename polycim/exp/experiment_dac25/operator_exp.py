from pipeline import run_pipeline
from baselines import im2col,SDK,vw_sdk
import polycim.op.benchmark as benchmark
from polycim.config import CIMConfig
import json
import argparse

def polycim(cim_row, cim_col, operator):

    skew = True
    virtual_axis = not skew
    cim_cfg = CIMConfig(
        n_row=1,
        n_group_vcol=cim_col,
        n_comp=cim_row,
        n_group=1
    )
    flops = operator.domain.count_val().get_num_si()
    op,cim_flops = run_pipeline(operator, skew=skew, cim_cfg=cim_cfg, save_dir=".temp_save")
    
    mean_cell_use = flops / cim_flops


    return cim_flops, mean_cell_use

def compare_conv2d(cim_row, cim_col, oh, ow, oc, ic, kh, kw, dilation=1):
    
    _kh = kh + (kh - 1) * (dilation - 1)
    _kw = kw + (kw - 1) * (dilation - 1)
    ih = oh + _kh - 1
    iw = ow + _kw - 1

    try:
        
        im2col_cycle = im2col(
            image_col=iw, 
            image_row=ih, 
            filter_col=_kw, 
            filter_row=_kh, 
            in_channel=ic, 
            out_channel=oc,
            array_row=cim_row, 
            array_col=cim_col
        )
    except Exception as e:
        print("im2col error: ", e)
        im2col_cycle = "Error"
    
    try:
        sdk_cycle = SDK(
            image_col=iw, 
            image_row=ih, 
            filter_col=_kw, 
            filter_row=_kh, 
            in_channel=ic, 
            out_channel=oc,
            array_row=cim_row, 
            array_col=cim_col
        )[0]
        if type(sdk_cycle) == list:
            sdk_cycle = sdk_cycle[0]
    except Exception as e:
        print("SDK error: ", e)
        sdk_cycle = "Error"
    
    try:
        vwsdk_cycle = vw_sdk(
            image_col=iw, 
            image_row=ih, 
            filter_col=_kw, 
            filter_row=_kh, 
            in_channel=ic, 
            out_channel=oc,
            array_row=cim_row, 
            array_col=cim_col
        )[0]
        if type(vwsdk_cycle) == list:
            vwsdk_cycle = vwsdk_cycle[0]    
    except Exception as e:
        print("vw_sdk error: ", e)
        vwsdk_cycle = "Error"

    operator = benchmark.get_op_conv2d(b=1, oc=oc, ic=ic, oh=oh, ow=ow, kh=kh, kw=kw, stride=1, virtual_axis=True)
    
    polycim_cycle = polycim(cim_row, cim_col, operator)[0]
    polycim_cycle = int(polycim_cycle)
    

    # whether polycim is faster than im2col ,sdk and vwsdk
    not_worse_than_im2col = im2col_cycle == "Error" or polycim_cycle <= im2col_cycle
    not_worse_than_sdk = sdk_cycle == "Error" or polycim_cycle <= sdk_cycle
    not_worse_than_vwsdk = vwsdk_cycle == "Error" or polycim_cycle <= vwsdk_cycle
    polycim_is_best = not_worse_than_im2col and not_worse_than_sdk and not_worse_than_vwsdk 

    # save info contains:
    # cim_row, cim_col, oh, ow, oc, ic, kh, kw
    # im2col_cycle, sdk_cycle, vwsdk_cycle, polycim_cycle
    save_json = {
        "setting": {
            "cim_row": cim_row,
            "cim_col": cim_col,
            "oh": oh,
            "ow": ow,
            "oc": oc,
            "ic": ic,
            "kh": kh,
            "kw": kw,
            "dilation": dilation
        },
        "cycle": {
            "im2col": im2col_cycle,
            "sdk": sdk_cycle,
            "vwsdk": vwsdk_cycle,
            "polycim": int(polycim_cycle),
            "polycim_is_best": polycim_is_best, 
        }
    }
    # print(json.dumps(save_json))
    # with open(f".temp_save/compare_conv2d_{cim_row}_{cim_col}_{oh}_{ow}_{oc}_{ic}_{kh}_{kw}.json", "w") as f:
    #     json.dump(save_json, f, indent=2) # save to json file
    return save_json

def compare_dwconv2d(cim_row, cim_col, oh, ow, ic, kh, kw, dilation=1):
    try:
        kh = kh + (kh - 1) * (dilation - 1)
        kw = kw + (kw - 1) * (dilation - 1)
        ih = oh + kh - 1
        iw = ow + kw - 1
        im2col_cycle = im2col(
            image_col=iw, 
            image_row=ih, 
            filter_col=kw, 
            filter_row=kh, 
            in_channel=1, 
            out_channel=1,
            array_row=cim_row, 
            array_col=cim_col
        )
        im2col_cycle = im2col_cycle * ic
    except Exception as e:
        print("im2col error: ", e)
        im2col_cycle = "Error"
    
    try:
        kh = kh + (kh - 1) * (dilation - 1)
        kw = kw + (kw - 1) * (dilation - 1)
        ih = oh + kh - 1
        iw = ow + kw - 1
        sdk_cycle = SDK(
            image_col=iw, 
            image_row=ih, 
            filter_col=kw, 
            filter_row=kh, 
            in_channel=1, 
            out_channel=1,
            array_row=cim_row, 
            array_col=cim_col
        )[0]
        sdk_cycle = sdk_cycle * ic
    except Exception as e:
        print("SDK error: ", e)
        sdk_cycle = "Error"
    
    try:
        kh = kh + (kh - 1) * (dilation - 1)
        kw = kw + (kw - 1) * (dilation - 1)
        ih = oh + kh - 1
        iw = ow + kw - 1
        vwsdk_cycle = vw_sdk(
            image_col=iw, 
            image_row=ih, 
            filter_col=kw, 
            filter_row=kh, 
            in_channel=1, 
            out_channel=1,
            array_row=cim_row, 
            array_col=cim_col
        )[0]
        vwsdk_cycle = vwsdk_cycle * ic
    except Exception as e:
        print("vw_sdk error: ", e)
        vwsdk_cycle = "Error"

    operator = benchmark.get_op_conv2d(b=1, oc=1, ic=1, oh=oh, ow=ow, kh=kh, kw=kw, stride=1, dilation=dilation, virtual_axis=True)
    polycim_cycle = polycim(cim_row, cim_col, operator)[0]
    polycim_cycle = polycim_cycle * ic

    # save info contains:
    # cim_row, cim_col, oh, ow, oc, ic, kh, kw
    # im2col_cycle, sdk_cycle, vwsdk_cycle, polycim_cycle
    save_json = {
        "setting": {
            "cim_row": cim_row,
            "cim_col": cim_col,
            "oh": oh,
            "ow": ow,
            "oc": oc,
            "ic": ic,
            "kh": kh,
            "kw": kw,
        },
        "cycle": {
            "im2col": im2col_cycle,
            "sdk": sdk_cycle,
            "vwsdk": vwsdk_cycle,
            "polycim": int(polycim_cycle),
        }
    }
    print(save_json)
    # with open(f".temp_save/compare_conv2d_{cim_row}_{cim_col}_{oh}_{ow}_{oc}_{ic}_{kh}_{kw}.json", "w") as f:
    #     json.dump(save_json, f, indent=2) # save to json file


def compare_conv2d_all():
    pass

def compare_conv3d(cim_row, cim_col, oh, ow, oz, kh, kw, kz, dilation=1):
    
    _kh = kh + (kh - 1) * (dilation - 1)
    _kw = kw + (kw - 1) * (dilation - 1)
    _kz = kz + (kz - 1) * (dilation - 1)
    ih = oh + _kh - 1
    iw = ow + _kw - 1
    iz = oz + _kz - 1

    try:
        im2col_cycle = im2col(
            image_col=iw, 
            image_row=ih, 
            filter_col=_kw, 
            filter_row=_kh, 
            in_channel=_kz, 
            out_channel=1,
            array_row=cim_row, 
            array_col=cim_col
        )
        im2col_cycle = im2col_cycle * oz
    except Exception as e:
        print("im2col error: ", e)
        im2col_cycle = "Error"
    
    try:
        sdk_cycle = SDK(
            image_col=iw, 
            image_row=ih, 
            filter_col=_kw, 
            filter_row=_kh, 
            in_channel=_kz, 
            out_channel=1,
            array_row=cim_row, 
            array_col=cim_col
        )[0]
        if type(sdk_cycle) == list:
            sdk_cycle = sdk_cycle[0]
        sdk_cycle = sdk_cycle * oz
    except Exception as e:
        print("SDK error: ", e)
        sdk_cycle = "Error"
    
    try:
        vwsdk_cycle = vw_sdk(
            image_col=iw, 
            image_row=ih, 
            filter_col=_kw, 
            filter_row=_kh, 
            in_channel=_kz, 
            out_channel=1,
            array_row=cim_row, 
            array_col=cim_col
        )[0]
        if type(vwsdk_cycle) == list:
            vwsdk_cycle = vwsdk_cycle[0]    
        vwsdk_cycle = vwsdk_cycle * oz
    except Exception as e:
        print("vw_sdk error: ", e)
        vwsdk_cycle = "Error"

    operator = benchmark.get_op_dwconv3d(ic=1, ox=oh, oy=ow, oz=oz, kx=kh, ky=kw, kz=kz, stride=1, virtual_axis=True)
    # polycim_cycle = polycim(cim_row, cim_col, operator)[0]
    # polycim_cycle = int(polycim_cycle)
    polycim_cycle = 0

    # whether polycim is faster than im2col ,sdk and vwsdk
    not_worse_than_im2col = im2col_cycle == "Error" or polycim_cycle <= im2col_cycle
    not_worse_than_sdk = sdk_cycle == "Error" or polycim_cycle <= sdk_cycle
    not_worse_than_vwsdk = vwsdk_cycle == "Error" or polycim_cycle <= vwsdk_cycle
    polycim_is_best = not_worse_than_im2col and not_worse_than_sdk and not_worse_than_vwsdk 

    # save info contains:
    # cim_row, cim_col, oh, ow, oc, ic, kh, kw
    # im2col_cycle, sdk_cycle, vwsdk_cycle, polycim_cycle
    save_json = {
        "setting": {
            "cim_row": cim_row,
            "cim_col": cim_col,
            "oh": oh,
            "ow": ow,
            "oz": oz,
            "kh": kh,
            "kw": kw,
            "kz": kz,
            "dilation": dilation
        },
        "cycle": {
            "im2col": im2col_cycle,
            "sdk": sdk_cycle,
            "vwsdk": vwsdk_cycle,
            "polycim": int(polycim_cycle),
            "polycim_is_best": polycim_is_best, 
        }
    }
    # print(json.dumps(save_json))
    # with open(f".temp_save/compare_conv2d_{cim_row}_{cim_col}_{oh}_{ow}_{oc}_{ic}_{kh}_{kw}.json", "w") as f:
    #     json.dump(save_json, f, indent=2) # save to json file
    return save_json

def compare_conv1d_45degree(cim_row, cim_col, oh, ow, oc, ic, kh):
    
    ih = oh + kh - 1
    iw = ow + kh - 1

    try:
        
        im2col_cycle = im2col(
            image_col=iw, 
            image_row=ih, 
            filter_col=kh, 
            filter_row=kh, 
            in_channel=ic, 
            out_channel=oc,
            array_row=cim_row, 
            array_col=cim_col
        )
    except Exception as e:
        print("im2col error: ", e)
        im2col_cycle = "Error"
    
    try:
        sdk_cycle = SDK(
            image_col=iw, 
            image_row=ih, 
            filter_col=kh, 
            filter_row=kh, 
            in_channel=ic, 
            out_channel=oc,
            array_row=cim_row, 
            array_col=cim_col
        )[0]
        if type(sdk_cycle) == list:
            sdk_cycle = sdk_cycle[0]
    except Exception as e:
        print("SDK error: ", e)
        sdk_cycle = "Error"
    
    try:
        vwsdk_cycle = vw_sdk(
            image_col=iw, 
            image_row=ih, 
            filter_col=kh, 
            filter_row=kh, 
            in_channel=ic, 
            out_channel=oc,
            array_row=cim_row, 
            array_col=cim_col
        )[0]
        if type(vwsdk_cycle) == list:
            vwsdk_cycle = vwsdk_cycle[0]    
    except Exception as e:
        print("vw_sdk error: ", e)
        vwsdk_cycle = "Error"

    operator = benchmark.get_op_conv1d(oc=oc, ic=ic, oh=oh, ow=ow, k=kh, virtual_axis=True)
    polycim_cycle = polycim(cim_row, cim_col, operator)[0]
    polycim_cycle = int(polycim_cycle)

    # whether polycim is faster than im2col ,sdk and vwsdk
    not_worse_than_im2col = im2col_cycle == "Error" or polycim_cycle <= im2col_cycle
    not_worse_than_sdk = sdk_cycle == "Error" or polycim_cycle <= sdk_cycle
    not_worse_than_vwsdk = vwsdk_cycle == "Error" or polycim_cycle <= vwsdk_cycle
    polycim_is_best = not_worse_than_im2col and not_worse_than_sdk and not_worse_than_vwsdk 

    # save info contains:
    # cim_row, cim_col, oh, ow, oc, ic, kh, kw
    # im2col_cycle, sdk_cycle, vwsdk_cycle, polycim_cycle
    save_json = {
        "setting": {
            "cim_row": cim_row,
            "cim_col": cim_col,
            "oh": oh,
            "ow": ow,
            "oc": oc,
            "ic": ic,
            "kh": kh,
        },
        "cycle": {
            "im2col": im2col_cycle,
            "sdk": sdk_cycle,
            "vwsdk": vwsdk_cycle,
            "polycim": int(polycim_cycle),
            "polycim_is_best": polycim_is_best, 
        }
    }
    # print(json.dumps(save_json))
    # with open(f".temp_save/compare_conv2d_{cim_row}_{cim_col}_{oh}_{ow}_{oc}_{ic}_{kh}_{kw}.json", "w") as f:
    #     json.dump(save_json, f, indent=2) # save to json file
    return save_json

def main_convnext():
    cim_config_list = [
       (32,8) ,(64,8)
    ]
    # oh=oh, ow=ow, oc=oc, ic=ic, kh=kh, kw=kw, dilation
    op_config_list = [
        {"oh":56,"ow":56, "oc":1, "ic":1, "kh":7, "kw":7, "dilation":1},
        {"oh":28,"ow":28, "oc":1, "ic":1, "kh":7, "kw":7, "dilation":1},
        {"oh":14,"ow":14, "oc":1, "ic":1, "kh":7, "kw":7, "dilation":1},
        {"oh":7,"ow":7, "oc":1, "ic":1, "kh":7, "kw":7, "dilation":1},
    ]
    with open("compare_conv2d_for_convnext.json", "w") as f:
        for cim_row, cim_col in cim_config_list:
            for op_args in op_config_list:
                result = compare_conv2d(
                    cim_row=cim_row, cim_col=cim_col,
                    **op_args
                )
                json.dump(result, f)
                f.write("\n")
                f.flush()

def main_RepLKNet():
    cim_config_list = [
        (32,8),(64,8)
    ]
    # oh=oh, ow=ow, oc=oc, ic=ic, kh=kh, kw=kw, dilation
    op_config_list = [
        {"oh":56,"ow":56, "oc":1, "ic":1, "kh":31, "kw":31, "dilation":1},
        {"oh":28,"ow":28, "oc":1, "ic":1, "kh":29, "kw":29, "dilation":1},
        {"oh":14,"ow":14, "oc":1, "ic":1, "kh":27, "kw":27, "dilation":1},
        {"oh":7,"ow":7, "oc":1, "ic":1, "kh":13, "kw":13, "dilation":1},
    ]
    with open("compare_conv2d_for_RepLKNet.json", "w") as f:
        for cim_row, cim_col in cim_config_list:
            for op_args in op_config_list:
                result = compare_conv2d(
                    cim_row=cim_row, cim_col=cim_col,
                    **op_args
                )
                json.dump(result, f)
                f.write("\n")
                f.flush()

def main_SLaK():
    cim_config_list = [
        (32,8),(64,8)
    ]
    # oh=oh, ow=ow, oc=oc, ic=ic, kh=kh, kw=kw, dilation
    op_config_list = [
        {"oh":56,"ow":56, "oc":1, "ic":1, "kh":51, "kw":51, "dilation":1},
        {"oh":56,"ow":56, "oc":1, "ic":1, "kh":5, "kw":5, "dilation":1},
        {"oh":28,"ow":28, "oc":1, "ic":1, "kh":49, "kw":49, "dilation":1},
        {"oh":28,"ow":28, "oc":1, "ic":1, "kh":5, "kw":5, "dilation":1},
        {"oh":14,"ow":14, "oc":1, "ic":1, "kh":47, "kw":47, "dilation":1},
        {"oh":14,"ow":14, "oc":1, "ic":1, "kh":5, "kw":5, "dilation":1},
        {"oh":7,"ow":7, "oc":1, "ic":1, "kh":13, "kw":13, "dilation":1},
        {"oh":7,"ow":7, "oc":1, "ic":1, "kh":5, "kw":5, "dilation":1},
    ]
    with open("compare_conv2d_for_SLaK.json", "w") as f:
        for cim_row, cim_col in cim_config_list:
            for op_args in op_config_list:
                result = compare_conv2d(
                    cim_row=cim_row, cim_col=cim_col,
                    **op_args
                )
                json.dump(result, f)
                f.write("\n")
                f.flush()

def main_EfficientNet():
    cim_config_list = [
        (32,8),(64,8)
    ]
    # oh=oh, ow=ow, oc=oc, ic=ic, kh=kh, kw=kw, dilation
    op_config_list = [
        {"oh":112,"ow":112, "oc":1, "ic":1, "kh":3, "kw":3, "dilation":1},
        {"oh":56,"ow":56, "oc":1, "ic":1, "kh":3, "kw":3, "dilation":1},
        # {"oh":56,"ow":56, "oc":1, "ic":1, "kh":5, "kw":5,"stride":2, "dilation":1},
        {"oh":28,"ow":28, "oc":1, "ic":1, "kh":5, "kw":5, "dilation":1},
        # {"oh":28,"ow":28, "oc":1, "ic":1, "kh":3, "kw":3, "stride":2, "dilation":1},
        {"oh":14,"ow":14, "oc":1, "ic":1, "kh":3, "kw":3, "dilation":1},
        {"oh":14,"ow":14, "oc":1, "ic":1, "kh":5, "kw":5, "dilation":1},
        {"oh":7,"ow":7, "oc":1, "ic":1, "kh":5, "kw":5, "dilation":1},
        {"oh":7,"ow":7, "oc":1, "ic":1, "kh":3, "kw":3, "dilation":1},
        # {"oh":56,"ow":56, "oc":1, "ic":1, "kh":3, "kw":3, "dilation":1},
        
    ]
    with open("compare_conv2d_for_EfficientNet.json", "w") as f:
        for cim_row, cim_col in cim_config_list:
            for op_args in op_config_list:
                result = compare_conv2d(
                    cim_row=cim_row, cim_col=cim_col,
                    **op_args
                )
                json.dump(result, f)
                f.write("\n")
                f.flush()

if __name__ == "__main__":
    # use argparser to judge which function to execute
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                       choices=["convnext", "RepLKNet", "SLaK", "EfficientNet"],
                       help="Model to evaluate: convnext, RepLKNet, or SLaK")

    args = parser.parse_args()
    # main_RepLKNet()
    if args.model == "convnext":
        main_convnext()
    elif args.model == "RepLKNet":
        main_RepLKNet()
    elif args.model == "SLaK":
        main_SLaK()
    elif args.model == "EfficientNet":
        main_EfficientNet()
    else:
        print("Invalid model name")

    # args = {
    #     "cim_row": 32,
    #     "cim_col": 8,
    #     "oh": 16,
    #     "ow": 16,
    #     "oz": 16,
    #     "kh": 3,
    #     "kw": 3,
    #     "kz": 3,
    #     "dilation": 1
    # }
    # print(compare_conv3d(**args))
    # {"oh":56,"ow":56, "oc":1, "ic":1, "kh":3, "kw":3, "dilation":1}

    # args = {
    #     "cim_row": 32,
    #     "cim_col": 8,
    #     "oh": 56,
    #     "ow": 56,
    #     "oc": 1,
    #     "ic": 1,
    #     "kh": 5,
    #     "kw": 5,
    #     "dilation": 1
    # }
    # print(compare_conv2d(**args))
    
    # args = {
    #     "cim_row": 32,
    #     "cim_col": 8,
    #     "oh": 28,
    #     "ow": 28,
    #     "oz": 28,
    #     "kh": 5,
    #     "kw": 5,
    #     "kz": 5,
    #     "dilation": 1
    # }
    # print(compare_conv3d(**args))
    
    # args_conv1d = {
    #     "cim_row": 64,
    #     "cim_col": 8,
    #     "oh": 64,
    #     "ow": 64,
    #     "oc": 1,
    #     "ic": 1,
    #     "kh": 5,
    # }
    # print(compare_conv1d_45degree(**args_conv1d))
    
    # args_conv2d = {
    #     "cim_row": 32,
    #     "cim_col": 8,
    #     "oh": 64,
    #     "ow": 1,
    #     "oc": 1,
    #     "ic": 1,
    #     "kh": 15,
    #     "kw": 1,
    # }
    # print(compare_conv2d(**args_conv2d))