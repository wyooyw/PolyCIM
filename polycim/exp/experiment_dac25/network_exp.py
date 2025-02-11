import models.onnx_parser as onnx_parser
import tempfile
import polycim.op.benchmark as benchmark
from pipeline import run_pipeline
from polycim.config import get_config
import json
import multiprocessing
import os
import istarmap
from tqdm import tqdm
def parse_op_info_into_operator(op_info,virtual_axis):

    if op_info["type"] == "conv2d":
        if op_info["group"] == 1:
            assert op_info["strides"][0] == op_info["strides"][1]
            operator = benchmark.get_op_conv2d(
                b=1, 
                oc=op_info["weight_tensor_shape"][0], 
                ic=op_info["weight_tensor_shape"][1], 
                oh=op_info["output_tensor_shape"][2], 
                ow=op_info["output_tensor_shape"][3], 
                kh=op_info["weight_tensor_shape"][2], 
                kw=op_info["weight_tensor_shape"][3], 
                stride=op_info["strides"][0],
                virtual_axis=virtual_axis
            )
            
        elif op_info["group"] == op_info["weight_tensor_shape"][0]:
            operator = benchmark.get_op_dwconv2d(
                ic=op_info["weight_tensor_shape"][0], 
                oh=op_info["output_tensor_shape"][2], 
                ow=op_info["output_tensor_shape"][3],
                kh=op_info["weight_tensor_shape"][2], 
                kw=op_info["weight_tensor_shape"][3], 
                stride=op_info["strides"][0],
                virtual_axis=virtual_axis
            )
            
        else:
            raise ValueError(f"Unsupported group number: {op_info['group']}")
    else:
        raise ValueError(f"Unsupported operator type: {op_info['type']}")

    return operator

def parse_op_info_to_key(op_info):
    keys = list(op_info.keys())
    keys = sorted(keys)
    keys = tuple(keys)

    values = []
    for key in keys:
        value = op_info[key]
        if not type(value) in (int, str):
            value = tuple(value)
        # print(value)
        # if not type(value) in (int, str, tuple):
        #     import pdb; pdb.set_trace()
        #     pass
        assert type(value) in (int, str, tuple), f"{value=}, {type(value)=}"
        values.append(value)
    values = tuple(values)

    key = (keys, values)
    print(key)
    return key
    
def _run_op(idx, op_info, skew):
    temp_dir = tempfile.mkdtemp()
    operator = parse_op_info_into_operator(op_info, virtual_axis=not skew)
    result = run_pipeline(operator, skew=skew, cim_cfg=get_config(), save_dir=temp_dir)
    return idx, result[1]

def run_op(idx, op_info):
    return _run_op(idx, op_info, skew=False)

def run_op_skew(idx, op_info):
    return _run_op(idx, op_info, skew=True)

def remove_reductant_op_info(op_info_list):
    keys = set()
    new_op_info = []
    for op_info in  op_info_list:
        key = parse_op_info_to_key(op_info)
        if key not in keys:
            new_op_info.append(op_info)
            keys.add(key)
    return new_op_info

def run_ops_parallel(op_list, skew):
    MAX_PROCESS_USE = int(os.environ.get("MAX_PROCESS_USE", 2))
    cim_flops_list = [0] * len(op_list)
    run_op_fn = run_op_skew if skew else run_op
    with multiprocessing.Pool(MAX_PROCESS_USE) as pool:
        results = pool.istarmap(run_op_fn, enumerate(op_list))
        for idx,cim_flops in tqdm(results,total=len(op_list)):
            cim_flops_list[idx] = cim_flops
    return cim_flops_list

def run_model(json_model_path, result_json_path, skew):
    with open(json_model_path, "r") as f:
        op_info_list = json.load(f)

    run_op_info_list = remove_reductant_op_info(op_info_list)
    run_op_info_key_to_index = {parse_op_info_to_key(op_info):idx for idx,op_info in enumerate(run_op_info_list)}

    run_cim_flops_list = run_ops_parallel(run_op_info_list, skew)

    cim_flops_list = []
    for op_info in op_info_list:
        op_key = parse_op_info_to_key(op_info)
        cim_flops = run_cim_flops_list[run_op_info_key_to_index[op_key]]
        cim_flops_list.append(cim_flops)

    # save cim_flops_list into json
    new_op_info_list = []
    for idx,op_info in enumerate(op_info_list):
        op_info["cim_flops"] = int(cim_flops_list[idx])
        new_op_info_list.append(op_info)
    with open(os.path.join(result_json_path), "w") as f:
        json.dump(new_op_info_list, f, indent=2)

    return cim_flops_list

if __name__=="__main__":
    for model in ["convnext_tiny"]:#, "RepLKNet-31B", "SLaK_tiny"]:
        result_dir = f"result/{model}"
        os.makedirs(result_dir, exist_ok=True)
        run_model(f"models/json/{model}.json", os.path.join(result_dir, "skew.json"), skew=True)
        run_model(f"models/json/{model}.json", os.path.join(result_dir, "im2col.json"), skew=False)