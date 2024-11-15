import models.onnx_parser as onnx_parser
import tempfile
import benchmark
from pipeline import run_pipeline
from config import get_config

def parse_op_info_into_operator(op_info):

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
            )
            
        elif op_info["group"] == op_info["weight_tensor_shape"][0]:
            operator = benchmark.get_op_dwconv2d(
                ic=op_info["weight_tensor_shape"][0], 
                oh=op_info["output_tensor_shape"][2], 
                ow=op_info["output_tensor_shape"][3], 
                kh=op_info["weight_tensor_shape"][2], 
                kw=op_info["weight_tensor_shape"][3], 
                stride=op_info["strides"][0],
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
    
op_cache = dict()
def run_op(op_info, save_dir):
    global op_cache

    temp_dir = tempfile.mkdtemp()
    key = parse_op_info_to_key(op_info)
    if key in op_cache:
        return
    
    operator = parse_op_info_into_operator(op_info)
    
    result = run_pipeline(operator, skew=True, cim_cfg=get_config(), save_dir=temp_dir)

    op_cache[key] = result

def run_model(onnx_path, save_dir):
    op_info_list = onnx_parser.extract_op_info_from_onnx(onnx_path)
    for op_info in op_info_list:
        run_op(op_info, save_dir)

if __name__=="__main__":
    run_model("models/convnext_tiny/convnext_tiny.onnx", ".temp_save")