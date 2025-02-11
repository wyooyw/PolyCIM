import onnx
from models.read_file import get_tensor_shape
import polycim.op.benchmark as benchmark
import json
import os
from tqdm import tqdm
def parse_conv_attr(model, node):
    name_to_attr = {attr.name: attr for attr in node.attribute}
    dilations = list(name_to_attr["dilations"].ints)
    group = name_to_attr["group"].i
    kernel_shape = list(name_to_attr["kernel_shape"].ints)
    pads = list(name_to_attr["pads"].ints)
    strides = list(name_to_attr["strides"].ints)

    input_tensor = node.input[0]
    input_tensor_shape = list(get_tensor_shape(model.graph, input_tensor))

    weight_tensor = node.input[1]
    weight_tensor_shape = list(get_tensor_shape(model.graph, weight_tensor))

    output_tensor = node.output[0]
    output_tensor_shape = list(get_tensor_shape(model.graph, output_tensor))

    if group > 1 and all(s==1 for s in strides):

        return {
            "type": "conv2d",
            "dilations": str(dilations),
            "group": str(group),
            "kernel_shape": str(kernel_shape),
            "pads": str(pads),
            "strides": str(strides),
            "input_tensor_shape": str(input_tensor_shape),
            "weight_tensor_shape": str(weight_tensor_shape),
            "output_tensor_shape": str(output_tensor_shape)
        }

    else:
        return {}

def extract_op_info_from_onnx(onnx_path):
    # 加载ONNX模型
    model = onnx.load(onnx_path)

    skip_op_type = set()

    op_info_list = []
    # 遍历模型中的每个节点
    for node in model.graph.node:
        # 检查节点是否为卷积算子
        if node.op_type == 'Conv':
            # 打印卷积算子的属性
            print(f"Node: {node.name}, Op: {node.op_type}")
            op_info = parse_conv_attr(model, node)
            # print(op_info)
            op_info_list.append(op_info)   
        else:
            skip_op_type.add(node.op_type)

    print(f"Skip op type: {skip_op_type}")

    return op_info_list

def extract_op_info_from_onnx_model_to_json(onnx_path, json_dir):
    op_info_list = extract_op_info_from_onnx(onnx_path)

    file_name = os.path.basename(onnx_path)
    file_name, file_extension = os.path.splitext(file_name)
    json_path = os.path.join(json_dir, f"{file_name}.json")

    with open(json_path, "w") as f:
        json.dump(op_info_list, f, indent=2)

def get_onnx_files(directory):
    onnx_files = []
    # os.walk遍历目录及子目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.onnx'):
                # 将文件的完整路径添加到列表中
                onnx_files.append(os.path.join(root, file))
    return onnx_files

def extract_op_info_from_onnx_to_json(onnx_dir, json_dir):
    for onnx_path in tqdm(get_onnx_files(onnx_dir)):
        print(f"{onnx_path=}")
        extract_op_info_from_onnx_model_to_json(onnx_path, json_dir)

if __name__ == "__main__":
    extract_op_info_from_onnx_to_json("models/onnx", "models/json")
    # import onnx
    # from onnx import shape_inference
    # path = "models/onnx/EfficientNet.onnx" #the path of your onnx model
    # onnx.save(onnx.shape_inference.infer_shapes(onnx.load(path)), path)