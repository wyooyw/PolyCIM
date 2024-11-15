import onnx
from models.read_file import get_tensor_shape
import benchmark

def parse_conv_attr(model, node):
    name_to_attr = {attr.name: attr for attr in node.attribute}
    dilations = name_to_attr["dilations"].ints
    group = name_to_attr["group"].i
    kernel_shape = name_to_attr["kernel_shape"].ints
    pads = name_to_attr["pads"].ints
    strides = name_to_attr["strides"].ints

    input_tensor = node.input[0]
    input_tensor_shape = get_tensor_shape(model.graph, input_tensor)

    weight_tensor = node.input[1]
    weight_tensor_shape = get_tensor_shape(model.graph, weight_tensor)

    output_tensor = node.output[0]
    output_tensor_shape = get_tensor_shape(model.graph, output_tensor)

    return {
        "type": "conv2d",
        "dilations": dilations,
        "group": group,
        "kernel_shape": kernel_shape,
        "pads": pads,
        "strides": strides,
        "input_tensor_shape": input_tensor_shape,
        "weight_tensor_shape": weight_tensor_shape,
        "output_tensor_shape": output_tensor_shape
    }

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



# def get_op_list_from_onnx(onnx_path):
#     op_info_list = extract_op_info_from_onnx(onnx_path)
#     op_list = parse_op_info_into_operator(op_info_list)
#     return op_list

if __name__ == "__main__":
    op_info_list = extract_op_info_from_onnx("models/convnext_tiny/convnext_tiny.onnx")
    op_list = parse_op_info_into_operator(op_info_list)