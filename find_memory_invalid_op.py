import json
from functools import reduce
from polycim.config import set_raw_config_by_path, get_config
import math


def calculate_input_memory(X_shape, W_shape, padding):
    cim_cfg = get_config()

    # 计算卷积操作的输入内存占用
    # 这里假设每个元素占用1个单位的内存
    # 你可以根据实际情况调整计算方法
    batch, ic, ih, iw = X_shape
    oc,ic,kh,kw = W_shape
    out_h = ih - kh + 1 + 2 * padding[0]
    out_w = iw - kw + 1 + 2 * padding[1]
    
    
    kernel_size = ic * kh * kw
    kernel_size_pad = math.ceil(kernel_size / (cim_cfg.n_comp * cim_cfg.n_row)) * (cim_cfg.n_comp * cim_cfg.n_row)
    input_memory = batch * out_h * out_w * kernel_size_pad
    
    # input_memory = reduce(lambda x, y: x * y, X_shape) * 3

    return input_memory

def calculate_dw_input_memory(X_shape, W_shape, padding):
    return reduce(lambda x, y: x * y, X_shape)

def find_memory_invalid_ops(model_data, hardware_memory_limit):
    total_conv = 0
    invalid_conv = 0
    invalid_dw_conv = 0
    for core, core_data in model_data.items():
        for stage, stage_data in core_data['stages'].items():
            for idx, instruction in enumerate(stage_data['instructions']):
                if instruction['op'] == 'conv':
                    X_shape = instruction['attr']['X_shape']
                    W_shape = instruction['attr']['W_shape']
                    padding = instruction['attr']['padding']
                    input_memory = calculate_input_memory(X_shape, W_shape, padding)
                    
                    if input_memory > hardware_memory_limit:
                        # print(f"Core: {core}, Stage: {stage}, Instruction Index: {idx}")
                        # print(f"Operation: {instruction}")
                        # print(f"input_memory: {input_memory}")
                        # print("\n")
                        invalid_conv += 1
                elif instruction['op'] == 'depthwise_conv':
                    X_shape = instruction['attr']['X_shape']
                    W_shape = instruction['attr']['W_shape']
                    padding = instruction['attr']['padding']
                    input_memory = calculate_dw_input_memory(X_shape, W_shape, padding)
                    
                    if input_memory > hardware_memory_limit:
                        # print(f"Core: {core}, Stage: {stage}, Instruction Index: {idx}")
                        # print(f"Operation: {instruction}")
                        # print(f"input_memory: {input_memory}")
                        # print("\n")
                        invalid_dw_conv += 1
                total_conv += 1
    print(f"total_conv: {total_conv}")
    percent = invalid_conv/total_conv*100
    print(f"invalid_conv: {invalid_conv} ({percent:.2f}%)")
    percent = invalid_dw_conv/total_conv*100
    print(f"invalid_dw_conv: {invalid_dw_conv} ({percent:.2f}%)")


test_data = [
    (
        "graphs/instructions_resnet18_0.5x_load_time_T4_B8.json",
        "config/dac25/config_gs_4.json",
    ),
    (
        "graphs/instructions_resnet18_0.5x_load_time_T8_B8.json",
        "config/dac25/config_gs_8.json",
    ),
    (
        "graphs/instructions_resnet18_0.5x_load_time_T12_B8.json",
        "config/dac25/config_gs_12.json",
    ),
    (
        "graphs/instructions_resnet18_0.5x_load_time_T16_B8.json",
        "config/dac25/config_gs_16.json",
    ),
    (
        "graphs/instructions_mobilenet_0.5x_load_time_T4_B8.json",
        "config/dac25/config_gs_4.json",
    ),
    (
        "graphs/instructions_mobilenet_0.5x_load_time_T8_B8.json",
        "config/dac25/config_gs_8.json",
    ),
    (
        "graphs/instructions_mobilenet_0.5x_load_time_T12_B8.json",
        "config/dac25/config_gs_12.json",
    ),
    (
        "graphs/instructions_mobilenet_0.5x_load_time_T16_B8.json",
        "config/dac25/config_gs_16.json",
    ),
]

for graph, config in test_data:
    set_raw_config_by_path(config)
    with open(graph, "r") as f:
        model_json = json.load(f)
    print("=" * 20)
    print(f"graph: {graph}, \nconfig: {config}")
    hardware_memory_limit = 262144  # 假设的硬件内存限制
    find_memory_invalid_ops(model_json, hardware_memory_limit)

