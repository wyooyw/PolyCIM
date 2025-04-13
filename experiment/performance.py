import os
import subprocess
import pandas as pd
from multiprocessing import Pool
from itertools import product
import json
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import ast

def run_polycim_op(config_path, pimsim_config_path, op_id, output_dir, op_def_json_path, options):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Construct the command
    cmd = [
        "polycim", "op",
        "--op-id", op_id,
        "--config-path", config_path,
        "--output-path", output_dir,
        "--data-movement-full-vectorize",
        "--pimsim-cfg-path", pimsim_config_path,
        "--polycim",
        "--unroll-level", "3",
        "--profile",
        "--op-def-json", op_def_json_path,
        *options
    ]
    
    # Save the command to a file
    cmd_file_path = os.path.join(output_dir, "command.txt")
    with open(cmd_file_path, "w") as cmd_file:
        cmd_file.write(" ".join(cmd) + "\n")
    
    # Define the log file path
    log_file_path = os.path.join(output_dir, "process.log")
    
    # Run the command and redirect output to the log file
    with open(log_file_path, "w") as log_file:
        subprocess.run(cmd, check=True, stdout=log_file, stderr=log_file)

def collect_results(base_output_dir, output_dirs):
    # Collect all result.csv files
    all_dfs = []
    for output_dir in output_dirs:
        result_csv_path = os.path.join(output_dir, "result.csv")
        if os.path.exists(result_csv_path):
            df = pd.read_csv(result_csv_path)
            all_dfs.append(df)
    
    # Concatenate all dataframes and save to result_all.csv
    if all_dfs:
        result_all_df = pd.concat(all_dfs, ignore_index=True)
        result_all_df.to_csv(os.path.join(base_output_dir, "result_all.csv"), index=False)
        result_all_df.to_excel(os.path.join(base_output_dir, "result_all.xlsx"), index=False)

def make_configs(demo_config, n_group_list, n_comp_list, n_bcol_list, save_dir):
    if isinstance(demo_config, str):
        with open(demo_config, "r") as f:
            demo_config = json.load(f)
    assert isinstance(demo_config, dict)
    
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    config_path_list = []
    # Iterate over all combinations of n_group, n_comp, and n_group_vcol
    for n_group, n_comp, n_bcol in product(n_group_list, n_comp_list, n_bcol_list):
        # Update the demo_config with the current combination
        demo_config["macro"]["n_group"] = n_group
        demo_config["macro"]["n_comp"] = n_comp
        demo_config["macro"]["n_bcol"] = n_bcol
        
        # Construct the filename
        filename = f"g{n_group}c{n_comp}b{n_bcol}.json"
        file_path = os.path.join(save_dir, filename)
        
        # Save the updated config to a JSON file
        with open(file_path, "w") as f:
            json.dump(demo_config, f, indent=4)

        cmd = [
            "cim-compiler", "config",
            "-i", file_path,
            "-o", file_path,
            "--yes"
        ]
        
        # Execute the command
        subprocess.run(cmd, check=True)

        config_path_list.append(file_path)
    return config_path_list

def draw_bar_chart(csv_path, save_path, labels):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Sort the dataframe by latency
    df['labels'] = labels
    df.sort_values(by='latency', inplace=True)
    
    # Extract the necessary columns
    utilization = df['utilization']
    latency = df['latency']
    sorted_labels = df['labels']
    
    # Create a horizontal bar chart with dual y-axes
    y = range(len(sorted_labels))
    height = 0.35  # the height of the bars

    # Adjust the figure size: increase width, decrease height
    fig, ax1 = plt.subplots(figsize=(12, 4))

    # Plot latency as a horizontal bar chart with a new color
    ax1.barh(y, latency, height, label='Latency', color='teal')
    ax1.set_ylabel('Configuration')
    ax1.set_xlabel('Latency (ms)', color='teal')
    ax1.set_title('Utilization and Latency by Configuration')
    ax1.set_yticks(y)
    ax1.set_yticklabels(sorted_labels)
    ax1.tick_params(axis='x', labelcolor='teal')

    # Set the maximum x-axis limit for latency
    ax1.set_xlim(right=0.1)

    # Create a second x-axis for utilization with a new color
    ax2 = ax1.twiny()
    ax2.plot(utilization, y, label='Utilization', color='coral', marker='o')
    ax2.set_xlabel('Utilization (%)', color='coral')
    ax2.tick_params(axis='x', labelcolor='coral')

    # Add legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Save the plot
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def parse_conv2d_op(idx, op):
    """
    {
        "type": "conv2d",
        "dilations": "[1, 1]",
        "group": "1",
        "kernel_shape": "[4, 4]",
        "pads": "[0, 0, 0, 0]",
        "strides": "[4, 4]",
        "input_tensor_shape": "[1, 3, 224, 224]",
        "weight_tensor_shape": "[96, 3, 4, 4]",
        "output_tensor_shape": "[1, 96, 56, 56]"
    }

    op_list["conv2d_b2o16i8h8w8k3"] = {
        "op": benchmark.get_op_conv2d(
            b=2, oc=16, ic=8, oh=8, ow=8, kh=3, kw=3, stride=1, virtual_axis=False
        ),
        "symmetry_info": ((3, 5), (4, 6)),
        "dim_types": ["b", "oc", "ic", "oh", "ow", "kh", "kw"],
        "verify_fn": conv2d,
    }
    """
    input_tensor_shape = ast.literal_eval(op["input_tensor_shape"])
    weight_tensor_shape = ast.literal_eval(op["weight_tensor_shape"])
    output_tensor_shape = ast.literal_eval(op["output_tensor_shape"])
    b,ic,ih,iw = input_tensor_shape
    oc,ic,kh,kw = weight_tensor_shape
    oh,ow = output_tensor_shape[2:]
    pads = ast.literal_eval(op["pads"])
    strides = ast.literal_eval(op["strides"])
    dilations = ast.literal_eval(op["dilations"])
    assert all(pad==pads[0] for pad in pads)
    assert all(stride==strides[0] for stride in strides)
    assert all(dilation==dilations[0] for dilation in dilations)
    assert kh == kw
    stride = strides[0]
    dilation = dilations[0]
    op_id = f"{idx}_conv2d_b{b}o{oc}i{ic}h{oh}w{ow}k{kh}k{kw}s{stride}d{dilation}"

    op_def = {
        op_id: {
            "op": f"benchmark.get_op_conv2d(b={b}, oc={oc}, ic={ic}, oh={oh}, ow={ow}, kh={kh}, kw={kw}, stride={stride}, virtual_axis=False)",
            "symmetry_info": "((3, 5), (4, 6))",
            "dim_types": "['b', 'oc', 'ic', 'oh', 'ow', 'kh', 'kw']",
            "verify_fn": f"partial(conv2d, stride={stride}, dilation={dilation})",
            "not_tiling": "[1, 2]",
        }
    }
    option = []
    # if stride == kh:
    #     option.append("--polycim-disable-pretile")
    #     option.append("--polycim-disable-affine")
    option.append("--polycim-disable-pretile")
    option.append("--polycim-disable-affine")

    return op_id, op_def, option

def parse_depthwise_conv2d_op(idx, op):
    input_tensor_shape = ast.literal_eval(op["input_tensor_shape"])
    weight_tensor_shape = ast.literal_eval(op["weight_tensor_shape"])
    output_tensor_shape = ast.literal_eval(op["output_tensor_shape"])
    b,ic,ih,iw = input_tensor_shape
    oc,ic,kh,kw = weight_tensor_shape
    oh,ow = output_tensor_shape[2:]
    pads = ast.literal_eval(op["pads"])
    strides = ast.literal_eval(op["strides"])
    dilations = ast.literal_eval(op["dilations"])
    assert all(pad==pads[0] for pad in pads)
    assert all(stride==strides[0] for stride in strides)
    assert all(dilation==dilations[0] for dilation in dilations)
    stride = strides[0]
    dilation = dilations[0]
    op_id = f"{idx}_dwconv2d_b{b}i{ic}h{oh}w{ow}k{kh}k{kw}s{stride}d{dilation}"

    op_def = {
        op_id: {
            "op": f"benchmark.get_op_dwconv2d(ic={ic}, oh={oh}, ow={ow}, kh={kh}, kw={kw}, stride={stride}, dilation={dilation}, virtual_axis=False)",
            "symmetry_info": "((1, 3), (2, 4))",
            "dim_types": "['c', 'oh', 'ow', 'kh', 'kw']",
            "verify_fn": f"partial(depth_wise_conv2d, stride={stride}, dilation={dilation})",
        }
    }

    option = []
    # if stride == kh:
    #     option.append("--polycim-disable-pretile")
    #     option.append("--polycim-disable-affine")
    option.append("--polycim-disable-pretile")
    option.append("--polycim-disable-affine")
        
    return op_id, op_def, option




def parse_network(network_path, save_dir):
    with open(network_path, "r") as f:
        network = json.load(f)
    network_name = os.path.basename(network_path).split(".")[0]
    op_ids = []
    op_defs = dict()
    options = []
    for idx,op in enumerate(network):
        # if idx >= 4:
        #     break
        weight_tensor_shape = eval(op["weight_tensor_shape"])
        out_channel = weight_tensor_shape[0]
        group = int(op["group"])
        is_depthwise = group == out_channel
        is_normal_conv = group == 1
        if is_normal_conv:
            op_id, op_def, option = parse_conv2d_op(idx, op)
        elif is_depthwise:
            op_id, op_def, option = parse_depthwise_conv2d_op(idx, op)
        else:
            print(f"Unsupported operation: {op}. skip.")
        assert op_id not in op_ids
        op_defs.update(op_def)
        op_ids.append(op_id)
        options.append(option)

    save_file_path = os.path.join(save_dir, f"op_defs_{network_name}.json")
    with open(save_file_path, "w") as f:
        json.dump(op_defs, f, indent=4)
    return op_ids, save_file_path, options

def main():
    network_name = "convnext_tiny"
    network_path = f"./polycim/exp/models/json/{network_name}.json"
    base_output_dir = f"./exp_result/performance/{network_name}_im2col_g8m8c32b64"  # Base directory for outputs  
    config_path = "/home/wangyiou/Desktop/pim_compiler/playground/polycim/exp/iccad25/compiler_configs/g8m8c32b64.json"
    pimsim_config_path = "/home/wangyiou/Desktop/pim_compiler/playground/polycim/exp/iccad25/pimsim_configs/g8m8c32b64.json"
    os.makedirs(base_output_dir, exist_ok=True)
    
    op_ids, op_def_json_path, options = parse_network(network_path, base_output_dir)
    # exit()
    n_op = len(op_ids)
    
    # Prepare output directories
    output_dirs = [os.path.join(base_output_dir, f"output_{op_id}") for op_id in op_ids]

    

    # Use multiprocessing to run polycim op commands concurrently
    with Pool(processes=min(n_op, 4)) as pool:
        pool.starmap(run_polycim_op, zip(
            [config_path] * n_op, 
            [pimsim_config_path] * n_op,
            op_ids, 
            output_dirs, 
            [op_def_json_path] * n_op,
            options
        ))

    # Collect results into a single CSV
    collect_results(base_output_dir, output_dirs)

if __name__ == "__main__":
    main()
    # draw_bar_chart(
    #     csv_path="exp_result/ablation_study_unroll3/result_all.csv",
    #     save_path="exp_result/ablation_study_unroll3/bar_chart.png",
    #     labels=["Baseline", "Disable PreTiling", "Disable Affine", "Disable Coalescing", "Random \nData Movement"]
    # )
