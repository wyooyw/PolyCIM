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
from datetime import datetime
def run_polycim_op(config_path, pimsim_config_path, profiler_config_path, op_id, output_dir, op_def_json_path, options, use_cache):
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    if op_id in use_cache:
        with open(os.path.join(output_dir, f"cache.txt"), "w") as f:
            f.write(use_cache[op_id])
        return

    # Construct the command
    cmd = [
        "polycim", "op",
        "--op-id", op_id,
        "--config-path", config_path,
        "--output-path", output_dir,
        "--data-movement-full-vectorize",
        "--pimsim-cfg-path", pimsim_config_path,
        "--profiler-cfg-path", profiler_config_path,
        "--polycim",
        "--unroll-level", "1",
        "--profile",
        "--op-def-json", op_def_json_path,
        "--profile-use-unrolled-code",
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
        if os.path.exists(os.path.join(output_dir, f"cache.txt")):
            with open(os.path.join(output_dir, f"cache.txt"), "r") as f:
                cache_op_id = f.read()
                op_id = output_dir.split("output_")[-1]
                output_dir = output_dir.replace(f"{op_id}", f"{cache_op_id}")
        
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


def parse_conv2d_op(idx, op, im2col=True, batch=1, cache=None, use_cache=None):
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
    b = batch
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
    op_signature = f"conv2d_b{b}o{oc}i{ic}h{oh}w{ow}k{kh}k{kw}s{stride}d{dilation}"
    op_id = f"{idx}_{op_signature}"

    if op_signature in cache:
        use_cache[op_id] = cache[op_signature]
    else:
        cache[op_signature] = op_id

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
    if im2col or stride == kh:
        option.append("--polycim-disable-pretile")
        option.append("--polycim-disable-affine")
    # option.append("--polycim-disable-pretile")
    # option.append("--polycim-disable-affine")

    return op_id, op_def, option

def parse_depthwise_conv2d_op(idx, op, im2col, batch=1, cache=None, use_cache=None):
    input_tensor_shape = ast.literal_eval(op["input_tensor_shape"])
    weight_tensor_shape = ast.literal_eval(op["weight_tensor_shape"])
    output_tensor_shape = ast.literal_eval(op["output_tensor_shape"])
    b,ic,ih,iw = input_tensor_shape
    _,_,kh,kw = weight_tensor_shape
    b = batch
    oh,ow = output_tensor_shape[2:]
    pads = ast.literal_eval(op["pads"])
    strides = ast.literal_eval(op["strides"])
    dilations = ast.literal_eval(op["dilations"])
    assert all(pad==pads[0] for pad in pads)
    assert all(stride==strides[0] for stride in strides)
    assert all(dilation==dilations[0] for dilation in dilations)
    stride = strides[0]
    dilation = dilations[0]
    op_signature = f"dwconv2d_b{b}i{ic}h{oh}w{ow}k{kh}k{kw}s{stride}d{dilation}"
    op_id = f"{idx}_{op_signature}"

    if op_signature in cache:
        use_cache[op_id] = cache[op_signature]
    else:
        cache[op_signature] = op_id

    op_def = {
        op_id: {
            "op": f"benchmark.get_op_dwconv2d(b={b}, ic={ic}, oh={oh}, ow={ow}, kh={kh}, kw={kw}, stride={stride}, dilation={dilation}, virtual_axis=False)",
            "symmetry_info": "((2, 4), (3, 5))",
            "dim_types": "['b', 'c', 'oh', 'ow', 'kh', 'kw']",
            "verify_fn": f"partial(depth_wise_conv2d, stride={stride}, dilation={dilation})",
        }
    }

    option = []
    if im2col or stride == kh:
        option.append("--polycim-disable-pretile")
        option.append("--polycim-disable-affine")
    # option.append("--polycim-disable-pretile")
    # option.append("--polycim-disable-affine")
        
    return op_id, op_def, option

def parse_group_conv2d_op(idx, op, im2col, batch=1, cache=None, use_cache=None):
    input_tensor_shape = ast.literal_eval(op["input_tensor_shape"])
    weight_tensor_shape = ast.literal_eval(op["weight_tensor_shape"])
    output_tensor_shape = ast.literal_eval(op["output_tensor_shape"])
    b,ic,ih,iw = input_tensor_shape
    _,_,kh,kw = weight_tensor_shape
    b = batch
    oc,oh,ow = output_tensor_shape[1:]
    group = int(op["group"])
    pads = ast.literal_eval(op["pads"])
    strides = ast.literal_eval(op["strides"])
    dilations = ast.literal_eval(op["dilations"])
    assert all(pad==pads[0] for pad in pads)
    assert all(stride==strides[0] for stride in strides)
    assert all(dilation==dilations[0] for dilation in dilations)
    stride = strides[0]
    dilation = dilations[0]
    op_signature = f"gconv2d_b{b}o{oc}i{ic}h{oh}w{ow}k{kh}k{kw}s{stride}d{dilation}g{group}"
    op_id = f"{idx}_{op_signature}"

    if op_signature in cache:
        use_cache[op_id] = cache[op_signature]
    else:
        cache[op_signature] = op_id

    op_def = {
        op_id: {
            "op": f"benchmark.get_op_group_conv2d(b={b}, group={group}, oc={oc}, ic={ic}, oh={oh}, ow={ow}, kh={kh}, kw={kw}, stride={stride}, virtual_axis=False)",
            "symmetry_info": "((4, 6), (5, 7))",
            "dim_types": "['b', 'g', 'oc', 'ic', 'oh', 'ow', 'kh', 'kw']",
            "verify_fn": f"partial(group_conv2d, stride={stride}, dilation={dilation})",
        }
    }

    option = []
    if im2col or stride == kh:
        option.append("--polycim-disable-pretile")
        option.append("--polycim-disable-affine")
    # option.append("--polycim-disable-pretile")
    # option.append("--polycim-disable-affine")
        
    return op_id, op_def, option



def parse_network(network_path, save_dir, im2col, batch):
    with open(network_path, "r") as f:
        network = json.load(f)
    network_name = os.path.basename(network_path).split(".")[0]
    op_ids = []
    op_defs = dict()
    options = []
    cache = dict()
    use_cache = dict()
    for idx,op in enumerate(network):
        # if idx >= 4:
        #     break
        weight_tensor_shape = eval(op["weight_tensor_shape"])
        out_channel = weight_tensor_shape[0]
        group = int(op["group"])
        is_depthwise = group == out_channel
        is_normal_conv = group == 1
        is_group_wise = 1 < group and group < out_channel
        if is_normal_conv:
            op_id, op_def, option = parse_conv2d_op(idx, op, im2col, batch, cache, use_cache)
        elif is_depthwise:
            op_id, op_def, option = parse_depthwise_conv2d_op(idx, op, im2col, batch, cache, use_cache)
        elif is_group_wise:
            op_id, op_def, option = parse_group_conv2d_op(idx, op, im2col, batch, cache, use_cache)
        else:
            print(f"Unsupported operation: {op}. skip.")
        assert op_id not in op_ids
        op_defs.update(op_def)
        op_ids.append(op_id)
        options.append(option)

    save_file_path = os.path.join(save_dir, f"op_defs_{network_name}.json")
    with open(save_file_path, "w") as f:
        json.dump(op_defs, f, indent=4)
    with open(os.path.join(save_dir, f"cache_{network_name}.json"), "w") as f:
        json.dump(cache, f, indent=4)
    return op_ids, save_file_path, options, cache, use_cache

def main(im2col, network_name, config, base_output_dir):
    # im2col=False
    im2col_str = "_im2col" if im2col else ""
    batch = 1

    polycim_home = os.environ["POLYCIM_HOME"]
    
    # network_name = "convnext_tiny"
    # config = "c32b64"
    network_path = os.path.join(polycim_home, f"polycim/exp/models/json/{network_name}.json")
    base_output_dir = os.path.join(base_output_dir, f"{network_name}_bs{batch}_{config}{im2col_str}")  # Base directory for outputs  
    config_path = os.path.join(polycim_home, f"polycim/exp/iccad25/compiler_configs/{config}.json")
    pimsim_config_path = os.path.join(polycim_home, f"polycim/exp/iccad25/cimsim_configs/{config}.json")
    profiler_config_path = f"{polycim_home}/polycim/exp/iccad25/profiler_config.json"
    os.makedirs(base_output_dir, exist_ok=True)
    
    op_ids, op_def_json_path, options, cache, use_cache = parse_network(network_path, base_output_dir, im2col=im2col, batch=batch)
    # exit()
    n_op = len(op_ids)
    # import pdb; pdb.set_trace()
    
    # Prepare output directories
    output_dirs = [os.path.join(base_output_dir, f"output_{op_id}") for op_id in op_ids]

    

    # Use multiprocessing to run polycim op commands concurrently
    with Pool(processes=min(n_op, 4)) as pool:
        pool.starmap(run_polycim_op, zip(
            [config_path] * n_op, 
            [pimsim_config_path] * n_op,
            [profiler_config_path] * n_op,
            op_ids, 
            output_dirs, 
            [op_def_json_path] * n_op,
            options,
            [use_cache] * n_op
        ))
    # import pdb; pdb.set_trace()

    # Collect results into a single CSV
    collect_results(base_output_dir, output_dirs)

def get_total_utilization(df, n_comp, n_group_vcol):
    total_flops = df['flops'].sum()
    total_compute_ops = df['compute_ops'].sum()
    flops_per_cim_compute = total_flops / total_compute_ops
    peak_flops_per_cim_compute = (
        n_comp * n_group_vcol
    )
    use_rate_percent = flops_per_cim_compute / peak_flops_per_cim_compute * 100
    return use_rate_percent

def gather_results(im2col_pth, polycim_pth, output_path, n_comp, n_group_vcol):
    im2col_df = pd.read_csv(im2col_pth)
    polycim_df = pd.read_csv(polycim_pth)
    
    # Calculate the sum of the latency column for each dataframe
    im2col_latency_sum = im2col_df['latency'].sum()
    polycim_latency_sum = polycim_df['latency'].sum()

    # total utilization
    im2col_total_utilization = get_total_utilization(im2col_df, n_comp, n_group_vcol)
    polycim_total_utilization = get_total_utilization(polycim_df, n_comp, n_group_vcol)
    
    # Create a new dataframe with the results
    results_df = pd.DataFrame({
        'Method': ['im2col', 'polycim'],
        'Total Latency': [im2col_latency_sum, polycim_latency_sum],
        'Total Utilization': [im2col_total_utilization, polycim_total_utilization]
    })

    
    # Save the new dataframe to the specified output path
    results_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    # main(im2col=False, network_name="convnext_tiny", config="g8m8c32b64")
    # main(im2col=False, network_name="convnext_tiny", config="c64b64")


    # time_str = datetime.now().strftime("%m-%d_%H-%M-%S") 
    # base_output_dir = f"./exp_result/performance_network/{time_str}"
    # for config in ["g8m8c64b64"]:
    #     for network_name in ["mobilenet_v2", "convnext_tiny", "EfficientNet"]:
    #     # for network_name in ["EfficientNet"]:
    #         main(im2col=True, network_name=network_name, config=config, base_output_dir=base_output_dir)
    #         main(im2col=False, network_name=network_name, config=config, base_output_dir=base_output_dir)
    
    
    # draw_bar_chart(
    #     csv_path="exp_result/ablation_study_unroll3/result_all.csv",
    #     save_path="exp_result/ablation_study_unroll3/bar_chart.png",
    #     labels=["Baseline", "Disable PreTiling", "Disable Affine", "Disable Coalescing", "Random \nData Movement"]
    # )

    base_output_dir = "exp_result/performance_network/05-12_10-02-29"
    n_comp = 64
    n_group_vcol = 64 // 8  
    for base_name in ["convnext_tiny_bs1_g8m8c64b64", "EfficientNet_bs1_g8m8c64b64", "mobilenet_v2_bs1_g8m8c64b64"]:
        output_path = os.path.join(base_output_dir, f"compare_{base_name}.csv")
        gather_results(
            im2col_pth=os.path.join(base_output_dir, f"{base_name}_im2col/result_all.csv"),
            polycim_pth=os.path.join(base_output_dir, f"{base_name}/result_all.csv"),
            output_path=output_path,
            n_comp=n_comp,
            n_group_vcol=n_group_vcol
        )
    