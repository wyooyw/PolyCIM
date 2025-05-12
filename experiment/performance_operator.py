import os
import subprocess
import pandas as pd
from multiprocessing import Pool
from itertools import product
import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def run_polycim_op(config_path, pimsim_config_path, profiler_config_path, op_id, output_dir, op_def_json_path, options):
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
        "--profiler-cfg-path", profiler_config_path,
        "--polycim",
        "--unroll-level", "1",
        "--profile",
        "--profile-use-unrolled-code",
        *options
    ]

    if op_def_json_path is not None:
        assert isinstance(op_def_json_path, str)
        assert os.path.isfile(op_def_json_path)
        cmd.extend([
            "--op-def-json", op_def_json_path
        ])

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

def draw_bar_chart(csv_path_list_groups, group_names, save_path, labels):
    # csv_path_list_groups is a list of lists, each containing paths for one hardware configuration
    num_groups = len(csv_path_list_groups)
    
    # Read all CSV files into a list of DataFrames
    all_dataframes = []
    for csv_path_list in csv_path_list_groups:
        dataframes = [pd.read_csv(csv_path) for csv_path in csv_path_list]
        all_dataframes.append(dataframes)

    # Assume all CSVs have the same structure and the 'latency' column exists
    all_latency_data = [[df['latency'] for df in dataframes] for dataframes in all_dataframes]
    all_x_tick_labels = [list([df['name'] for df in dataframes][1]) for dataframes in all_dataframes]

    # Create a single bar chart
    fig, ax = plt.subplots(figsize=(8 * num_groups, 6))

    # Calculate speedup with respect to the first CSV in each group
    all_speedup_data = [[latency_data[0] / latency for latency in latency_data] for latency_data in all_latency_data]
    num_rows = len(all_latency_data[0][0])

    # Create a bar chart
    x = np.arange(num_rows)  # the label locations
    width = 0.2  # the width of the bars

    for group_idx, (speedup_data, x_tick_labels) in enumerate(zip(all_speedup_data, all_x_tick_labels)):
        for i, (speedup, label) in enumerate(zip(speedup_data, labels)):
            ax.bar(x + i * width + group_idx * (num_rows + 1) * width, speedup, width, label=label if group_idx == 0 else "")

        # Add dashed line to separate groups
        if group_idx < num_groups - 1:
            ax.axvline(x=(group_idx + 1) * (num_rows + 1) * width - width / 2, color='gray', linestyle='--', linewidth=1)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Operators')
    ax.set_ylabel('Speedup')
    ax.set_title('Speedup for Different Macros')
    ax.set_xticks(x + width * (len(labels) - 1) / 2 + (num_rows + 1) * width * np.arange(num_groups))
    ax.set_xticklabels([label for x_tick_labels in all_x_tick_labels for label in x_tick_labels])

    # Add a shared legend
    ax.legend(loc='upper center', ncol=len(labels), bbox_to_anchor=(0.5, 1.05))

    # Remove the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    fig.tight_layout()

    # Save the plot
    plt.savefig(save_path)
    plt.close()

def main(op_ids, im2col, config_name, base_out_dir):
    os.makedirs(base_out_dir, exist_ok=True)
    batch = 1
    # network_name = "EfficientNet"
    # network_path = f"./polycim/exp/models/json/{network_name}.json"
    polycim_home = os.environ["POLYCIM_HOME"]

    # im2col_name = "_im2col" if im2col else "_polycim"
    base_output_dir = base_out_dir #os.path.join(base_out_dir, f"bs{batch}{im2col_name}_{config_name}")  # Base directory for outputs  
    config_path = f"{polycim_home}/polycim/exp/iccad25/compiler_configs/{config_name}.json"
    pimsim_config_path = f"{polycim_home}/polycim/exp/iccad25/cimsim_configs/{config_name}.json"
    profiler_config_path = f"{polycim_home}/polycim/exp/iccad25/profiler_config.json"
    os.makedirs(base_output_dir, exist_ok=True)
    assert os.path.isfile(config_path), config_path
    assert os.path.isfile(pimsim_config_path), pimsim_config_path
    
    # op_ids, op_def_json_path, options = parse_network(network_path, base_output_dir, im2col=im2col, batch=batch)
    # exit()
    n_op = len(op_ids)
    op_def_json_path = None
    if im2col:
        options = [
            ("--polycim-disable-pretile", "--polycim-disable-affine")
        ] * n_op
    else:
        options = [
            []
        ] * n_op
    
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
            options
        ))

    # Collect results into a single CSV
    collect_results(base_output_dir, output_dirs)


def gather_result(polycim_pth, im2col_pth, output_path):
    """
    Gather along columns
    for each column in polycim_pth, change column name from 'col' to 'polycim.col'
    for each column in im2col_pth, change column name from 'col' to 'im2col.col'
    then gather into a single dataframe
    save csv and excel to output path
    """
    polycim_df = pd.read_csv(polycim_pth)
    im2col_df = pd.read_csv(im2col_pth)
    for col in polycim_df.columns:
        polycim_df.rename(columns={col: f"polycim.{col}"}, inplace=True)
    for col in im2col_df.columns:
        im2col_df.rename(columns={col: f"im2col.{col}"}, inplace=True)
    result_df = pd.concat([polycim_df, im2col_df], axis=1)

    # calculate new columns: speedup
    def skip_by_keywords(col_name, keywords):
        return any(keyword in col_name for keyword in keywords)
    def keep_by_keywords(col_name, keywords):
        return any(keyword in col_name for keyword in keywords)
    result_df["speedup"] = result_df["im2col.latency"] / result_df["polycim.latency"]
    result_df["less_compute_ops_rate"] = (result_df["im2col.compute_ops"] - result_df["polycim.compute_ops"]) / result_df["im2col.compute_ops"]
    columns_order = ([col for col in result_df.columns if keep_by_keywords(col, [
        "latency", "compute_ops", "utilization", "data_movement_cost_value", "name", "speedup", "less_compute_ops_rate"
    ])])
    result_df = result_df[columns_order]


    # save
    result_df.to_csv(output_path, index=False)
    result_df.to_excel(output_path.replace(".csv", ".xlsx"), index=False)

if __name__ == "__main__":
    time_str = datetime.now().strftime("%m-%d_%H-%M-%S") 
    output_dir = f"./exp_result/performance_operator/{time_str}"
    # op_ids = [f"new_C{i}" for i in range(1, 6)]
    op_ids = [f"new_C{i}" for i in [7,8]]
    # op_ids = [f"new_C3"]

    
    for config_name in ["g8m8c16b32", "g8m8c32b64", "g8m8c64b64"]:
        config_output_dir = os.path.join(output_dir, f"{config_name}")
        print(f"{config_name=}")
        # for im2col in [True, False]:
        im2col_base_output_dir = os.path.join(config_output_dir, f"imcol")
        main(op_ids, True, config_name, im2col_base_output_dir)
        polycim_base_output_dir = os.path.join(config_output_dir, f"polycim")
        main(op_ids, False, config_name, polycim_base_output_dir)
        
        gather_result(
            polycim_pth=os.path.join(polycim_base_output_dir, "result_all.csv"),
            im2col_pth=os.path.join(im2col_base_output_dir, "result_all.csv"),
            output_path=os.path.join(config_output_dir, f"compare_{config_name}.csv")
        )
    # main(True, "c32b64", "./exp_result/performance_operator_unrolled_c3d")
    # main(False, "c32b64", "./exp_result/performance_operator_unrolled_c3d")
    # main(True, "c64b64", "./exp_result/performance_operator_unrolled_c3d")
    # main(False, "c64b64", "./exp_result/performance_operator_unrolled_c3d")
    # draw_bar_chart(
    #     csv_path_list_groups=[
    #         [
    #             "exp_result/performance_operator/bs1_im2col_c16b32/result_all.csv",
    #             "exp_result/performance_operator/bs1_c16b32/result_all.csv",
    #         ],
    #         [
    #             "exp_result/performance_operator/bs1_im2col_c32b64/result_all.csv",
    #             "exp_result/performance_operator/bs1_c32b64/result_all.csv",
    #         ],
    #         [
    #             "exp_result/performance_operator/bs1_im2col_c64b64/result_all.csv",
    #             "exp_result/performance_operator/bs1_c64b64/result_all.csv",
    #         ],
    #     ],
    #     group_names=["16x32", "32x64", "64x64"],
    #     save_path="./bar_chart.png",
    #     labels=["Im2Col", "PolyCIM"]
    # )

    # gather_result(
    #     polycim_pth="./exp_result/performance_operator/bs1_c64b64/result_all.csv",
    #     im2col_pth="./exp_result/performance_operator/bs1_im2col_c64b64/result_all.csv",
    #     output_path="./compare_bs1_c64b64.csv"
    # )

    # gather_result(
    #     polycim_pth="./exp_result/performance_operator_unrolled_c3d/bs1_c16b32/result_all.csv",
    #     im2col_pth="./exp_result/performance_operator_unrolled_c3d/bs1_im2col_c16b32/result_all.csv",
    #     output_path="./compare_bs1_c16b32_unrolled.csv"
    # )
