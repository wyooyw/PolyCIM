import os
import subprocess
import pandas as pd
from multiprocessing import Pool
from itertools import product
import json
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def run_polycim_op(config_path, pimsim_config_path, op_id, output_dir, options):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    assert isinstance(options, list)
    assert all(isinstance(option, str) for option in options)

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

    min_latency = df['latency'].min()
    df['latency'] = df['latency'] / min_latency
    
    # Sort the dataframe by latency
    df['labels'] = labels
    df.sort_values(by='latency', inplace=True)

    # print(df)
    
    # Extract the necessary columns
    utilization = df['utilization']
    latency = df['latency']
    sorted_labels = df['labels']
    print(f"{utilization=}")
    print(f"{latency=}")
    print(f"{sorted_labels=}")
    
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

def main():
    op_id = "C2"  # Example operator ID
    base_output_dir = "./exp_result/ablation_study_unroll3"  # Base directory for outputs

    # Define the list of config paths and other parameters
    # config_path = make_configs(
    #     demo_config="/home/wangyiou/Desktop/pim_compiler/playground/polycim/exp/iccad25/compiler_configs/c32b64.json",
    #     n_group_list=[1],
    #     n_comp_list=[32],
    #     n_bcol_list=[64],
    #     save_dir=os.path.join(base_output_dir, "configs")
    # )[0]
    config_path = "/home/wangyiou/Desktop/pim_compiler/playground/polycim/exp/iccad25/compiler_configs/c32b64.json"
    pimsim_config_path = "/home/wangyiou/Desktop/pim_compiler/playground/polycim/exp/iccad25/pimsim_configs/c32b64.json"

    options = [
        [], # baseline
        ["--polycim-disable-pretile"],
        ["--polycim-disable-affine"],
        ['--disable-hardware-mapping-coalescing'],
        ['--data-movement-search', '--data-movement-search-time', '0'],
    ]
    
    # result_all_path = "./result_all.csv"  # Path for the final concatenated result

    # Prepare output directories
    output_dirs = [os.path.join(base_output_dir, f"output_{i}") for i in range(len(options))]

    # Use multiprocessing to run polycim op commands concurrently
    with Pool(processes=min(len(options), 4)) as pool:
        pool.starmap(run_polycim_op, zip(
            [config_path] * len(options), 
            [pimsim_config_path] * len(options),
            [op_id]*len(options), 
            output_dirs, 
            options
        ))

    # Collect results into a single CSV
    collect_results(base_output_dir, output_dirs)

if __name__ == "__main__":
    # main()
    draw_bar_chart(
        csv_path="exp_result/ablation_study_unroll3/result_all.csv",
        save_path="exp_result/ablation_study_unroll3/bar_chart.png",
        labels=["Baseline", "Disable PreTiling", "Disable Affine", "Disable Coalescing", "Random \nData Movement"]
    )
