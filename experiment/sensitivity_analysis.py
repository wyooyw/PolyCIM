import os
import subprocess
import pandas as pd
from multiprocessing import Pool
from itertools import product
import json
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from datetime import datetime

def run_polycim_op(config_path, op_id, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct the command
    cmd = [
        "polycim", "op",
        "--op-id", op_id,
        "--config-path", config_path,
        "--output-path", output_dir,
        "--data-movement-full-vectorize",
        "--polycim",
        "--unroll-level", "1",
        # "--polycim-disable-affine"
        # "--verify",
        # "--profile"
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

def draw_heatmap(csv_path, save_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # df = df[(df['cim_config.n_comp'] <= 80) & (df['cim_config.n_bcol'] <= 160)]


    # Pivot the DataFrame to get the format suitable for a heatmap
    heatmap_data = df.pivot(index="cim_config.n_comp", columns="cim_config.n_bcol", values="utilization")

    # Create the heatmap with a reversed colormap
    plt.figure(figsize=(8, 4))
    ax = sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="YlGnBu_r", cbar_kws={'format': '%.0f%%'},
                     )

    # Manually format the annotations with a percentage sign
    for text in ax.texts:
        text.set_text(f"{float(text.get_text()):.1f}%")

    # Set the labels and title
    plt.xlabel("Number of rows")
    plt.ylabel("Number of columns")
    # plt.title("Utilization Heatmap")

    # Save the heatmap as a PDF file
    plt.savefig(save_path)
    plt.close()

def main(op_id, config_name, base_output_dir):
    polycim_home = os.environ["POLYCIM_HOME"]

    base_output_dir = os.path.join(base_output_dir, op_id)

    # Define the list of config paths and other parameters
    config_paths = make_configs(
        demo_config=os.path.join(polycim_home, "polycim/exp/iccad25/compiler_configs", config_name),
        n_group_list=[1],
        n_comp_list=list(range(16,16 + 16 * 5, 16)),
        n_bcol_list=list(range(32,32 + 32 * 5, 32)),
        # n_comp_list=[32],
        # n_bcol_list=[64],
        save_dir=os.path.join(base_output_dir, "configs")
    )
    
    # Prepare output directories
    output_dirs = [os.path.join(base_output_dir, f"output_{i}") for i in range(len(config_paths))]

    # Use multiprocessing to run polycim op commands concurrently
    with Pool(processes=min(len(config_paths), 4)) as pool:
        pool.starmap(run_polycim_op, zip(config_paths, [op_id]*len(config_paths), output_dirs))

    # Collect results into a single CSV
    collect_results(base_output_dir, output_dirs)

def main_change_group():
    op_id = "C1"  # Example operator ID
    base_output_dir = "./exp_result/sensitivity_analysis/change_group"  # Base directory for outputs

    # Define the list of config paths and other parameters
    config_paths = make_configs(
        demo_config="/home/wangyiou/Desktop/pim_compiler/playground/polycim/exp/iccad25/compiler_configs/c32b64.json",
        n_group_list=[8,16,24,32,40,48,56,64],
        n_comp_list=[32],
        n_bcol_list=[64],
        save_dir=os.path.join(base_output_dir, "configs")
    )
    
    # result_all_path = "./result_all.csv"  # Path for the final concatenated result

    # Prepare output directories
    output_dirs = [os.path.join(base_output_dir, f"output_{i}") for i in range(len(config_paths))]

    # Use multiprocessing to run polycim op commands concurrently
    with Pool(processes=min(len(config_paths), 4)) as pool:
        pool.starmap(run_polycim_op, zip(config_paths, [op_id]*len(config_paths), output_dirs))

    # Collect results into a single CSV
    collect_results(base_output_dir, output_dirs)

if __name__ == "__main__":
    # time_str = datetime.now().strftime("%m-%d_%H-%M-%S") 
    # base_output_dir = f"./exp_result/sensitivity_analysis/{time_str}"
    # op_ids = [f"new_C{i}" for i in range(1, 9)]
    # # op_ids = [f"new_C1"]
    # config_name = "c32b64.json"
    # for op_id in op_ids:
    #     main(op_id, config_name, base_output_dir)


    
    draw_heatmap(
        csv_path="exp_result/sensitivity_analysis/05-12_06-21-02/new_C7/result_all.csv",
        save_path="exp_result/sensitivity_analysis/05-12_06-21-02/new_C7/heatmap.png"
    )
