import islpy as isl
import os
import json
from dataclasses import dataclass
import subprocess
def dump_op_basic_info(op, path):
    n_compute = op.domain.count_val()
    with open(path, "w") as f:
        f.write(f"n_compute: {n_compute}\n")
        f.write(f"domain: {op.domain}\n")
        f.write(f"access_I: {op.access_I}\n")
        f.write(f"access_O: {op.access_O}\n")
        f.write(f"access_W: {op.access_W}\n")

@dataclass
class ProfileResult:
    stats: str
    save_path: int

def backend_compile_and_profile_pass(op_list, save_dir, config_file):
    assert save_dir is not None
    assert os.path.exists(save_dir), f"{save_dir=}"
    assert os.path.isdir(save_dir), f"{save_dir=}"
    assert os.path.exists(config_file), f"{config_file=}"
    assert os.path.isfile(config_file), f"{config_file=}"
    
    backend_compile_cmd_list = []
    result_list = []
    for idx,op in enumerate(op_list):
        dsl = op.dsl
        save_path_dir = os.path.join(save_dir, f"{idx}")
        os.makedirs(save_path_dir, exist_ok=True)
        save_path_file = os.path.join(save_path_dir, f"{idx}.cim")
        with open(save_path_file, "w") as f:
            f.write(dsl)

        # run cim compiler
        subprocess.run([
            "cim-compiler", "compile",
            "--input-file", save_path_file,
            "--output-dir", save_path_dir,
            "--config-file", config_file
        ], check=True)

        # run simulator to profile
        code_file = os.path.join(os.path.abspath(save_path_dir), "final_code.json")
        output_dir = os.path.join(os.path.abspath(save_path_dir), "output")
        # -i /home/wangyiou/project/cim_compiler_frontend/playground/.result/2024-12-14/AlexNet/bit_sparse/0_conv/final_code.json \
        # -d /home/wangyiou/project/cim_compiler_frontend/playground/.result/2024-12-14/AlexNet/bit_sparse/0_conv/global_image \
        # -o temp/output \
        # -c config/config.json \
        # --code-format legacy \
        # --save-unrolled-code \
        # --save-stats
        subprocess.run([
            "cim-compiler", "simulate",
            "-i", code_file,
            "-o", output_dir,
            "-c", config_file,
            "--code-format", "cimflow",
            "--save-stats"
        ], check=True)
        exit()
        # os.makedirs(output_path, exist_ok=True)
        # config_path = "/home/wangyiou/project/cim_compiler_frontend/playground/config/config.json"
        # simulator_path = "/home/wangyiou/project/cim_compiler_frontend/playground"
        # cd_cmd = f"cd {simulator_path}"
        # run_cmd = f"python utils/simulate_and_stats.py --input {input_path} --output {output_path} --config {config_path}" 
        # backend_simulator_cmd = f"{cd_cmd} && {run_cmd}"  
        # os.system(backend_simulator_cmd) 

        # save op info
        dump_op_basic_info(op, os.path.join(save_path_dir, "op_info.txt"))

        # stats json
        stats_json_path = os.path.join(output_path, "stats.json")
        if os.path.exists(stats_json_path):
            with open(stats_json_path, "r") as f:
                stats = json.load(f)
        else:
            stats = None
        
        result = ProfileResult(stats=stats, save_path=save_path_dir)
        result_list.append(result)
    return result_list