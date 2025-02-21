import islpy as isl
import os
import json
from dataclasses import dataclass
import subprocess
from polycim.passes.base import DepthFirstPass
from polycim.passes.base import Schedule
from polycim.passes.base import SchedulePassResult
from typing import Optional
from polycim.config import CIMConfig

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
    # assert os.path.exists(save_dir), f"{save_dir=}"
    # assert os.path.isdir(save_dir), f"{save_dir=}"
    if os.path.exists(save_dir):
        assert os.path.isdir(save_dir), f"{save_dir=}"
    else:
        os.makedirs(save_dir, exist_ok=True)

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

        # save buffer info
        buffer_manager = op.buffer_manager
        buffer_info_path = os.path.join(save_path_dir, "buffer_info.json")
        with open(buffer_info_path, "w") as f:
            json.dump(buffer_manager.get_buffer_name_to_info_dict(), f, indent=4)

        # Save origin operand shape
        origin_operand_shape = op.attr["origin_operand_shape"]
        with open(os.path.join(save_path_dir, "origin_operand_shape.json"), "w") as f:
            json.dump(origin_operand_shape, f, indent=4)

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
        return

        exit()
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

def backend_compile(op, save_dir, config_file):
    dsl = op.dsl

    os.makedirs(save_dir, exist_ok=True)
    cim_source_file = os.path.join(save_dir, f"code.cim")
    with open(cim_source_file, "w") as f:
        f.write(dsl)

    # run cim compiler
    subprocess.run([
        "cim-compiler", "compile",
        "--input-file", cim_source_file,
        "--output-dir", save_dir,
        "--config-file", config_file
    ], check=True)

    # save buffer info
    buffer_manager = op.buffer_manager
    buffer_info_path = os.path.join(save_dir, "buffer_info.json")
    with open(buffer_info_path, "w") as f:
        json.dump(buffer_manager.get_buffer_name_to_info_dict(), f, indent=4)

    # Save origin operand shape
    origin_operand_shape = op.attr["origin_operand_shape"]
    with open(os.path.join(save_dir, "origin_operand_shape.json"), "w") as f:
        json.dump(origin_operand_shape, f, indent=4)

    # run simulator to profile
    code_file = os.path.join(os.path.abspath(save_dir), "final_code.json")
    output_dir = os.path.join(os.path.abspath(save_dir), "output")

class BackendCompilePass(DepthFirstPass):
    def __init__(self, 
            args,
            cim_config: CIMConfig,
            fix_schedule: Optional[Schedule]=None, 
            schedule_as_key: bool=False,
        ):
        super().__init__(
            fix_schedule=fix_schedule, 
            schedule_as_key=schedule_as_key
        )
        assert self.fix_schedule is None
        assert self.schedule_as_key is False
        
        self.args = args
        self.cim_config = cim_config
        self.cnt = 0

    def apply(self, operator):
        
        backend_compile(
            operator, 
            save_dir=os.path.join(self.args.output_path, operator.attr["name"], str(self.cnt)),
            config_file=self.args.config_path
        )
        self.cnt += 1
        return [SchedulePassResult(operator, Schedule())]