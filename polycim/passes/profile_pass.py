import json
import os
import subprocess

from polycim.passes.base import BreadthFirstPass
from polycim.utils.tee import Tee
import sys

def profile(temp_dir, pimsim_cfg_path, profiler_cfg_path, op_name, op_id, use_unrolled_code=False):
    # 1. convert format
    op_dir = os.path.join(temp_dir, op_name, op_id)
    if use_unrolled_code:
        cimflow_code_path = os.path.join(op_dir, "sim_output", "unrolled_code.json")
    else:
        cimflow_code_path = os.path.join(op_dir, "final_code.json")
    legacy_code_path = os.path.join(op_dir, "final_code.cimflow.json")

    subprocess.run(
        [
            "cim-compiler",
            "convert",
            "--src-type",
            "cimflow",
            "--dst-type",
            "cimflow",
            "--src-file",
            cimflow_code_path,
            "--dst-file",
            legacy_code_path,
            "--filter-out-invalid-instructions",
        ],
        check=True,
    )

    # 2. profiler-config
    os.makedirs(os.path.join(op_dir, "cim-sim"), exist_ok=True)
    report_path = os.path.join(op_dir, "cim-sim", f"cimsim_report.json")
    profiler_report_path = os.path.join(op_dir, "cim-sim", "profiler_report.json")
    with open(profiler_cfg_path, "r") as f:
        profiler_cfg = json.load(f)
    profiler_cfg["json_file"] = profiler_report_path
    save_profiler_cfg_path = os.path.join(op_dir, "cim-sim", "profiler_config.json")
    with open(save_profiler_cfg_path, "w") as f:
        json.dump(profiler_cfg, f, indent=2)

    # 2. profile
    """
    pimsim ./pimsim_configs/config-m1g1c32b64.json C1.json -r -c -j ./C1.report.json
    """
    pimsim_cmd = [
        "cim-sim",
        pimsim_cfg_path,
        save_profiler_cfg_path,
        # 加一个profile config地址
        legacy_code_path,
        "-r",
        # "-c",
        "-j",
        report_path,
    ]
    pimsim_cmd_str = " ".join(pimsim_cmd)
    print(f"{pimsim_cmd_str}")
    log_file_path = os.path.join(op_dir, "cim-sim", "cimsim.log")
    with open(log_file_path, "w") as log_file:
        process = subprocess.Popen(
            pimsim_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # 实时读取输出并同时写入控制台和文件
        for line in process.stdout:
            sys.stdout.write(line)  # 打印到控制台
            sys.stdout.flush()  # 确保实时显示
            log_file.write(line)  # 写入文件

        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, pimsim_cmd)

    # 3. parse report
    with open(report_path, "r") as f:
        report = json.load(f)
    return report


class ProfilePass(BreadthFirstPass):
    def __init__(self, args):
        super().__init__()
        self.op_list = list()
        self.args = args

    def apply(self, operator):
        self.op_list.append(operator)

    def apply_all(self):
        for i, op in enumerate(self.op_list):
            # op_dir = os.path.join(self.args.output_path, op.attr["name"], str(i))
            report = profile(
                self.args.output_path,
                self.args.pimsim_cfg_path,
                self.args.profiler_cfg_path,
                op.attr["name"],
                str(i),
                use_unrolled_code=self.args.profile_use_unrolled_code,
            )
            op.attr["ProfilePass"] = {
                "latency": report["latency_"],
                "total_energy": report["total_energy_"],
            }

    def get_result(self):
        return self.op_list
