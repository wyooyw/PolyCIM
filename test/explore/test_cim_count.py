import subprocess
import tempfile
import os
import json
import pytest

@pytest.mark.parametrize("cim_cfg_path, op_id, cim_count", [
    *[("configs/c16b32.json", op_id, cim_count)
        for op_id, cim_count in [
            ("test", 4), #("C1", 3136), ("C2", 784), 
        ]
    ],
    *[("configs/c32b64.json", op_id, cim_count)
        for op_id, cim_count in [
            ("C1", 1568), ("C2", 392),
        ]
    ],
    *[("configs/g2m2c16b32.json", op_id, cim_count)
        for op_id, cim_count in [
            ("test", 2), #("C1", 3136), ("C2", 784), 
        ]
    ],
])
def test_cim_count(cim_cfg_path, op_id, cim_count):
    cim_cfg_path = os.path.join(os.path.dirname(__file__), cim_cfg_path)
    with tempfile.TemporaryDirectory() as temp_dir:
        subprocess.run([
            "polycim", "explore",
            "--op-id", op_id,
            "--config-path", cim_cfg_path,
            "--output-path", temp_dir
        ], check=True)

        # run simulator
        op_dir = os.path.join(temp_dir, op_id, "0")
        code_path = os.path.join(op_dir, "final_code.json")
        sim_output_dir = os.path.join(op_dir, "sim_output")
        subprocess.run([
            "cim-compiler", "simulate",
            "--code-file", code_path,
            "--config-file", cim_cfg_path,
            "--output-dir", sim_output_dir,
            "--code-format", "cimflow",
            "--save-stats"
        ], check=True)
    
        # get stats.json
        stats_path = os.path.join(sim_output_dir, "stats.json")
        assert os.path.exists(stats_path), f"{stats_path=}"
        with open(stats_path, "r") as f:
            stats = json.load(f)
        assert stats["CIMComputeInst"] == cim_count, f"{stats['CIMComputeInst']=}"

if __name__ == "__main__":
    test_cim_count("configs/g2m2c16b32.json", "test", 2)
