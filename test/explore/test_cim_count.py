import subprocess
import tempfile
import os
import json
import pytest

@pytest.mark.parametrize("cim_cfg_path, op_id, cim_count", [
    # *[("configs/c16b32.json", op_id, cim_count)
    #     for op_id, cim_count in [
    #         ("C1", 3136), ("C2", 784),
    #     ]
    # ],
    *[("configs/c32b64.json", op_id, cim_count)
        for op_id, cim_count in [
            ("C1", 1568), ("C2", 392),
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
    
        # get stats.json
        stats_path = os.path.join(temp_dir, op_id, "0", "output", "stats.json")
        assert os.path.exists(stats_path), f"{stats_path=}"
        with open(stats_path, "r") as f:
            stats = json.load(f)
        assert stats["CIMComputeInst"] == cim_count
