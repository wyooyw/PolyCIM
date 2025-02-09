import subprocess
import tempfile
import os
import json
import pytest

@pytest.mark.parametrize("op_id, cim_count", [
    ("C1", 3136),
])
def test_cim_count(op_id, cim_count):
    with tempfile.TemporaryDirectory() as temp_dir:
        subprocess.run([
            "polycim", "explore",
            "--op-id", op_id,
            "--output-path", temp_dir
        ], check=True)
    
        # get stats.json
        stats_path = os.path.join(temp_dir, op_id, "0", "output", "stats.json")
        assert os.path.exists(stats_path), f"{stats_path=}"
        with open(stats_path, "r") as f:
            stats = json.load(f)
        assert stats["CIMComputeInst"] == cim_count
