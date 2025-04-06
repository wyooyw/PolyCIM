import subprocess
import tempfile
import os
import json
import pytest
import numpy as np
from polycim.codegen_.codegen_data_layout_convert import run_data_layout_convert_executable
from cim_compiler.utils.df_layout import tensor_bits_to_int8
import math
from functools import reduce
from polycim.config import set_raw_config_by_path, get_config
from polycim.op.calculate import depth_wise_conv2d
from polycim.exp.op_list import get_op_list
from polycim.exp.iccad25.exp_iccad25 import run_op
import pandas as pd

def ceil(a, b):
    return int(math.ceil(a / b))

@pytest.mark.parametrize("cim_cfg_path, op_id, cim_count", [
    ("g2r2c16b64.json", "conv2d_b1o8i1h8w8k3", 32),
    ("g2r2c16b64.json", "conv2d_b1o8i8h8w8k3", 256),
    ("g2r2c16b64.json", "conv2d_b1o16i8h8w8k3", 512),
    ("g2r2c16b64.json", "conv2d_b2o16i8h8w8k3", 1024),
    ("g2r2c16b64.json", "conv2d_b1o326i256h8w8k3s2", 512),

    ("g4r4c32b64.json", "conv2d_b1o8i1h8w8k3", 16),
    ("g4r4c32b64.json", "conv2d_b1o8i8h8w8k3", 48),
    ("g4r4c32b64.json", "conv2d_b1o16i8h8w8k3", 96),
    ("g4r4c32b64.json", "conv2d_b2o16i8h8w8k3", 192),
    ("g4r4c32b64.json", "conv2d_b1o326i256h8w8k3s2", 128),
])
def test_result(cim_cfg_path, op_id, cim_count):
    
    polycim_home = os.environ["POLYCIM_HOME"]
    compiler_cfg_path = os.path.join(polycim_home, "config/cimflow_test", cim_cfg_path)
    # pimsim_cfg_path = os.path.join(polycim_home, " ", cim_cfg_path)
    with tempfile.TemporaryDirectory() as temp_dir:
        cmd = [
            "polycim", "explore", 
            "--op-id", op_id, 
            "--config-path", compiler_cfg_path, 
            # "--pimsim-cfg-path", pimsim_cfg_path, 
            "--output-path", temp_dir, 
            "--data-movement-full-vectorize",
            "--cimflow",
            "--verify"
        ]
        subprocess.run(cmd, check=True)

        # get result from result.csv
        result_path = os.path.join(temp_dir, "result.csv")
        df = pd.read_csv(result_path)
        cim_compute_ops = df.at[0, 'cim_compute_ops']
        if cim_count != -1:
            assert cim_compute_ops == cim_count, f"{cim_compute_ops=} != {cim_count=}"

        check_result = bool(df.at[0, "check_result"])
        assert check_result

if __name__ == "__main__":
    test_result("g2r2c16b64.json", "conv2d_b1o16i8h8w8k3", 512)