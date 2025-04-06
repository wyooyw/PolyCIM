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

@pytest.mark.parametrize("graph_instruction_path, config_path", [
    # resnet18
    ("graphs/instructions_resnet18_0.5x_load_time_T4_B8.json", "config/dac25/config_gs_4.json"),
    ("graphs/instructions_resnet18_0.5x_load_time_T8_B8.json", "config/dac25/config_gs_8.json"),
    ("graphs/instructions_resnet18_0.5x_load_time_T12_B8.json", "config/dac25/config_gs_12.json"),
    ("graphs/instructions_resnet18_0.5x_load_time_T16_B8.json", "config/dac25/config_gs_16.json"),

    # mobilenet
    ("graphs/instructions_mobilenet_0.5x_load_time_T4_B8.json", "config/dac25/config_gs_4.json"),
    ("graphs/instructions_mobilenet_0.5x_load_time_T8_B8.json", "config/dac25/config_gs_8.json"),
    ("graphs/instructions_mobilenet_0.5x_load_time_T12_B8.json", "config/dac25/config_gs_12.json"),
    ("graphs/instructions_mobilenet_0.5x_load_time_T16_B8.json", "config/dac25/config_gs_16.json"),
])
def test_cimflow_network(graph_instruction_path, config_path):
    POLYCIM_HOME = os.environ.get("POLYCIM_HOME")
    graph_instruction_path = os.path.join(POLYCIM_HOME, graph_instruction_path)
    config_path = os.path.join(POLYCIM_HOME, config_path)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        cmd = [
            "polycim", "cimflow_network", 
            "-i", graph_instruction_path,
            "-o", temp_dir,
            "-c", config_path
        ]
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    # test_result("g2r2c16b64.json", "conv2d_b1o16i8h8w8k3", 512)
    pass