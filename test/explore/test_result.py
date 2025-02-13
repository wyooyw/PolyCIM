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

def get_verify_fn(op_id):
    op_info = get_op_list()[op_id]
    return op_info["verify_fn"]

def ceil(a, b):
    return int(math.ceil(a / b))

def save_and_convert(exe_path, data_path, converted_data_path, data_np):
    with open(data_path, "w") as f:
        np.savetxt(f, data_np.reshape(-1), fmt="%d")

    run_data_layout_convert_executable(
        exe_path,
        data_path,
        converted_data_path
    )

@pytest.mark.parametrize("cim_cfg_path, op_id, cim_count", [
    *[("configs/c16b32.json", op_id, cim_count)
        for op_id, cim_count in [
            ("C1", 3136), ("C2", 784),
        ]
    ],
    *[("configs/g2m2c16b32.json", op_id, cim_count)
        for op_id, cim_count in [
            ("C1", 3136 // 2), ("C2", 784 // 2),
        ]
    ],
    *[("configs/g4m4c16b32.json", op_id, cim_count)
        for op_id, cim_count in [
            ("C1", 3136 // 4), ("C2", 784 // 4),
        ]
    ],
    *[("configs/c32b64.json", op_id, cim_count)
        for op_id, cim_count in [
            ("C1", 1568), ("C2", 392), ("C4", 28), ("C7", 7), ("C12", 98)
        ]
    ],
    *[("configs/g2m2c32b64.json", op_id, cim_count)
        for op_id, cim_count in [
            ("C1", 1568 // 2), ("C2", 392 // 2), ("C4", 28 // 2), ("C7", ceil(7, 2)), ("C12", 98 // 2) 
        ]
    ],
    *[("configs/g4m4c32b64.json", op_id, cim_count)
        for op_id, cim_count in [
            ("C1", ceil(1568, 4)), ("C2", ceil(392, 4)), ("C4", ceil(28, 4)), ("C7", ceil(7, 4)), ("C12", -1) 
        ]
    ],
])
def test_result(cim_cfg_path, op_id, cim_count):
    cim_cfg_path = os.path.join(os.path.dirname(__file__), cim_cfg_path)
    set_raw_config_by_path(cim_cfg_path)

    with tempfile.TemporaryDirectory() as temp_dir:

        # temp_dir = ".temp"
        # os.makedirs(temp_dir, exist_ok=True)
        subprocess.run([
            "polycim", "explore",
            "--op-id", op_id,
            "--config-path", cim_cfg_path,
            "--output-path", temp_dir
        ], check=True)
        
        op_dir = os.path.join(temp_dir, op_id, "0")
        data_dir = os.path.join(op_dir, "data")
        os.makedirs(data_dir, exist_ok=True)

        # get buffer info
        buffer_info_path = os.path.join(op_dir, "buffer_info.json")
        with open(buffer_info_path, "r") as f:
            buffer_info = json.load(f)

        # get origin operand shape
        origin_operand_shape_path = os.path.join(op_dir, "origin_operand_shape.json")
        with open(origin_operand_shape_path, "r") as f:
            origin_operand_shape = json.load(f)

        # prepare data
        cim_mask_shape = buffer_info["cim_mask_global"]["shape"]
        cim_mask_np = np.ones(cim_mask_shape, dtype=np.int8).reshape(-1, 8)
        cim_mask_np = tensor_bits_to_int8(cim_mask_np)
        cim_mask_data = bytearray(cim_mask_np)
        
        input_shape = origin_operand_shape["I"]
        input_np = np.random.randint(-2, 3, size=input_shape, dtype=np.int8)
        # input_np = np.ones(input_shape, dtype=np.int8)

        weight_shape = origin_operand_shape["W"]
        weight_np = np.random.randint(-2, 3, size=weight_shape, dtype=np.int8)
        # weight_np = np.ones(weight_shape, dtype=np.int8)

        # convert data
        I_exe_path = os.path.join(op_dir, "convert_I.o")
        I_data_path = os.path.join(data_dir, "I.txt")
        I_converted_data_path = os.path.join(data_dir, "I_converted.txt")
        save_and_convert(I_exe_path, I_data_path, I_converted_data_path, input_np)

        W_exe_path = os.path.join(op_dir, "convert_W.o")
        W_data_path = os.path.join(data_dir, "W.txt")
        W_converted_data_path = os.path.join(data_dir, "W_converted.txt")
        save_and_convert(W_exe_path, W_data_path, W_converted_data_path, weight_np)
        
        
        # concat input and weight into global memory image
        I_converted_np = np.loadtxt(I_converted_data_path, dtype=np.int8)
        W_converted_np = np.loadtxt(W_converted_data_path, dtype=np.int8)
        I_converted_data = bytearray(I_converted_np)
        W_converted_data = bytearray(W_converted_np)
        global_memory_image_path = os.path.join(data_dir, "global_memory_image.bin")
        total_data = cim_mask_data + I_converted_data + W_converted_data
        with open(global_memory_image_path, "wb") as f:
            f.write(total_data)

        # run the simulator
        code_path = os.path.join(op_dir, "final_code.json")
        sim_output_dir = os.path.join(op_dir, "sim_output")
        subprocess.run([
            "cim-compiler", "simulate",
            "--code-file", code_path,
            "--data-file", global_memory_image_path,
            "--config-file", cim_cfg_path,
            "--output-dir", sim_output_dir,
            "--code-format", "cimflow",
            "--save-stats"
        ], check=True)

        # check stats
        stats_path = os.path.join(sim_output_dir, "stats.json")
        assert os.path.exists(stats_path), f"{stats_path=}"
        with open(stats_path, "r") as f:
            stats = json.load(f)
        if cim_count >= 0:
            assert stats["CIMComputeInst"] == cim_count, f"{stats['CIMComputeInst']=}, {cim_count=}"
        else:
            print("skip stats check")

        # get the output
        output_path = os.path.join(sim_output_dir, "image.bin")
        assert os.path.exists(output_path), f"{output_path=}"
        with open(output_path, "rb") as f:
            output_global_image = f.read()
        
        output_offset = len(total_data)
        # output_size = 28 * 28 * 4
        output_size = reduce(lambda x, y: x * y, buffer_info["O_aligned"]["shape"])
        output_data = output_global_image[output_offset:output_offset+output_size]
        output_np = np.frombuffer(output_data, dtype=np.int8)
        print(f"{output_np=}")
        # np.savetxt(os.path.join(sim_output_dir, "output.txt"), output_np.reshape(-1), fmt="%d")
        
        O_exe_path = os.path.join(op_dir, "convert_O.o")
        O_data_path = os.path.join(data_dir, "O.txt")
        O_converted_data_path = os.path.join(data_dir, "O_converted.txt")
        save_and_convert(O_exe_path, O_data_path, O_converted_data_path, output_np)
        
        # read O_converted
        output_shape = origin_operand_shape["O"]
        O_converted_np = np.loadtxt(O_converted_data_path, dtype=np.int8).reshape(output_shape)
        print(f"output shape={O_converted_np.shape}, dtype={O_converted_np.dtype}, max={O_converted_np.max()}, min={O_converted_np.min()}")

        verify_fn = get_verify_fn(op_id)
        golden = verify_fn(input_np, weight_np)
        print(f"golden shape={golden.shape}, dtype={golden.dtype}, max={golden.max()}, min={golden.min()}")
        print(f"{temp_dir = } (will be removed after test)")
        # import pdb; pdb.set_trace()
        assert np.all(O_converted_np==golden), f"{O_converted_np=}, {golden=}"

        
if __name__ == "__main__":
    # test_result("configs/c32b64.json", "C2", 392)
    # test_result("configs/c32b64.json", "C4", 28)
    # test_result("configs/c32b64.json", "C7", 7)
    # test_result("configs/c32b64.json", "C12", 98)

    # test_result("configs/g2m2c32b64.json", "C2", 196)
    # test_result("configs/g2m2c32b64.json", "C4", 14)
    # test_result("configs/g2m2c32b64.json", "C7", 4)
    test_result("configs/g4m4c32b64.json", "C12", -1)
    # test_result("configs/g2m2c32b64.json", "C12", 49)
    
    # test_result("configs/c32b64.json", "d2h4", -1)
    # pass