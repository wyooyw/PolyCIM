import subprocess
import tempfile
import os
import json
import pytest
import numpy as np
from polycim.codegen_.codegen_data_layout_convert import run_data_layout_convert_executable
from cim_compiler.utils.df_layout import tensor_bits_to_int8
import math

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
    *[("configs/c32b64.json", op_id, cim_count)
        for op_id, cim_count in [
            ("C1", 1568), ("C2", 392),
        ]
    ],
])
def test_result(cim_cfg_path, op_id, cim_count):
    cim_cfg_path = os.path.join(os.path.dirname(__file__), cim_cfg_path)
    with tempfile.TemporaryDirectory() as temp_dir:
        subprocess.run([
            "polycim", "explore",
            "--op-id", op_id,
            "--config-path", cim_cfg_path,
            "--output-path", temp_dir
        ], check=True)
        # import pdb; pdb.set_trace()
        # exit()
    
        # get stats.json
        # stats_path = os.path.join(temp_dir, op_id, "0", "output", "stats.json")
        # assert os.path.exists(stats_path), f"{stats_path=}"
        # with open(stats_path, "r") as f:
        #     stats = json.load(f)
        # assert stats["CIMComputeInst"] == cim_count, f"{stats['CIMComputeInst']=}, {cim_count=}"

        # prepare data
        op_dir = os.path.join(temp_dir, op_id, "0")
        data_dir = os.path.join(op_dir, "data")
        os.makedirs(data_dir, exist_ok=True)

        n_group_vcol = 4 # cim_cfg.n_group_vcol
        mask_num = int(math.ceil(n_group_vcol / 8)) * 8
        cim_mask = np.array([1] * mask_num).reshape(-1, 8)
        cim_mask_np = tensor_bits_to_int8(cim_mask)
        cim_mask_data = bytearray(cim_mask_np)
        
        input_np = np.arange(0,16, dtype=np.int8).reshape(4,4)
        input_np = np.pad(input_np, ((1,1),(1,1)), mode="constant", constant_values=0)
        assert input_np.shape == (6,6)
        # input_np = np.ones((58,58), dtype=np.int32)
        weight_np = np.array([[1,0,-1],[1,-1,0],[1,1,-1]], dtype=np.int8)

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

        # get the output
        output_path = os.path.join(sim_output_dir, "image.bin")
        assert os.path.exists(output_path), f"{output_path=}"
        with open(output_path, "rb") as f:
            output_global_image = f.read()
        
        output_offset = len(total_data)
        # output_size = 28 * 28 * 4
        output_size = 2 * 2 * 4
        output_data = output_global_image[output_offset:output_offset+output_size]
        output_np = np.frombuffer(output_data, dtype=np.int8)
        print(f"{output_np=}")
        # np.savetxt(os.path.join(sim_output_dir, "output.txt"), output_np.reshape(-1), fmt="%d")
        
        O_exe_path = os.path.join(op_dir, "convert_O.o")
        O_data_path = os.path.join(data_dir, "O.txt")
        O_converted_data_path = os.path.join(data_dir, "O_converted.txt")
        save_and_convert(O_exe_path, O_data_path, O_converted_data_path, output_np)
        
        # read O_converted
        O_converted_np = np.loadtxt(O_converted_data_path, dtype=np.int8).reshape(4,4)
        print(f"input_np: \n{input_np}")
        print(f"weight_np: \n{weight_np}")
        print(f"O_converted_np: \n{O_converted_np}")
        golden = np.array([[ -1,   2,   3,  12],
                           [ -6,   4,   5,  22],
                           [-14,   8,   9,  34],
                           [-21,  -3,  -3,   9]], dtype=np.int8)
        assert np.all(O_converted_np==golden), f"{O_converted_np=}, {golden=}"
        
if __name__ == "__main__":
    test_result("configs/c16b32.json", "test", 4)