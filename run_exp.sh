export CONFIG_PATH=$PWD/config/config.json
export POLYCIM_HOME=$PWD
# export MAX_PROCESS_USE=1
export PYTHONPATH=$PWD
# export LOG_LEVEL="DEBUG"
source log_config.sh

# pytest test/schedule/test_coalescing.py
# pytest -n 4 test/explore/test_cimflow_result.py
# pytest -n 4  test/explore/test_polycim_result.py

cur_time=$(date +%Y-%m-%d_%H-%M-%S)
save_dir=".save/${cur_time}"
polycim cimflow_network \
-i graphs/instructions_mobilenet_0.5x_load_time_T4_B8.json \
-o $save_dir \
-c $PWD/config/dac25/config_gs_4.json

# op_name=conv2d_b2o16i8h8w8k3
# cur_time=$(date +%Y-%m-%d_%H-%M-%S)
# save_dir=".save/${cur_time}"
# # --config-path $PWD/config/dac25/c32b64.json \
# polycim explore \
# --op-id $op_name \
# --config-path $PWD/config/dac25/g4r4c32b64.json \
# --pimsim-cfg-path $PWD/polycim/exp/iccad25/pimsim_configs/c32b64.json \
# --output-path $save_dir \
# --data-movement-full-vectorize \
# --cimflow
# --disable-affine

# --disable-second-stage
# --disable-pretile \
# --disable-affine
# --disable-pretile \

# pytest -n 4 test/explore/test_result.py
# python test/explore/test_result.py


# cim-compiler convert \
# --dst-type legacy \
# --src-file .save/$op_name/0/final_code.json \
# --dst-file .save/$op_name/0/final_code.legacy.json \
# --filter-out-invalid-instructions \
# --add-single-core-id

# cp .save/$op_name/0/final_code.legacy.json \
# /home/wangyiou/project/github-pim-sim/pim-sim/test_data/polycim/${op_name}_align.json

# cim-compiler convert \
# --src-type cimflow \
# --dst-type legacy \
# --src-file .save/C1/0/final_code.json \
# --dst-file .save/C1/0/final_code_legacy.json \
# --filter-out-cim-output-and-cim-transfer

# pytest -n 4 test/explore/test_result.py
# python test/explore/test_result.py

# cim-compiler simulate \
# --code-file .save/C4/0/final_code.legacy2.json \
# --config-file test/explore/configs/c32b64.json \
# --output-dir .save/C4/0/sim_output_legacy \
# --code-format legacy \
# --save-stats
# --save-unrolled-code