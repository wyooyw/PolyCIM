# configs
export POLYCIM_HOME=$PWD
export CIMCOMPILER_HOME=/home/wangyiou/project/CIMCompiler
export PYTHONPATH=$PWD
source log_config.sh

# run test
pytest -n 4 test

# run network
# cur_time=$(date +%Y-%m-%d_%H-%M-%S)
# save_dir=".save/${cur_time}"
# polycim cimflow_network \
# -i graphs/instructions_mobilenet_0.5x_load_time_T4_B8.json \
# -o $save_dir \
# -c $PWD/config/dac25/config_gs_4.json

# run op (cimflow)
# op_name=conv2d_b2o16i8h8w8k3
# cur_time=$(date +%Y-%m-%d_%H-%M-%S)
# save_dir=".save/${cur_time}"
# polycim explore \
# --op-id $op_name \
# --config-path $PWD/config/cimflow_test/g4r4c32b64.json \
# --pimsim-cfg-path $PWD/polycim/exp/iccad25/pimsim_configs/c32b64.json \
# --output-path $save_dir \
# --data-movement-full-vectorize \
# --cimflow \
# --verify

# run op (polycim)
# op_name=C1
# cur_time=$(date +%Y-%m-%d_%H-%M-%S)
# save_dir=".save/${cur_time}"
# polycim explore \
# --op-id $op_name \
# --config-path $PWD/polycim/exp/iccad25/compiler_configs/c32b64.json \
# --pimsim-cfg-path $PWD/polycim/exp/iccad25/pimsim_configs/c32b64.json \
# --output-path $save_dir \
# --data-movement-full-vectorize \
# --polycim \
# --verify
