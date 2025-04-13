# configs
export POLYCIM_HOME=$PWD
export CIMCOMPILER_HOME=/home/wangyiou/project/CIMCompiler
export PYTHONPATH=$PWD
source log_config.sh

# extract graphs
if [ ! -d "graphs" ]; then
    echo "Extracting graphs.tar.gz..."
    tar -xzvf graphs.tar.gz
fi

python experiment/performance.py
exit

# python experience/sensitivity_analysis.py

# run test
# pytest -n 4 test
# pytest -n 4 test/end2end/test_polycim_op.py
# python test/end2end/test_polycim_op.py
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
# polycim op \
# --op-id $op_name \
# --config-path $PWD/config/cimflow_test/g4r4c32b64.json \
# --pimsim-cfg-path $PWD/polycim/exp/iccad25/pimsim_configs/c32b64.json \
# --output-path $save_dir \
# --data-movement-full-vectorize \
# --cimflow \
# --verify

# run op (polycim)
op_name=C15
cur_time=$(date +%Y-%m-%d_%H-%M-%S)
save_dir=".save/${cur_time}"
polycim op \
--op-id $op_name \
--config-path $PWD/polycim/exp/iccad25/compiler_configs/c32b64.json \
--pimsim-cfg-path $PWD/polycim/exp/iccad25/pimsim_configs/c32b64.json \
--output-path $save_dir \
--data-movement-full-vectorize \
--polycim \
--unroll-level 1
# --polycim-disable-affine-prune
# --op-def-json $PWD/op_def.json
# --polycim-disable-affine-prune
# --polycim-disable-pretile-prune

# --verify \
# --profile \
# --polycim-disable-pretile \
# --polycim-disable-affine
# --polycim-disable-affine

# --disable-hardware-mapping-coalescing
# --data-movement-search \
# --data-movement-search-time 30

# --data-movement-search \
# --data-movement-search-time 120
# --data-movement-solver

# --pimsim-cfg-path $PWD/polycim/exp/iccad25/pimsim_configs/c32b64.json \

# --polycim-disable-affine

