# configs
export POLYCIM_HOME=$PWD
export CIMCOMPILER_HOME=/app/CIMCompiler/CIMCompiler
export PYTHONPATH=$PWD
source log_config.sh
# python3 experiment/performance_operator.py
# python3 experiment/sensitivity_analysis.py
# python3 experiment/performance_network.py
# exit
# extract graphs
# if [ ! -d "graphs" ]; then
#     echo "Extracting graphs.tar.gz..."
#     tar -xzvf graphs.tar.gz
# fi

# polycim op \
# --op-id 2_conv2d_b1o8i32h1w1k1k1s1d1 \
# --config-path /app/CIMCompiler/PolyCIM/polycim/exp/iccad25/compiler_configs/g8m8c32b64.json \
# --output-path ./exp_result/performance_network/05-12_07-52-06/EfficientNet_bs1_g8m8c32b64_im2col/output_2_conv2d_b1o8i32h1w1k1k1s1d1 \
# --data-movement-full-vectorize \
# --pimsim-cfg-path /app/CIMCompiler/PolyCIM/polycim/exp/iccad25/cimsim_configs/g8m8c32b64.json \
# --profiler-cfg-path /app/CIMCompiler/PolyCIM/polycim/exp/iccad25/profiler_config.json \
# --polycim \
# --unroll-level 1 \
# --profile \
# --profile-use-unrolled-code \
# --polycim-disable-pretile \
# --polycim-disable-affine


# python experiment/performance_network.py
# exit

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
# op_name=C15
# cur_time=$(date +%Y-%m-%d_%H-%M-%S)
# save_dir=".save/im2col_${cur_time}"
# polycim op \
# --op-id $op_name \
# --config-path $PWD/polycim/exp/iccad25/compiler_configs/c32b64.json \
# --pimsim-cfg-path $PWD/polycim/exp/iccad25/pimsim_configs/c32b64.json \
# --output-path $save_dir \
# --data-movement-full-vectorize \
# --polycim
# --verify

# --op-def-json exp_result/performance_network_unrolled/mobilenet_v2_bs1_c32b64/op_defs_mobilenet_v2.json
# --profile

# --profile-use-unrolled-code

# run op (polycim)
op_name=dw5x5
cur_time=$(date +%Y-%m-%d_%H-%M-%S)
save_dir=".save/${cur_time}"
polycim op \
--op-id $op_name \
--config-path $PWD/polycim/exp/iccad25/compiler_configs/g8m8c32b64.json \
--pimsim-cfg-path $PWD/polycim/exp/iccad25/cimsim_configs/g8m8c32b64.json \
--profiler-cfg-path $PWD/polycim/exp/iccad25/profiler_config.json \
--output-path $save_dir \
--data-movement-full-vectorize \
--polycim \
--unroll-level 1 \
--profile \
--verify
# --polycim-disable-pretile \
# --polycim-disable-affine

# --profile \
# --op-def-json exp_result/performance/convnext_tiny_im2col_g8m8c32b64/op_defs_convnext_tiny.json
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

