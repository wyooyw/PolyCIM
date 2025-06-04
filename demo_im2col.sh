export POLYCIM_HOME=$PWD
export CIMCOMPILER_HOME=/app/CIMCompiler/CIMCompiler
export PYTHONPATH=$PWD
source log_config.sh

op_name=new_C1
cur_time=$(date +%Y-%m-%d_%H-%M-%S)
save_dir=".save/${cur_time}"
polycim op \
--op-id $op_name \
--config-path $PWD/polycim/exp/iccad25/compiler_configs/g8m8c32b64.json \
--pimsim-cfg-path $PWD/polycim/exp/iccad25/cimsim_configs/g8m8c32b64.json \
--profiler-cfg-path $PWD/polycim/exp/iccad25/profiler_config.json \
--output-path $save_dir \
--data-movement-full-vectorize \
--polycim-disable-affine \
--polycim-disable-pretile \
--polycim \
--unroll-level 1 \
--profile