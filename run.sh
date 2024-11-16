export POLYCIM_COMPILER_HOME=/home/yikun/pim-compiler/PolyCIM
export BACKEND_COMPILER_HOME=/home/yikun/pim-compiler/CIMCompiler
export RUN_BEHAVIOR_SIMULATOR_FOR_CONV=1
export CONFIG_PATH=/home/yikun/pim-compiler/PolyCIM/config/config_gs_16.json

python3 network_pipeline.py \
--read-json instructions/partition_2_resnet_instructions.json \
--save-dir ./.temp_save