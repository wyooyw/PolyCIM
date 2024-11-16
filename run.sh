export POLYCIM_COMPILER_HOME=/home/yikun/pim-compiler/PolyCIM
export BACKEND_COMPILER_HOME=/home/yikun/pim-compiler/CIMCompiler
export CONFIG_PATH=$POLYCIM_COMPILER_HOME/config/config_gs_16.json

export READ_JSON=./instructions/partition_2_resnet_instructions.json
export SAVE_DIR=./.temp_save

export RUN_BEHAVIOR_SIMULATOR_FOR_CONV=1

python3 network_pipeline.py \
--read-json $READ_JSON \
--save-dir $SAVE_DIR