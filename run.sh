export BACKEND_COMPILER_HOME=/home/wangyiou/project/cim_compiler_frontend/playground
export RUN_BEHAVIOR_SIMULATOR_FOR_CONV=1
export LOG_LEVEL="INFO"

python network_pipeline.py \
--read-json instructions/partition_0_resnet_instructions.json \
--save-dir ./.temp_save