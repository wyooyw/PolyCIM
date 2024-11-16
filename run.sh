export BACKEND_COMPILER_HOME=/home/wangyiou/project/cim_compiler_frontend/playground
export RUN_BEHAVIOR_SIMULATOR_FOR_CONV=1

python network_pipeline.py \
--read-json instructions/partition_2_extracted_model_instructions.json \
--save-dir ./.temp_save