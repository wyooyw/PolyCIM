export CONFIG_PATH=$PWD/config/config.json
export MAX_PROCESS_USE=1
export PYTHONPATH=$PWD
export LOG_LEVEL="DEBUG"

# python -u polycim/cli/main.py explore \
# --op-id C2 \
# --config-path $PWD/test/explore/configs/c16b32.json \
# --output-path .save
python test/explore/test_result.py
# python .save/C2/solution_0/schedule_code.py