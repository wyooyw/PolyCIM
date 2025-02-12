export CONFIG_PATH=$PWD/config/config.json
export MAX_PROCESS_USE=1
export PYTHONPATH=$PWD
# export LOG_LEVEL="DEBUG"
source log_config.sh

# python -u polycim/cli/main.py explore \
# --op-id test \
# --config-path $PWD/test/explore/configs/g2m2c16b32.json \
# --output-path .save
# python test/explore/test_result.py
python test/explore/test_result.py