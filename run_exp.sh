export CONFIG_PATH=$PWD/config/config.json
# export MAX_PROCESS_USE=1
export PYTHONPATH=$PWD
# export LOG_LEVEL="DEBUG"
source log_config.sh

python -u polycim/cli/main.py explore \
--op-id C3 \
--config-path $PWD/test/explore/configs/c32b64.json \
--output-path .save
# pytest -n 4 test/explore/test_result.py
# python test/explore/test_result.py