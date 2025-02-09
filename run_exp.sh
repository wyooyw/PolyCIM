export CONFIG_PATH=$PWD/config/config.json
export MAX_PROCESS_USE=1
export PYTHONPATH=$PWD
# export NEW_ALGO=1
# python wyo_network_pipeline.py
# python pipeline.py
# pytest test/use_rate_test.py
# python experiment_dac25/operator_exp.py --model EfficientNet
# python experiment_dac25/network_exp.py
# python pipeline.py
# pytest test/test_multi_level_tiling.py
python -u polycim/depth_first/pipeline.py
# python .save/2025-01-14_14-58-29_32x64/C3/solution_2/schedule_code.py
# python -u pipeline.py
# pytest test/test_multi_level_tiling.py
# python -u models/onnx_parser.py