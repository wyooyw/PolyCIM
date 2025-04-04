export CONFIG_PATH=$PWD/config/config.json
# export MAX_PROCESS_USE=1
export PYTHONPATH=$PWD
# export LOG_LEVEL="DEBUG"
source log_config.sh

op_name=C2
cur_time=$(date +%Y-%m-%d_%H-%M-%S)
save_dir=".save/${cur_time}"

polycim explore \
--op-id $op_name \
--config-path $PWD/polycim/exp/iccad25/compiler_configs/c32b64.json \
--pimsim-cfg-path $PWD/polycim/exp/iccad25/pimsim_configs/c32b64.json \
--output-path $save_dir \
--data-movement-full-vectorize
# --disable-second-stage
# --disable-pretile \
# --disable-affine
# --disable-pretile \

# pytest -n 4 test/explore/test_result.py
# python test/explore/test_result.py


# cim-compiler convert \
# --dst-type legacy \
# --src-file .save/$op_name/0/final_code.json \
# --dst-file .save/$op_name/0/final_code.legacy.json \
# --filter-out-invalid-instructions \
# --add-single-core-id

# cp .save/$op_name/0/final_code.legacy.json \
# /home/wangyiou/project/github-pim-sim/pim-sim/test_data/polycim/${op_name}_align.json

# cim-compiler convert \
# --src-type cimflow \
# --dst-type legacy \
# --src-file .save/C1/0/final_code.json \
# --dst-file .save/C1/0/final_code_legacy.json \
# --filter-out-cim-output-and-cim-transfer

# pytest -n 4 test/explore/test_result.py
# python test/explore/test_result.py

# cim-compiler simulate \
# --code-file .save/C4/0/final_code.legacy2.json \
# --config-file test/explore/configs/c32b64.json \
# --output-dir .save/C4/0/sim_output_legacy \
# --code-format legacy \
# --save-stats
# --save-unrolled-code