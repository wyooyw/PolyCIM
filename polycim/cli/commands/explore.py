import argparse
import glob
import os
import shutil
import subprocess
import tempfile
import json

from polycim.config import get_config
from polycim.depth_first.pipeline import run_op_list

from polycim.utils.logger import get_logger
from polycim.cli.common import show_args, to_abs_path
from polycim.config import set_raw_config_by_path

logger = get_logger(__name__)

def parse_explore_args(subparsers):
    parser = subparsers.add_parser('explore')
    parser.add_argument("--op-id", "-i", required=True, type=str, help="operator id")
    parser.add_argument("--config-path", "-c", required=True, type=str, help="config path")
    parser.add_argument("--output-path", "-o", required=True, type=str, help="output path")
    parser.add_argument("--data-movement-full-vectorize", action="store_true", help="data movement full vectorize")
    parser.add_argument("--force-axis-align", action="store_true", help="force axis aligned")

def run_explore(args):
    args.output_path = to_abs_path(args.output_path)
    args.config_path = to_abs_path(args.config_path)
    set_raw_config_by_path(args.config_path)

    logger.info("Begin to explore operator.")
    logger.info(show_args(args))

    cim_cfg = get_config()

    pad_count = True
    delay_apply = True
    num_macros = cim_cfg.n_macro
    enable_weight_rewrite = True

    from polycim.exp.op_list import get_op_list
    op_list = get_op_list()
    op_list = {args.op_id: op_list[args.op_id]}

    # curr_time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # curr_time_str = curr_time_str + f"_{cim_cfg.n_comp}x{cim_cfg.n_group_vcol*8}"
    save_dir = os.path.join(args.output_path)
    run_op_list(
        op_list, 
        save_dir, 
        pad_count=pad_count, 
        delay_apply=delay_apply, 
        num_macros=num_macros, 
        enable_weight_rewrite=enable_weight_rewrite,
        force_axis_align=args.force_axis_align,
        cim_config=cim_cfg
    )
