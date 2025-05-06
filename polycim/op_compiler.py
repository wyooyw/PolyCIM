import os
from dataclasses import dataclass

from polycim.passes import (AffinePass, BackendCompilePass, BufferMappingPass,
                            CodegenPass, DumpOpPass, FilterSingleOpPass,
                            HardwareMapping5DPass, HardwareMappingPass,
                            MappingMultiMacroPass, PreTilingPass, ProfilePass,
                            TensorizePass, UtilizationEvaluatePass, VerifyPass)
from polycim.passes.base import PassManager
from polycim.utils.logger import get_logger
from dataclasses import dataclass, asdict


logger = get_logger(__name__)


def parse_op_list(op_list):
    name_and_cfg = list(op_list.items())
    assert len(name_and_cfg) == 1
    name, config = name_and_cfg[0]

    op = config["op"]
    op.attr["origin_op"] = op
    op.attr["symmetry_info"] = config["symmetry_info"]
    op.attr["dim_types"] = config["dim_types"]
    op.attr["name"] = name
    if "max_tiling_level" in config:
        op.attr["max_tiling_level"] = config["max_tiling_level"]
    if "not_tiling" in config:
        op.attr["not_tiling"] = config["not_tiling"]
    return op


@dataclass
class Column:
    name: str
    attr_keys: list[str] = None  # use op.attr[key1][key2]... to get the value
    constant_value: str = None


def save_table(op_list, columns, output_path, format="csv"):
    """
    将操作列表中的属性保存为表格文件

    Args:
        op_list: list of op - 操作列表
        columns: list of Column - 要保存的列定义
        output_path: str - 输出文件路径
        format: str - 输出格式，支持 "csv" 或 "xlsx"
    """
    # 准备表格数据
    header = [col.name for col in columns]
    rows = []
    for op in op_list:
        row = []
        for col in columns:
            if col.constant_value:
                value = col.constant_value
            elif col.attr_keys:
                # 通过属性键列表逐层获取属性值
                value = op.attr
                for key in col.attr_keys:
                    try:
                        value = value[key]
                    except (KeyError, TypeError):
                        value = None
                        break
            else:
                import pdb; pdb.set_trace()
                assert False
            row.append(value)
        rows.append(row)

    if format == "csv":
        import csv

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)

    elif format == "xlsx":
        import pandas as pd

        df = pd.DataFrame(rows, columns=header)
        df.to_excel(output_path, index=False)

    else:
        raise ValueError(f"Unsupported format: {format}")

    logger.info(f"table saved to {output_path}")


def pretiling_influence_utilization(args, cim_config, op):
    pass_manager = PassManager(
        [
            PreTilingPass(args, schedule_as_key=True),
            AffinePass(args),
            HardwareMappingPass(args, cim_config),
            UtilizationEvaluatePass(args, cim_config, keep_single_best=True),
        ]
    )
    result = pass_manager.apply(op)
    for op in result:
        pre_tile_sizes = op.attr["pre_tile_sizes"]
        tile_size_1 = pre_tile_sizes[2][1] if len(pre_tile_sizes[2]) == 2 else -1
        tile_size_2 = pre_tile_sizes[3][1] if len(pre_tile_sizes[3]) == 2 else -1
        op.attr["pre_tile_sizes_pretty"] = (tile_size_1, tile_size_2)

    sorted_result = sorted(
        result, key=lambda x: x.attr["UtilizationEvaluatePass"]["utilization"]
    )

    save_table(
        sorted_result,
        [
            Column(name="name", attr_keys=["name"]),
            Column(name="pre_tile_sizes", attr_keys=["pre_tile_sizes"]),
            Column(name="pre_tile_sizes_pretty", attr_keys=["pre_tile_sizes_pretty"]),
            Column(
                name="compute_ops", attr_keys=["UtilizationEvaluatePass", "compute_ops"]
            ),
            Column(
                name="utilization", attr_keys=["UtilizationEvaluatePass", "utilization"]
            ),
        ],
        "pretiling_influence_utilization.xlsx",
        format="xlsx",
    )
    pass_manager.show_time_per_pass()
    logger.info(f"num_ops: {pass_manager.get_num_ops()}")


def utilization_influence_latency(args, cim_config, op, max_keep=32):
    pass_manager = PassManager(
        [
            PreTilingPass(args, schedule_as_key=True),
            AffinePass(args, schedule_as_key=True),
            HardwareMappingPass(args, cim_config),
            UtilizationEvaluatePass(args, cim_config),
            FilterSingleOpPass(n_keep=max_keep),
            # DumpOpPass(args, cim_config),
            MappingMultiMacroPass(args, cim_config),
            BufferMappingPass(args, cim_config),
            TensorizePass(args, cim_config),
            CodegenPass(args, cim_config),
            BackendCompilePass(
                args, cim_config, n_workers=4, compile_data_layout=False
            ),
            ProfilePass(args),
        ]
    )
    result = pass_manager.apply(op)
    save_table(
        result,
        [
            Column(name="name", attr_keys=["name"]),
            Column(name="pre_tile_sizes", attr_keys=["pre_tile_sizes"]),
            Column(name="affine schedule", attr_keys=["AffinePass", "schedule"]),
            Column(
                name="utilization", attr_keys=["UtilizationEvaluatePass", "utilization"]
            ),
            Column(name="latency", attr_keys=["ProfilePass", "latency"]),
        ],
        f"utilization_influence_latency_{max_keep}.xlsx",
        format="xlsx",
    )
    pass_manager.show_time_per_pass()
    logger.info(f"num_ops: {pass_manager.get_num_ops()}")


def pruning_search_space(args, cim_config, op):
    pass_manager_pruning = PassManager(
        [
            PreTilingPass(args),
            AffinePass(args),
        ]
    )
    result = pass_manager_pruning.apply(op)
    n_op_pruning = len(result)
    print(f"n_op_pruning: {n_op_pruning}")
    import pdb

    pdb.set_trace()

    pass_manager_full = PassManager(
        [
            PreTilingPass(args, prune=False),
            AffinePass(args, prune=False),
        ]
    )
    result = pass_manager_full.apply(op)
    n_op_full = len(result)
    print(f"n_op_full: {n_op_full}")


def run_polycim(args, cim_config, op, max_keep=32):
    pass_list = [
        PreTilingPass(args, schedule_as_key=False),
        AffinePass(args, schedule_as_key=False),
        HardwareMappingPass(args, cim_config),
        UtilizationEvaluatePass(args, cim_config),
        FilterSingleOpPass(n_keep=1),
        DumpOpPass(args, cim_config),
        MappingMultiMacroPass(args, cim_config),
    ]
    if args.stage2:
        pass_list.extend([
            BufferMappingPass(args, cim_config),
            TensorizePass(args, cim_config),
            CodegenPass(args, cim_config, unroll_level=args.unroll_level)
        ])
    if args.backend_compile:
        pass_list.append(BackendCompilePass(args, cim_config, n_workers=4, compile_data_layout=True))
    if args.verify:
        assert args.backend_compile
        pass_list.append(VerifyPass(args))
    if args.profile:
        assert args.backend_compile
        pass_list.append(ProfilePass(args))

    pass_manager = PassManager(pass_list)
    result = pass_manager.apply(op)
    # for op in result:
    #     op.attr["unroll_level"] = args.unroll_level
    show_columns = [
        Column(name="name", attr_keys=["name"]),
        Column(name="pre_tile_sizes", attr_keys=["pre_tile_sizes"]),
        Column(name="affine schedule", attr_keys=["AffinePass", "schedule"]),
        Column(name="h2s mapping", attr_keys=["HardwareMappingPass", "h2s_mapping"]),
        Column(name="coalescing schedule", attr_keys=["HardwareMappingPass", "coalescing_schedule"]),
        Column(name="post-tiling schedule", attr_keys=["HardwareMappingPass", "tiling_schedule"]),
        Column(
            name="utilization", attr_keys=["UtilizationEvaluatePass", "utilization"]
        ),
        Column(
            name="compute_ops", attr_keys=["UtilizationEvaluatePass", "compute_ops"]
        ),
        Column(
            name="flops", attr_keys=["UtilizationEvaluatePass", "flops"]
        ),
        Column(name="check_result", attr_keys=["VerifyPass", "check_result"]),
        Column(
            name="cim_compute_ops",
            attr_keys=["VerifyPass", "inst_stats", "CIMComputeInst"],
        ),
        Column(name="latency", attr_keys=["ProfilePass", "latency"]),
        Column(name="data_movement_cost_value", attr_keys=["BufferMappingPass", "cost"]),
        Column(name="data_movement_search_time", attr_keys=["BufferMappingPass", "time"]),
    ]
    
    cim_config_dict = asdict(cim_config)
    keys = sorted(list(cim_config_dict.keys()))
    for key in keys:
        show_columns.append(
            Column(name=f"cim_config.{key}", constant_value=str(cim_config_dict[key]))
        )

    args_config_dict = vars(args)
    keys = sorted(list(args_config_dict.keys()))
    for key in keys:
        show_columns.append(
            Column(name=f"args.{key}", constant_value=str(args_config_dict[key])),
        )

    time_per_pass = pass_manager.get_time_per_pass(sort_by_time=False)
    total_time = sum([t for p,t in time_per_pass])
    for pass_name, time in time_per_pass:
        show_columns.append(
            Column(name=f"pass_manager.time.{pass_name}", constant_value=str(time)),
        )
    show_columns.append(
        Column(name="pass_manager.total_time", constant_value=str(total_time)),
    )

    save_table(
        result,
        show_columns,
        os.path.join(args.output_path, "result.csv"),
        format="csv",
    )

    pass_manager.show_time_per_pass()
    logger.info(f"num_ops: {pass_manager.get_num_ops()}")


def run_cimflow(args, cim_config, op, max_keep=32):
    pass_list = [
        HardwareMapping5DPass(args, cim_config),
        BufferMappingPass(args, cim_config),
        TensorizePass(args, cim_config),
        CodegenPass(args, cim_config),
        BackendCompilePass(
            args, cim_config, n_workers=1, compile_data_layout=args.verify
        ),
    ]
    if args.verify:
        pass_list.append(VerifyPass(args))

    pass_manager = PassManager(pass_list)
    result = pass_manager.apply(op)

    save_table(
        result,
        [
            Column(name="name", attr_keys=["name"]),
            Column(name="pre_tile_sizes", attr_keys=["pre_tile_sizes"]),
            Column(name="affine schedule", attr_keys=["AffinePass", "schedule"]),
            Column(
                name="utilization", attr_keys=["UtilizationEvaluatePass", "utilization"]
            ),
            Column(
                name="compute_ops", attr_keys=["UtilizationEvaluatePass", "compute_ops"]
            ),
            Column(name="latency", attr_keys=["ProfilePass", "latency"]),
            Column(name="energy", attr_keys=["ProfilePass", "total_energy"]),
            Column(name="check_result", attr_keys=["VerifyPass", "check_result"]),
            Column(
                name="cim_compute_ops",
                attr_keys=["VerifyPass", "inst_stats", "CIMComputeInst"],
            ),
        ],
        os.path.join(args.output_path, "result.csv"),
        format="csv",
    )

    pass_manager.show_time_per_pass()
    logger.info(f"num_ops: {pass_manager.get_num_ops()}")
    return result
