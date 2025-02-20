from polycim.passes import (
    PreTilingPass,
    AffinePass,
    HardwareMappingPass,
    UtilizationEvaluatePass,
)
from polycim.passes.base import PassManager

def run_op_list(args, op_list, save_dir, pad_count, delay_apply, num_macros, cim_config):
    pass_manager = PassManager([
        PreTilingPass(args, schedule_as_key=True),
        AffinePass(args),
        HardwareMappingPass(args, cim_config),
        UtilizationEvaluatePass(args, cim_config),
    ])
    # exit()
    for name, config in op_list.items():
        op = config["op"]
        op.attr["symmetry_info"] = config["symmetry_info"]
        op.attr["dim_types"] = config["dim_types"]
        if "max_tiling_level" in config:
            op.attr["max_tiling_level"] = config["max_tiling_level"]
        if "not_tiling" in config:
            op.attr["not_tiling"] = config["not_tiling"]

        result = pass_manager.apply(op)
        for key, value in result.items():
            print(f"key:")
            for schedule in key.schedule_list:
                print(f"\t{schedule.__class__.__name__}={schedule.dumps()}")
            for op in value:
                print(f"\t{op.attr['UtilizationEvaluatePass::compute_ops']}")
        
        