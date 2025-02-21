from polycim.passes import (
    PreTilingPass,
    AffinePass,
    HardwareMappingPass,
    UtilizationEvaluatePass,
    DumpOpPass,
    MappingMultiMacroPass,
    TensorizePass,
    CodegenPass,
    BackendCompilePass,
    FilterSingleOpPass,
    BufferMappingPass,
)
from polycim.passes.base import PassManager

def run_op_list(args, op_list, save_dir, pad_count, delay_apply, num_macros, cim_config):
    pass_manager = PassManager([
        PreTilingPass(args, schedule_as_key=False),
        AffinePass(args),
        HardwareMappingPass(args, cim_config),
        UtilizationEvaluatePass(args, cim_config),
        DumpOpPass(args, cim_config),
        FilterSingleOpPass(),
        MappingMultiMacroPass(args, cim_config),
        BufferMappingPass(args, cim_config),
        TensorizePass(args, cim_config),
        CodegenPass(args, cim_config),
        BackendCompilePass(args, cim_config),
    ])

    for name, config in op_list.items():
        op = config["op"]
        op.attr["origin_op"] = op
        op.attr["symmetry_info"] = config["symmetry_info"]
        op.attr["dim_types"] = config["dim_types"]
        op.attr["name"] = name
        if "max_tiling_level" in config:
            op.attr["max_tiling_level"] = config["max_tiling_level"]
        if "not_tiling" in config:
            op.attr["not_tiling"] = config["not_tiling"]

        result = pass_manager.apply(op)
        exit()
        # unique_key = set()
        # for op in result:
        #     key = op.attr["PassManager::schedule_keys"]
        #     if key in unique_key:
        #         continue
        #     unique_key.add(key)
        #     print(f"key:")
        #     for schedule in key.schedule_list:
        #         print(f"\t{schedule.__class__.__name__}={schedule.dumps()}")
        #     print(f"\t{op.attr['UtilizationEvaluatePass']['compute_ops']}")
        
        