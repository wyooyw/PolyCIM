import json
import islpy as isl
from typing import Optional
from polycim.passes.base import Schedule
from polycim.passes.base import DepthFirstPass
from polycim.passes.base import SchedulePassResult
import polycim.utils.utils as utils
from polycim.passes.multi_level_tiling import (
    remove_all_one_factors,
    combine_tilesize_by_symmetry_info,
    multi_level_splitting_var_level,
)
from polycim.utils.math import factorize
from polycim.depth_first.mapping_multiple_macro import mapping_multiple_macro
from polycim.codegen_.codegen_data_layout_convert import (
            gcc_compile_data_layout_convert_code
        )
from polycim.config import CIMConfig
import os
class MappingMultiMacroSchedule(Schedule):
    def __init__(self):
        super().__init__()
        
class MappingMultiMacroPass(DepthFirstPass):
    def __init__(self, 
            args,
            cim_config: CIMConfig,
            fix_schedule: Optional[MappingMultiMacroSchedule]=None, 
            schedule_as_key: bool=False,
        ):
        super().__init__(
            fix_schedule=fix_schedule, 
            schedule_as_key=schedule_as_key
        )
        assert self.fix_schedule is None
        assert self.schedule_as_key is False
        
        self.args = args
        self.cim_config = cim_config
        self.cnt = 0

    def apply(self, operator):
        
        new_op = mapping_multiple_macro(self.args, operator, self.cim_config)
        # save data layout convert code
        data_layout_convert_code = new_op.attr["data_layout_convert_code"]
        save_op_dir = os.path.join(self.args.output_path, operator.attr["name"], str(self.cnt))
        os.makedirs(save_op_dir, exist_ok=True)
        # import pdb; pdb.set_trace()
        for key, value in data_layout_convert_code.items():
            code_path = os.path.join(save_op_dir, f"convert_{key}.cpp")   
            with open(code_path, "w") as f:
                f.write(value)
            exe_path = os.path.join(save_op_dir, f"convert_{key}.o")
            gcc_compile_data_layout_convert_code(code_path, exe_path)

        self.cnt += 1

        schedule = MappingMultiMacroSchedule()
        result = SchedulePassResult(new_op, schedule)
        return [result]