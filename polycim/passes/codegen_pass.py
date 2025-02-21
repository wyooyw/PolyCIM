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
from polycim.codegen_.codegen_cimdsl import codegen_pass
from polycim.config import CIMConfig

class CodegenPass(DepthFirstPass):
    def __init__(self, 
            args,
            cim_config: CIMConfig,
            fix_schedule: Optional[Schedule]=None, 
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
        
        new_op = codegen_pass([operator])[0]
        result = SchedulePassResult(new_op, Schedule())
        return [result]