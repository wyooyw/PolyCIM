import json
import islpy as isl
from typing import Optional
from polycim.passes.base import Schedule
from polycim.passes.base import SchedulePass
from polycim.passes.base import SchedulePassResult
import polycim.utils.utils as utils
from polycim.passes.multi_level_tiling import (
    remove_all_one_factors,
    combine_tilesize_by_symmetry_info,
    multi_level_splitting_var_level,
)
from polycim.utils.math import factorize
from polycim.passes.hardware_merge_tiling import (
    get_coalescing_schedule_from_mapping,
    get_reverse_coalescing_schedule_from_mapping,
    _get_hardware_tiling_schedule
)
from collections import OrderedDict
from polycim.config import get_config, CIMConfig


def get_mapping_from_bases(bases):
    reuse_axis = dict()
    all_array_ids = {base.reuse_array_id for base in bases if base.reuse_array_id != -1}
    for array_id in all_array_ids:
        reuse_axis[array_id] = list()

    for i,base in enumerate(bases):
        if base.reuse_array_id != -1:
            reuse_axis[base.reuse_array_id].append(f"s{i}")

    mapping = OrderedDict()
    for array_id, axis_list in reuse_axis.items():
        # h0: row, reuse output;
        # h1: col, reuse input
        hardware_axis = f"h{(1-array_id)}"

        mapping[hardware_axis] = tuple(axis_list)

    return mapping

class HardwareMappingSchedule(Schedule):
    def __init__(self, s2h_mapping, tiling_factors):
        super().__init__()
        self.s2h_mapping = s2h_mapping
        self.tiling_factors = tiling_factors
    
    def dumps(self):
        return json.dumps({
            "s2h_mapping":self.s2h_mapping,
            "tiling_factors":self.tiling_factors
        })
    
    def parse(self, data):
        assert isinstance(data, dict)
        self.s2h_mapping = data["s2h_mapping"]
        self.tiling_factors = data["tiling_factors"]

class HardwareMappingPass(SchedulePass):
    def __init__(self, 
            args,
            cim_config: CIMConfig,
            fix_schedule: Optional[HardwareMappingSchedule]=None, 
            schedule_as_key: bool=False,
        ):
        super().__init__(
            fix_schedule=fix_schedule, 
            schedule_as_key=schedule_as_key
        )
        self.args = args
        self.cim_config = cim_config
        assert self.fix_schedule is None or isinstance(self.fix_schedule, PreTilingSchedule)

    def apply(self, operator):
        mapping = get_mapping_from_bases(operator.attr["affine::bases"])
        operator.history_schedules.append({"s2h_mapping":mapping})
        coalescing_schedule = get_coalescing_schedule_from_mapping(mapping, operator)
        reverse_coalescing_schedule = get_reverse_coalescing_schedule_from_mapping(mapping, operator)
        tiling_factor = [
            self.cim_config.n_comp, self.cim_config.n_group_vcol
        ]
        tiling_schedule = _get_hardware_tiling_schedule(coalescing_schedule.range().dim(isl.dim_type.set), tiling_factor)
        new_op = operator.apply_schedule(coalescing_schedule, 
            skip_simplify=True, 
            name="coalescing"
        )
        new_op = new_op.apply_schedule(tiling_schedule, skip_simplify=True, name="tiling")
        schedule = HardwareMappingSchedule(mapping, tiling_factor)

        if self.fix_schedule is not None:
            assert self.fix_schedule.s2h_mapping == mapping
            assert self.fix_schedule.tiling_factors == tiling_factor
            
        result = SchedulePassResult(new_op, schedule)
        return [result]