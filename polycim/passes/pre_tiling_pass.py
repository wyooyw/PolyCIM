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

def change_dim_types_for_pre_tiling(split_factors, dim_types):
    new_dim_types = []
    for dim, dim_factors in enumerate(split_factors):
        if len(dim_factors) == 1:
            new_dim_types.append(dim_types[dim])
        elif len(dim_factors) == 2:
            new_dim_types.append(f"{dim_types[dim]}_o")
            new_dim_types.append(f"{dim_types[dim]}_i")
        elif len(dim_factors) == 3:
            new_dim_types.append(f"{dim_types[dim]}_o")
            new_dim_types.append(f"{dim_types[dim]}_m")
            new_dim_types.append(f"{dim_types[dim]}_i")
        else:
            raise ValueError(f"dim_factors={dim_factors} is not supported")
    return new_dim_types

class PreTilingSchedule(Schedule):
    def __init__(self, tile_sizes):
        super().__init__()
        self.tile_sizes = tile_sizes
    
    def dumps(self):
        return json.dumps(self.tile_sizes)

    def __hash__(self):
        return hash(self.tile_sizes)

    def __eq__(self, other):
        return self.tile_sizes == other.tile_sizes

class PreTilingPass(DepthFirstPass):
    def __init__(self, 
            args,
            fix_schedule: Optional[PreTilingSchedule]=None, 
            schedule_as_key: bool=False,
        ):
        super().__init__(
            fix_schedule=fix_schedule, 
            schedule_as_key=schedule_as_key
        )
        self.args = args
        assert self.fix_schedule is None or isinstance(self.fix_schedule, PreTilingSchedule)

    def apply(self, operator):
        symmetry_info = operator.attr.get("symmetry_info", None)
        dim_types = operator.attr.get("dim_types", None)
        max_tiling_level = operator.attr.get("max_tiling_level", 2)
        not_tiling = operator.attr.get("not_tiling", None)

        domain = operator.domain
        assert max_tiling_level in (2,3), f"{max_tiling_level=}"

        assert domain.is_box(), f"domain={domain} is not box"
        n_iter = domain.dim(isl.dim_type.set)

        domain_shape = utils.get_static_box_shape(domain)
        dim_factors = []
        for i,dim_size in enumerate(domain_shape):
            if not_tiling is not None and i in not_tiling:
                dim_factors.append(((dim_size,),))
                continue
            factors = factorize(dim_size, max_tiling_level)
            # factors = [tuple(factor) for factor in factors if factor[0]!=1 and factor[1]!=1]
            factors = remove_all_one_factors(factors)
            factors = list({tuple(factor) for factor in factors})
            # print(f"{factors=}")
            # factors = reversed(factors)
            # factors = [(dim_size,),*factors]
            dim_factors.append(tuple(factors))
        dim_factors = tuple(dim_factors)

        if symmetry_info is None:
            combination_list = list(itertools.product(*dim_factors))
        else:
            combination_list = combine_tilesize_by_symmetry_info(dim_factors, symmetry_info)
        
        
        if self.args.disable_pretile:
            combination_list = [
                combination for combination in combination_list if all(
                    [len(factors) == 1 for factors in combination]
                )
            ]

        if self.fix_schedule is not None:
            combination_list = [
                self.fix_schedule.tile_sizes
            ]

        result_list = []
        for idx,combination in enumerate(combination_list):
            new_operator = multi_level_splitting_var_level(operator, combination)
            new_dim_types = change_dim_types_for_pre_tiling(combination, dim_types)
            new_operator.set_attr("dim_types", new_dim_types, overwrite=True)
            new_operator.set_attr("pre_tile_sizes", combination)

            schedule = PreTilingSchedule(combination)
            result = SchedulePassResult(new_operator, schedule)
            result_list.append(result)

        return result_list
