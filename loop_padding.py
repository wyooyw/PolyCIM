import islpy as isl
from base_operator import BasicOperator
from config import get_config
import utils

def loop_padding(op, _):
    
    domain = op.domain
    domain_size = domain.dim(isl.dim_type.set)
    domain_ub = [int(str(domain.dim_max_val(i))) for i in range(domain_size)]
    domain_lb = [int(str(domain.dim_min_val(i))) for i in range(domain_size)]

    cim_cfg = get_config()
    sizes = utils.get_box_hull_shape(domain)
    padding_inner_size=[
        # [0, cim_cfg.n_row - 1], # begin, size
        [0, cim_cfg.n_comp - 1], # begin, size
        [0, sizes[-3] - 1], # begin, size
        [0, sizes[-2] - 1], # begin, size
        [0, cim_cfg.n_group_vcol - 1]
    ]
    inner_loop_level = len(padding_inner_size)

    domain_outer = domain.project_out(isl.dim_type.set, domain_size-inner_loop_level, inner_loop_level)
    domain_padding = domain_outer.insert_dims(isl.dim_type.set, domain_size-inner_loop_level, inner_loop_level)
    for i in range(inner_loop_level):
        level = domain_size-inner_loop_level+i
        padding_lb = padding_inner_size[i][0]
        padding_ub = padding_inner_size[i][1]
        assert padding_inner_size[i][0] <= domain_lb[level]
        assert padding_inner_size[i][1] >= domain_ub[level]

        domain_padding = domain_padding.set_dim_name(isl.dim_type.set, level, domain.get_dim_name(isl.dim_type.set, level))
        domain_padding = domain_padding.lower_bound_val(isl.dim_type.set, level, isl.Val(padding_lb))
        domain_padding = domain_padding.upper_bound_val(isl.dim_type.set, level, isl.Val(padding_ub))

    return BasicOperator(
        domain = domain_padding,
        access_I = op.access_I,
        access_O = op.access_O,
        access_W = op.access_W,
        history_domains = [*op.history_domains, domain_padding],
        history_schedules = [*op.history_schedules, {"padding_inner_size":padding_inner_size}],
        attr={key:value for key,value in op.attr.items()}
    )
    # import pdb; pdb.set_trace()
    # pass

def loop_padding_pass(op_list, padding_inner_size):
    new_op_list = []
    for op in op_list:
        new_op = loop_padding(op, padding_inner_size)
        new_op_list.append(new_op)
    return new_op_list