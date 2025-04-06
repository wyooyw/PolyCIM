import islpy as isl

import polycim.utils.utils as utils
from polycim.utils.dominate import get_dominate_iters_of_pw_multi_aff
from polycim.utils.draw import (FrameInfo, _extract_frame_info,
                                extract_frame_info, extract_time_list,
                                extract_val_from_singleton_set, get_macro_hash)
from polycim.utils.utils import get_mpf_lb_up_from_domain


def get_static_bound_dims(op):
    n_dim = op.domain.dim(isl.dim_type.set)
    static_bound_dims = set()
    print(f"get static_bound_dims begin")
    for iter_id in range(n_dim):
        lb, ub = get_mpf_lb_up_from_domain(op.domain, iter_id)
        if lb.is_cst() and ub.is_cst():
            static_bound_dims.add(iter_id)
    print(f"{static_bound_dims=}")
    return static_bound_dims


def compress_domain_set(domain_set, compress_iter_ids):
    n_dim = domain_set.dim(isl.dim_type.set)
    iters = []
    for iter_id in range(n_dim):
        if iter_id in compress_iter_ids:
            min_val = int(str(domain_set.dim_min_val(iter_id)))
            iters.append(str(min_val))
        else:
            iters.append(f"i{iter_id}")
    compress_domain_str = "{ [" + ",".join(iters) + "] }"
    compress_domain = isl.Set(compress_domain_str)
    domain_set = domain_set.intersect(compress_domain)
    return domain_set


def _extract_frame_info(
    domain, acc_rel_input, acc_rel_macro, acc_rel_output, timestamp, macro_j, macro_k
):
    """
    ret: frame(input, output, macro) at this timestamp. What data access in this timestamp.

    We assume that schedule is an identity schedule.
    """
    acc_rel_macro = acc_rel_macro.as_map()
    domain_n_dim = domain.n_dim()
    hardware_n_dim = 2
    # print(type(domain),domain)
    # print(type(timestamp),timestamp)
    # ipdb.set_trace()
    domain = domain.intersect(timestamp)
    macro_data = [[None for _ in range(macro_j)] for _ in range(macro_k)]

    # import pdb; pdb.set_trace()
    def record(point):
        multi_val = point.get_multi_val()
        domain_frame = isl.Set(
            f"{{ [{','.join([str(multi_val.get_val(i)) for i in range(domain_n_dim)])}] }}"
        )
        macro_point_data = (
            acc_rel_macro.intersect_domain(domain_frame).range().remove_redundancies()
        )
        assert (
            macro_point_data.is_singleton()
        ), f"macro_data should be singleton, but {macro_point_data}!"
        pos_j = int(str(multi_val.get_val(domain_n_dim - 1)))
        pos_k = int(str(multi_val.get_val(domain_n_dim - 2)))
        macro_data[pos_k][pos_j] = extract_val_from_singleton_set(macro_point_data)

    domain.foreach_point(record)
    # return None
    return FrameInfo(None, None, macro_data)


def extract_frame_info(software_op, cim_cfg, compress_iter_ids):
    #
    # if not software_op.schedule.is_identity():
    # software_op = schedule_identity(software_op)

    # get time stamp
    domain_set = software_op.domain.as_set()
    domain_set = compress_domain_set(domain_set, compress_iter_ids)
    time_list = extract_time_list(domain_set, domain_set.n_dim() - 2)
    print(f"{len(time_list)=}")
    macro_hash_list = set()

    for timestamp in time_list:
        frame_info = _extract_frame_info(
            domain=domain_set,
            acc_rel_input=software_op.access_I,
            acc_rel_macro=software_op.access_W,
            acc_rel_output=software_op.access_O,
            timestamp=timestamp,
            macro_j=cim_cfg.n_group_vcol,
            macro_k=cim_cfg.n_comp,
        )
        macro_hash = get_macro_hash(frame_info.macro)
        if macro_hash in macro_hash_list:
            continue
        else:
            macro_hash_list.add(macro_hash)
            yield timestamp, frame_info


# def count_minimal_needed_macro(op, cim_cfg):
#     cnt = 0
#     access_W = op.access_W
#     not_involve_dims = get_non_dominate_iters_of_pw_multi_aff(access_W.as_pw_multi_aff(), return_name=False)
#     trival_dims = get_static_bound_dims(op) - not_involve_dims
#     # trival_dims = set()
#     for value in extract_frame_info(op, cim_cfg, compress_iter_ids = not_involve_dims | trival_dims):
#         cnt += 1

#     n_dim = op.domain.dim(isl.dim_type.set)
#     for iter_id in trival_dims:
#         # outer = op.domain.project_out(isl.dim_type.set, iter_id + 1, n_dim-iter_id-1)
#         dim_size = op.domain.dim_max_val(iter_id) - op.domain.dim_min_val(iter_id) + 1
#         cnt *= dim_size

#     return cnt


def count_minimal_needed_macro(op, cim_cfg):
    domain = op.domain
    n_dim = domain.dim(isl.dim_type.set)
    shape = utils.get_box_hull_shape(domain)
    intra_macro_iters = [n_dim - 2, n_dim - 1]
    inter_macro_iters = [n_dim - 4, n_dim - 3]
    cnt = shape[inter_macro_iters[0]] * shape[inter_macro_iters[1]]

    access_W = op.access_W
    dominate_weight_iters = get_dominate_iters_of_pw_multi_aff(
        access_W.as_pw_multi_aff(), return_name=False
    )
    dominate_weight_iters = (
        set(dominate_weight_iters) - set(intra_macro_iters) - set(inter_macro_iters)
    )
    for dominate_weight_iter in dominate_weight_iters:
        cnt *= shape[dominate_weight_iter]

    return cnt
