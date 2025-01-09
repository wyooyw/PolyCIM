from draw import (
    extract_frame_info,
    get_macro_hash,
    _extract_frame_info,
    extract_time_list,
    extract_val_from_singleton_set,
    FrameInfo
)
from utils import get_static_box_shape, get_mpf_lb_up_from_domain
import islpy as isl
def get_pieces_from_pw_multi_aff(pw_multi_aff):
    record = []
    pw_multi_aff.foreach_piece(lambda x,y: record.append((x,y)))
    return record
    
def get_dominate_iters_of_pw_multi_aff(pw_multi_aff, return_name=True):
    """
    {[i0,i1,..,ik] -> [f(i1,i2)]}
    return {i1,i2}
    """
    dim_names = [pw_multi_aff.get_dim_name(isl.dim_type.in_, i) for i in range(pw_multi_aff.dim(isl.dim_type.in_))]
    n_dim_range = pw_multi_aff.dim(isl.dim_type.out)

    # dominate_dims = set()
    # for i in range(pw_multi_aff.dim(isl.dim_type.in_)):
    #     if pw_multi_aff.involves_dims(isl.dim_type.in_, i, 1):
    #         dominate_dims.add(dim_names[i] if return_name else i)

    dominate_dims2 = set()
    for cond, multi_aff in get_pieces_from_pw_multi_aff(pw_multi_aff):
        for dim in range(n_dim_range):
            aff = multi_aff.get_at(dim)
            for i in range(aff.dim(isl.dim_type.in_)):
                # coef = aff.get_coefficient_val(isl.dim_type.in_, i) 
                if aff.involves_dims(isl.dim_type.in_, i, 1):
                    dominate_dims2.add(dim_names[i] if return_name else i)

    return dominate_dims2

def get_non_dominate_iters_of_pw_multi_aff(pw_multi_aff, return_name=True):
    dominate_dims = get_dominate_iters_of_pw_multi_aff(pw_multi_aff, return_name=return_name)
    if return_name:
        dim_names = [pw_multi_aff.get_dim_name(isl.dim_type.in_, i) for i in range(pw_multi_aff.dim(isl.dim_type.in_))]
        non_dominate_dims = set(dim_names) - dominate_dims
    else:
        non_dominate_dims = set(range(pw_multi_aff.dim(isl.dim_type.in_))) - dominate_dims
    return non_dominate_dims

def get_static_bound_dims(op):
    n_dim = op.domain.dim(isl.dim_type.set)
    static_bound_dims = set()
    print(f"get static_bound_dims begin")
    for iter_id in range(n_dim):
        lb,ub = get_mpf_lb_up_from_domain(op.domain, iter_id)
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

def _extract_frame_info(domain, acc_rel_input, acc_rel_macro, acc_rel_output, timestamp, macro_j, macro_k):
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
    macro_data =  [[None for _ in range(macro_j)] for _ in range(macro_k)]
    # import pdb; pdb.set_trace()
    def record(point):
        multi_val = point.get_multi_val()
        domain_frame = isl.Set(f"{{ [{','.join([str(multi_val.get_val(i)) for i in range(domain_n_dim)])}] }}")
        macro_point_data = acc_rel_macro.intersect_domain(domain_frame).range().remove_redundancies()
        assert macro_point_data.is_singleton(), f"macro_data should be singleton, but {macro_point_data}!"
        pos_j = int(str(multi_val.get_val(domain_n_dim-1)))
        pos_k = int(str(multi_val.get_val(domain_n_dim-2)))
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
    time_list = extract_time_list(domain_set, domain_set.n_dim()-2)
    print(f"{len(time_list)=}")
    macro_hash_list = set()
    
    for timestamp in time_list:
        frame_info = _extract_frame_info(domain = domain_set, 
                            acc_rel_input = software_op.access_I, 
                            acc_rel_macro = software_op.access_W, 
                            acc_rel_output = software_op.access_O, 
                            timestamp = timestamp, 
                            macro_j = cim_cfg.n_group_vcol, 
                            macro_k = cim_cfg.n_comp)
        macro_hash = get_macro_hash(frame_info.macro)
        if macro_hash in macro_hash_list:
            continue
        else:
            macro_hash_list.add(macro_hash)
            yield timestamp, frame_info

def count_minimal_needed_macro(op, cim_cfg):
    cnt = 0
    access_W = op.access_W
    not_involve_dims = get_non_dominate_iters_of_pw_multi_aff(access_W.as_pw_multi_aff(), return_name=False)
    trival_dims = get_static_bound_dims(op) - not_involve_dims
    # trival_dims = set()
    for value in extract_frame_info(op, cim_cfg, compress_iter_ids = not_involve_dims | trival_dims):
        cnt += 1
    
    n_dim = op.domain.dim(isl.dim_type.set)
    for iter_id in trival_dims:
        # outer = op.domain.project_out(isl.dim_type.set, iter_id + 1, n_dim-iter_id-1)
        dim_size = op.domain.dim_max_val(iter_id) - op.domain.dim_min_val(iter_id) + 1
        cnt *= dim_size
    
    return cnt

