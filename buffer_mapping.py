import islpy as isl
from utils import (
    get_box_hull_shape,
    rename_all_dims_for_basic_set,
    rename_all_dims_for_basic_map,
    rename_out_dims_for_basic_map
)
import utils
from base_operator import BasicOperator, DataMovementOperator, DataMovement
from tqdm import tqdm

def find_domain_iters_exist_in_range(aff, return_name=True):

    n_iters_domain = aff.dim(isl.dim_type.in_)
    coefs = [aff.get_coefficient_val(isl.dim_type.in_, i) for i in range(n_iters_domain)]
    domain_iter_index_exist_in_range = []
    for i,coef in enumerate(coefs):
        if not coef.is_zero():
            domain_iter_index_exist_in_range.append(i)

    if return_name:
        domain_iter_names_exist_in_range = [aff.get_dim_name(isl.dim_type.in_, i) for i in domain_iter_index_exist_in_range]
        return domain_iter_names_exist_in_range

    return domain_iter_index_exist_in_range

def find_domain_iters_exist_in_range_list_of_aff(affs):
    domain_iter_names_exist_in_range = []
    for aff in affs:
        domain_iter_names_exist_in_range.extend(find_domain_iters_exist_in_range(aff))
    domain_iter_names_exist_in_range = list(set(domain_iter_names_exist_in_range))
    return domain_iter_names_exist_in_range

def pw_aff_to_aff(pw_aff):
    affs = pw_aff.get_pieces()
    assert len(affs)==1, f"{affs=}"
    aff = affs[0][1]
    assert type(aff)==isl.Aff
    return aff

def build_domain_aligned_buffer_exclude_iters(domain, buffer_name, exclude_iter_names = []):
    """
    args:
        domain: [i,j,k]
        exclude_iter_names: [j]
    return:
        access relation: [i,j,k] -> A[i,k]
    """

    shape = get_box_hull_shape(domain)
    iter_names = domain.get_var_names(isl.dim_type.set)
    n_iter = len(iter_names)

    # assert exclude_iter_names is subset of iter_names
    assert set(exclude_iter_names).issubset(set(iter_names)), f"{exclude_iter_names=}, {iter_names=}"

    iter_in_array_names = [iter_name for iter_name in iter_names if iter_name not in exclude_iter_names]
    access_relation = isl.BasicMap(f"{{ [{','.join(iter_names)}] -> {buffer_name}[{','.join(iter_in_array_names)}] }}")
    access_relation = access_relation.intersect_domain(domain)
    access_relation = rename_out_dims_for_basic_map(access_relation)
    return access_relation

def map_domain_aligned_buffer_to_origin_buffer_v2(domain, acc_rel):
    """
    1.get domain aligned buffer
    2.map this aligned buffer to origin buffer
    if a domain's iter not exist in buffer's iter, then no need to map it to buffer.
    """
    buffer_name = acc_rel.get_tuple_name(isl.dim_type.out)
    align_buffer_name = f"{buffer_name}_aligned"

    n_buf_dim = acc_rel.dim(isl.dim_type.out)
    lower_bound_per_dim = [acc_rel.dim_min(i) for i in range(n_buf_dim)]
    upper_bound_per_dim = [acc_rel.dim_max(i) for i in range(n_buf_dim)]

    lower_bound_per_dim = [pw_aff_to_aff(pw_aff) for pw_aff in lower_bound_per_dim]
    upper_bound_per_dim = [pw_aff_to_aff(pw_aff) for pw_aff in upper_bound_per_dim]

    domain_iter_names_exist_in_range_ub_lb = find_domain_iters_exist_in_range_list_of_aff(
        lower_bound_per_dim + upper_bound_per_dim
    )
    domain_iter_names = acc_rel.get_var_names(isl.dim_type.in_)
    domain_iter_names_not_exist_in_lb_ub = list(set(domain_iter_names) - set(domain_iter_names_exist_in_range_ub_lb))

    aligned_acc_rel = build_domain_aligned_buffer_exclude_iters(
        domain, 
        align_buffer_name, 
        domain_iter_names_not_exist_in_lb_ub
    )
    # one to many
    # buffer_mapping = acc_rel.reverse().apply_range(aligned_acc_rel)
    
    # many to one
    buffer_mapping = aligned_acc_rel.reverse().apply_range(acc_rel)
    assert buffer_mapping.is_single_valued()
    return buffer_mapping, aligned_acc_rel    





# def is_continuous(isl_obj, ):

def make_affs_to_aff_list(affs):
    aff_list = isl.AffList.alloc(affs[0].get_ctx(), 0)
    for aff in affs:
        aff_list = aff_list.add(aff)
    return aff_list

def make_pw_affs_to_aff_list(pw_affs):
    pw_aff_list = isl.PwAffList.alloc(pw_affs[0].get_ctx(), 0)
    for pw_aff in pw_affs:
        pw_aff_list = pw_aff_list.add(pw_aff)
    return pw_aff_list

def map_prefix_domain_aligned_buffer_to_aligned_buffer_v2(domain, acc_rel, level):
    n_buf_dim = acc_rel.dim(isl.dim_type.out)
    n_iter_dim = acc_rel.dim(isl.dim_type.in_)
    # assert n_buf_dim==n_iter_dim, f"{n_buf_dim=}, {n_iter_dim=}"

    iter_names = domain.get_var_names(isl.dim_type.set)
    prefix_acc_rel = acc_rel.project_out_except(iter_names[:level], [isl.dim_type.in_])
    local_buffer_dynamic_shape = utils.get_dynamic_shape_from_dynamic_map(prefix_acc_rel) #[level:]
    # print(f"{local_buffer_dynamic_shape = }")
    # check prefix_acc_rel is continous on given dim
    
    lower_bound_per_dim = [prefix_acc_rel.dim_min(i) for i in range(n_buf_dim)]
    lb_aff_per_dim = lower_bound_per_dim
    
    n_local_buf_dim = n_buf_dim # - level

    param_names = acc_rel.get_var_names(isl.dim_type.param)
    domain_names = acc_rel.get_var_names(isl.dim_type.in_)
    range_names = acc_rel.get_var_names(isl.dim_type.out)

    local_buffer_iters = [utils.get_unique_name() for i in range(n_local_buf_dim)]
    # insert local buffer dim for lb_aff_per_dim
    for i in range(len(lb_aff_per_dim)):
        
        lb_aff = lb_aff_per_dim[i]
        lb_aff = lb_aff.insert_dims(isl.dim_type.in_, level, n_local_buf_dim)
        for j in range(n_local_buf_dim):
            lb_aff = lb_aff.set_dim_id(isl.dim_type.in_, level + j, isl.Id(local_buffer_iters[j]))
        lb_aff_per_dim[i] = lb_aff
    
    # build buffer's access relation
    affs = []
    aff_domain_iters = lb_aff_per_dim[0].get_var_names(isl.dim_type.in_)
    aff_domain_def = ','.join(aff_domain_iters)
    for i in range(n_buf_dim):
        affs.append(lb_aff_per_dim[i])
        
    
    assert n_local_buf_dim <= n_buf_dim, f"{n_local_buf_dim=}, {n_buf_dim=}"
    for i in range(n_local_buf_dim):
        aff_lb = affs[n_buf_dim - n_local_buf_dim + i]
        aff_i = isl.Aff(f"{{ [{aff_domain_def}] -> [({local_buffer_iters[i]})] }}")
        aff = aff_lb.add(aff_i)
        affs[n_buf_dim - n_local_buf_dim + i] = aff
    
    pw_aff_list = utils.make_pw_affs_to_aff_list(affs)
    
    assign_buffer_acc_rel = isl.MultiPwAff.from_pw_aff_list(affs[0].space.insert_dims(isl.dim_type.out, 0, len(pw_aff_list)-1), pw_aff_list)
    assign_buffer_acc_rel = isl.PwMultiAff.from_multi_pw_aff(assign_buffer_acc_rel)
    assign_buffer_acc_rel = isl.Map.from_pw_multi_aff(assign_buffer_acc_rel)

    # basic_maps = assign_buffer_acc_rel.get_basic_maps()
    # assert len(basic_maps)==1, f"{len(basic_maps)=}"
    # assign_buffer_acc_rel = basic_maps[0]
    
    # build local buffer's access relation
    affs = []
    for i in range(n_local_buf_dim):
        aff = isl.Aff(f"{{ [{aff_domain_def}] -> [({local_buffer_iters[i]})] }}")
        affs.append(aff)
    aff_list = make_affs_to_aff_list(affs)
    local_buffer_acc_rel = isl.BasicMap.from_aff_list(affs[0].domain().space, aff_list)
    
    # build assign domain
    assign_domain = domain.project_out_except(iter_names[:level], [isl.dim_type.set])
    assign_domain = assign_domain.add_dims(isl.dim_type.set, n_local_buf_dim)
    for i in range(n_local_buf_dim):
        assign_domain = assign_domain.set_dim_name(isl.dim_type.set, level + i, local_buffer_iters[i])

    ub_mpf = utils.multi_pw_aff_from_pw_affs(local_buffer_dynamic_shape)
    ub_mpf = ub_mpf.add_constant_val(isl.Val.int_from_si(ub_mpf.get_ctx(),-1))

    # if level==2:
    #     import pdb; pdb.set_trace()
    assign_domain = utils.mpf_upper_bound_for_basic_set(ub_mpf, assign_domain, n_local_buf_dim)
    assign_domain = utils.zero_lower_bound_for_basic_set(assign_domain, n_local_buf_dim)

    # set tuple name
    buffer_name = acc_rel.get_tuple_name(isl.dim_type.out)
    assign_buffer_acc_rel = assign_buffer_acc_rel.set_tuple_name(isl.dim_type.out, buffer_name)
    local_buffer_acc_rel = local_buffer_acc_rel.set_tuple_name(isl.dim_type.out, f"{buffer_name}_{level}")

    
    return assign_domain, local_buffer_acc_rel, assign_buffer_acc_rel

def apply_skew(domain, acc_rel):
    skew_map = isl.BasicMap("{ [i,j,k] -> [i,j+k,k] }")
    new_domain = skew_map.intersect_domain(domain).range()
    new_acc_rel = skew_map.reverse().apply_range(acc_rel)

    new_acc_rel = rename_all_dims_for_basic_map(new_acc_rel)
    new_domain = rename_all_dims_for_basic_set(new_domain)
    return new_domain, new_acc_rel

def apply_tile(domain, acc_rel):
    tile_map = isl.BasicMap("{ [i,j,k] -> [floor(i/2),floor(j/2),floor(k/2),i%2,j%2,k%2] }")
    new_domain = tile_map.intersect_domain(domain).range()
    new_acc_rel = tile_map.reverse().apply_range(acc_rel)

    new_acc_rel = rename_all_dims_for_basic_map(new_acc_rel)
    new_domain = rename_all_dims_for_basic_set(new_domain)
    return new_domain, new_acc_rel

def get_range_dim_size(acc_rel, pos, return_int=True):
    assert type(acc_rel)==isl.BasicMap
    dim_min_pw_aff = acc_rel.dim_min(pos)
    dim_max_pw_aff = acc_rel.dim_max(pos)
    
    dim_len = dim_max_pw_aff.sub(dim_min_pw_aff)
    dim_len = dim_len.add_constant_val(isl.Val.one(dim_len.get_ctx()))
    
    dim_len_ub = dim_len.max_val()
    assert dim_len_ub.is_int()
    if return_int:
        dim_len_ub = int(str(dim_len_ub))

    return dim_len_ub

def get_static_shape_from_dynamic_map(isl_map, return_list_int = True):
    # num dim in out
    n_out = isl_map.dim(isl.dim_type.out)
    shape = [get_range_dim_size(isl_map, pos, return_list_int) for pos in range(n_out)]
    if return_list_int:
        shape = [shape_i for shape_i in shape]
    return shape




def insert_const_dim_in_range(map_, pos, si):
    map_ = map_.insert_dims(isl.dim_type.out, pos, 1)
    val = isl.Val.int_from_si(map_.get_ctx(), si)
    map_ = map_.upper_bound_val(isl.dim_type.out, pos, val)
    map_ = map_.lower_bound_val(isl.dim_type.out, pos, val)
    return map_

def insert_many_const_dim_in_range(map_, pos, size, si):
    map_ = map_.insert_dims(isl.dim_type.out, pos, size)
    val = isl.Val.int_from_si(map_.get_ctx(), si)
    for i in range(size):
        map_ = map_.upper_bound_val(isl.dim_type.out, pos + i, val)
        map_ = map_.lower_bound_val(isl.dim_type.out, pos + i, val)
    return map_

def align_compute_and_assign_schedules(compute_schedule, assign_schedules, levels):
    level_to_assign_schedule = dict()
    assign_schedule_to_level = dict()
    for assign_schedule, level in zip(assign_schedules, levels):
        if level in level_to_assign_schedule:
            level_to_assign_schedule[level].append(assign_schedule)
        else:
            level_to_assign_schedule[level] = [assign_schedule]

        assign_schedule_to_level[assign_schedule] = level

    sorted_levels = sorted(list(level_to_assign_schedule.keys()))
    print(f"{sorted_levels=}")

    # insert dims
    for level in levels:
        assign_schedules_at_level = level_to_assign_schedule[level]

        # insert constant dims for assign schedules at current level
        for i in range(len(assign_schedules_at_level)):
            assign_schedules_at_level[i] = insert_const_dim_in_range(assign_schedules_at_level[i], level, i)

        # insert dims for other schedule
        const = len(assign_schedules_at_level)
        for other_level in levels:
            if other_level==level:
                continue
            assign_schedules_at_level[i] = insert_const_dim_in_range(assign_schedules_at_level[i], level, const)
        compute_schedule = insert_const_dim_in_range(compute_schedule, level, const)

    # padding schedule at end
    max_range_size = compute_schedule.dim(isl.dim_type.out)
    for level in levels:
        assign_schedules_at_level = level_to_assign_schedule[level]
        for assign_schedule in assign_schedules_at_level:
            range_size = assign_schedule.dim(isl.dim_type.out)
            max_range_size = max(max_range_size, range_size)

    cur_range_size = compute_schedule.dim(isl.dim_type.out)
    compute_schedule = insert_many_const_dim_in_range(compute_schedule, cur_range_size, max_range_size - cur_range_size, 0)
    for level in levels:
        assign_schedules_at_level = level_to_assign_schedule[level]
        for i in range(len(assign_schedules_at_level)):
            assign_schedule = assign_schedules_at_level[i]
            cur_range_size = assign_schedule.dim(isl.dim_type.out)
            assign_schedule = insert_many_const_dim_in_range(assign_schedule, cur_range_size, max_range_size - cur_range_size, 0)
            assign_schedules_at_level[i] = assign_schedule
    
    union_schedule = compute_schedule
    for level in levels:
        assign_schedules_at_level = level_to_assign_schedule[level]
        for assign_schedule in assign_schedules_at_level:
            union_schedule = union_schedule.add_map(assign_schedule)

    return union_schedule

def map_access_to_buffer(origin_access, assign_buffer_input_access, assign_buffer_output_access, level):
    """
    origin_access: compute domain -> global array
    assign_buffer_input_access: assign domain -> global array
    assign_buffer_output_access: assign domain -> local buffer

    return: compute domain -> local buffer
    """
    
    assign_buffer_input_access = assign_buffer_input_access.move_dims(isl.dim_type.param, 0, isl.dim_type.in_, 0, level)
    assign_buffer_output_access = assign_buffer_output_access.move_dims(isl.dim_type.param, 0, isl.dim_type.in_, 0, level)
    
    global_local_buffer_mapping = assign_buffer_input_access.reverse().apply_range(assign_buffer_output_access)

    origin_access = origin_access.move_dims(isl.dim_type.param, 0, isl.dim_type.in_, 0, level)
    local_buffer_access = origin_access.apply_range(global_local_buffer_mapping)
    local_buffer_access = local_buffer_access.move_dims(isl.dim_type.in_, 0, isl.dim_type.param, 0, level)
    return local_buffer_access

def insert_single_buffer_single_level(op, buffer_name, buffer_level):
    map_buf_align_to_ori, aligned_acc_rel = map_domain_aligned_buffer_to_origin_buffer_v2(op.domain, op.get_access_by_name(buffer_name))
    assign_domain, assign_local_buffer_acc_rel, assign_global_buffer_acc_rel = map_prefix_domain_aligned_buffer_to_aligned_buffer_v2(op.domain, aligned_acc_rel, buffer_level)
    compute_local_buffer_acc_rel = map_access_to_buffer(aligned_acc_rel, assign_global_buffer_acc_rel, assign_local_buffer_acc_rel, buffer_level)
    
    accesses = {
        "I": op.access_I,
        "O": op.access_O,
        "W": op.access_W
    }
    accesses[buffer_name] = compute_local_buffer_acc_rel
    new_op = DataMovementOperator(
        domain = op.domain,
        access_I = accesses["I"],
        access_O = accesses["O"],
        access_W = accesses["W"],
    )
    datamove = DataMovement(
        domain = assign_domain,
        access_I = assign_global_buffer_acc_rel.intersect_domain(assign_domain),
        access_O = assign_local_buffer_acc_rel.intersect_domain(assign_domain),
        level = buffer_level
    )
    new_op.insert_buffer(buffer_name, datamove)
    
    # compute_schedule = utils.identity_map_from_set(op.domain)
    # assign_schedule = utils.identity_map_from_set(assign_domain)

    # compute_domain = op.domain.set_tuple_name("S")
    # compute_schedule = compute_schedule.set_tuple_name(isl.dim_type.in_, "S")

    # assign_domain = assign_domain.set_tuple_name("T")
    # assign_schedule = assign_schedule.set_tuple_name(isl.dim_type.in_, "T")

    # union_domain = compute_domain.add_set(assign_domain) #.add_set(assign_domain2)
    # union_schedule = align_compute_and_assign_schedules(compute_schedule, [assign_schedule], [buffer_level])

    # ast = utils.gen_ast(union_domain,union_schedule,None)
    # code = utils.gen_code(union_domain,union_schedule,None)
    return new_op

def insert_single_buffer_single_level_pass(op_list, buffer_name, buffer_level):
    new_codes = []
    for op in tqdm(op_list):
        new_op = insert_single_buffer_single_level(op, buffer_name, buffer_level)
        new_codes.append(new_op)
    return new_codes

def parse_buffer_levels(op, buffer_levels):
    n_domain_dim = op.domain.dim(isl.dim_type.set)
    new_buffer_levels= []
    for buffer_level in buffer_levels:
        if buffer_level < 0:
            buffer_level = n_domain_dim + buffer_level + 1
        new_buffer_levels.append(buffer_level)
    # check increase
    for i in range(1, len(new_buffer_levels)):
        assert new_buffer_levels[i] > new_buffer_levels[i-1], f"{new_buffer_levels=}"
    return new_buffer_levels

def insert_single_buffer_multi_level(
    op, buffer_name, buffer_levels
):
    buffer_levels = parse_buffer_levels(op, buffer_levels)

    assert isinstance(buffer_levels, list)
    buffer_levels = sorted(buffer_levels)

    map_buf_align_to_ori, aligned_acc_rel = map_domain_aligned_buffer_to_origin_buffer_v2(op.domain, op.get_access_by_name(buffer_name))

    compute_acc_rel = aligned_acc_rel
    data_movement_list = []
    
    for buffer_level in buffer_levels:
        assign_domain, assign_local_buffer_acc_rel, assign_global_buffer_acc_rel = map_prefix_domain_aligned_buffer_to_aligned_buffer_v2(op.domain, compute_acc_rel, buffer_level)
        compute_acc_rel = map_access_to_buffer(compute_acc_rel, assign_global_buffer_acc_rel, assign_local_buffer_acc_rel, buffer_level)
        datamove = DataMovement(
            domain = assign_domain,
            access_I = assign_global_buffer_acc_rel.intersect_domain(assign_domain),
            access_O = assign_local_buffer_acc_rel.intersect_domain(assign_domain),
            level = buffer_level
        )
        data_movement_list.append(datamove)

    accesses = {
        "I": op.access_I,
        "O": op.access_O,
        "W": op.access_W
    }
    accesses[buffer_name] = compute_acc_rel
    new_op = DataMovementOperator(
        domain = op.domain,
        access_I = accesses["I"],
        access_O = accesses["O"],
        access_W = accesses["W"],
        history_domains=op.history_domains, 
        history_schedules=op.history_schedules
    )

    print(f"domain: {op.domain}\n")
    print(f"access_I: {op.access_I}\n")
    print(f"access_O: {op.access_O}\n")
    print(f"access_W: {op.access_W}\n")
    
    for idx,data_movement in enumerate(data_movement_list):
        new_op.insert_buffer(buffer_name, data_movement)
        print(f"{idx}. {data_movement.domain=}\n")
    print("\n-----------------------\n")

    

    return new_op

def insert_single_buffer_multi_level_pass(op_list, buffer_name, buffer_levels):
    """
    new_ops = insert_single_buffer_multi_level_pass(new_ops, buffer_name="W", buffer_levels=[0, 2])
    """
    new_codes = []
    for op in tqdm(op_list):
        new_op = insert_single_buffer_multi_level(op, buffer_name, buffer_levels)
        new_codes.append(new_op)
    return new_codes

"""
Buffer Searching
"""
def buffer_level_serching():
    pass

if __name__=="__main__":


    """
    domain: { [i0, i1, i2, i3, i4, i5] : i1 >= -2 + i0 and 0 <= i1 <= 63 and i1 <= i0 and 0 <= i4 <= 3 and 0 <= i5 <= 3 and 4*floor((i5)/4) >= -63 + 4i3 + i5 and 4*floor((i5)/4) <= 4i3 + i5 and -2 - 66i0 + 4i2 - 4i3 + i4 - i5 + 4*floor((i5)/4) <= 4*floor((i4)/4) <= -66i0 + 4i2 - 4i3 + i4 - i5 + 4*floor((i5)/4) }

    access_I: { [i0, i1, i2, i3, i4, i5] -> I[o0, o1] : o1 = i0 and (i4 + o0) mod 2 = 0 and (2i0 - i4 + o0) mod 4 = 0 and i1 >= -2 + i0 and 0 <= i1 <= 63 and i1 <= i0 and 0 <= i4 <= 3 and 0 <= i5 <= 3 and -66i0 + 4i2 <= o0 <= 3 - 66i0 + 4i2 and 4*floor((i5)/4) >= -63 + 4i3 + i5 and 4*floor((i5)/4) >= 4i3 + i5 - o0 and 4*floor((i5)/4) <= 2 + 4i3 + i5 - o0 and 4*floor((i5)/4) <= 4i3 + i5 }

    access_O: { [i0, i1, i2, i3, i4, i5] -> O[o0, o1] : o1 = i1 and (-i5 + o0) mod 4 = 0 and i1 >= -2 + i0 and 0 <= i1 <= 63 and i1 <= i0 and 0 <= i4 <= 3 and 0 <= i5 <= 3 and o0 >= 4i3 and 0 <= o0 <= 63 and o0 <= 3 + 4i3 and -2 - 66i0 + 4i2 + i4 - o0 <= 4*floor((i4)/4) <= -66i0 + 4i2 + i4 - o0 }

    access_W: { [i0, i1, i2, i3, i4, i5] -> W[o0, o1] : o1 = i0 - i1 and (i4 + i5 + o0) mod 2 = 0 and (2i0 - i4 + i5 + o0) mod 4 = 0 and i1 >= -2 + i0 and 0 <= i1 <= 63 and i1 <= i0 and 0 <= i4 <= 3 and 0 <= i5 <= 3 and 0 <= o0 <= 2 and 4*floor((i4)/4) >= -63 - 66i0 + 4i2 + i4 - o0 and -3 - 66i0 + 4i2 - 4i3 + i4 - o0 <= 4*floor((i4)/4) <= -66i0 + 4i2 - 4i3 + i4 - o0 and 4*floor((i4)/4) <= -66i0 + 4i2 + i4 - o0 }
    """
    operator = BasicOperator(
        domain = isl.BasicSet(
            "{ [i0, i1, i2, i3, i4, i5] : i1 >= -2 + i0 and 0 <= i1 <= 63 and i1 <= i0 and 0 <= i4 <= 3 and 0 <= i5 <= 3 and 4*floor((i5)/4) >= -63 + 4i3 + i5 and 4*floor((i5)/4) <= 4i3 + i5 and -2 - 66i0 + 4i2 - 4i3 + i4 - i5 + 4*floor((i5)/4) <= 4*floor((i4)/4) <= -66i0 + 4i2 - 4i3 + i4 - i5 + 4*floor((i5)/4) }"
        ),
        access_I = isl.BasicMap("{ [i0, i1, i2, i3, i4, i5] -> I[o0, o1] : o1 = i0 and (i4 + o0) mod 2 = 0 and (2i0 - i4 + o0) mod 4 = 0 and i1 >= -2 + i0 and 0 <= i1 <= 63 and i1 <= i0 and 0 <= i4 <= 3 and 0 <= i5 <= 3 and -66i0 + 4i2 <= o0 <= 3 - 66i0 + 4i2 and 4*floor((i5)/4) >= -63 + 4i3 + i5 and 4*floor((i5)/4) >= 4i3 + i5 - o0 and 4*floor((i5)/4) <= 2 + 4i3 + i5 - o0 and 4*floor((i5)/4) <= 4i3 + i5 }"),
        access_O = isl.BasicMap("{ [i0, i1, i2, i3, i4, i5] -> O[o0, o1] : o1 = i1 and (-i5 + o0) mod 4 = 0 and i1 >= -2 + i0 and 0 <= i1 <= 63 and i1 <= i0 and 0 <= i4 <= 3 and 0 <= i5 <= 3 and o0 >= 4i3 and 0 <= o0 <= 63 and o0 <= 3 + 4i3 and -2 - 66i0 + 4i2 + i4 - o0 <= 4*floor((i4)/4) <= -66i0 + 4i2 + i4 - o0 }"),
        access_W = isl.BasicMap("{ [i0, i1, i2, i3, i4, i5] -> W[o0, o1] : o1 = i0 - i1 and (i4 + i5 + o0) mod 2 = 0 and (2i0 - i4 + i5 + o0) mod 4 = 0 and i1 >= -2 + i0 and 0 <= i1 <= 63 and i1 <= i0 and 0 <= i4 <= 3 and 0 <= i5 <= 3 and 0 <= o0 <= 2 and 4*floor((i4)/4) >= -63 - 66i0 + 4i2 + i4 - o0 and -3 - 66i0 + 4i2 - 4i3 + i4 - o0 <= 4*floor((i4)/4) <= -66i0 + 4i2 - 4i3 + i4 - o0 and 4*floor((i4)/4) <= -66i0 + 4i2 + i4 - o0 }"),
    )
    new_op = insert_single_buffer_multi_level(operator, "I", [4])
    # print(code)
    
def test():
    domain = isl.BasicSet("{ [i,j,k]: 0 <= i < 4 and 0 <= j < 4 and 0 <= k < 4}")
    acc_rel=isl.BasicMap("{ [i,j,k] -> A[j,k] }")
    # acc_rel2=isl.BasicMap("{ [i,j,k] -> B[i,j] }")
    domain, acc_rel = apply_skew(domain, acc_rel)
    domain, acc_rel = apply_tile(domain, acc_rel)
    print(f"{domain = }")
    print(f"{acc_rel = }")
    print("----------------------------")
    # acc_rel=isl.BasicMap("{ [i,j,k] -> A[i * 2, k] }")
    map_buf_align_to_ori, aligned_acc_rel = map_domain_aligned_buffer_to_origin_buffer_v2(domain, acc_rel)
    # map_buf_align_to_ori2, aligned_acc_rel2 = map_domain_aligned_buffer_to_origin_buffer_v2(domain, acc_rel2)
    print(f"{aligned_acc_rel = }")
    print("----------------------------")
    assign_domain, local_buffer_acc_rel, assign_buffer_acc_rel = map_prefix_domain_aligned_buffer_to_aligned_buffer_v2(domain, aligned_acc_rel, 4)
    # assign_domain2, local_buffer_acc_rel2, assign_buffer_acc_rel2 = map_prefix_domain_aligned_buffer_to_aligned_buffer_v2(domain, aligned_acc_rel2, 4)
    print(f"{assign_domain = }")
    print(f"{local_buffer_acc_rel = }")
    print(f"{assign_buffer_acc_rel = }")

    print("----------------------------")
    print(f"{domain=}")
    print(f"{assign_domain=}")
    compute_schedule = utils.identity_map_from_set(domain)
    print(f"{compute_schedule=}")
    assign_schedule = utils.identity_map_from_set(assign_domain)
    # assign_schedule2 = utils.identity_map_from_set(assign_domain2)
    print(f"{assign_schedule=}")


    compute_domain = domain.set_tuple_name("S")
    compute_schedule = compute_schedule.set_tuple_name(isl.dim_type.in_, "S")

    assign_domain = assign_domain.set_tuple_name("T")
    assign_schedule = assign_schedule.set_tuple_name(isl.dim_type.in_, "T")

    # assign_domain2 = assign_domain2.set_tuple_name("P")
    # assign_schedule2 = assign_schedule2.set_tuple_name(isl.dim_type.in_, "P")

    union_domain = compute_domain.add_set(assign_domain) #.add_set(assign_domain2)
    # assign_schedule2
    union_schedule = align_compute_and_assign_schedules(compute_schedule, [assign_schedule], [4])
    print("--------------------------------------------")
    print(f"{type(union_domain)}, {union_domain=}\n")
    print(f"{type(union_schedule)}, {union_schedule=}\n")
    ast = utils.gen_ast(union_domain,union_schedule,None)
    code = utils.gen_code(union_domain,union_schedule,None)
    print(ast,"\n")
    print(code)
    print(type(ast), ast.get_type())
    print("\n-------------------------------------\n")

    from ast_ import codegen_str
    print(codegen_str(ast))

    exit()
    print(f"{assign_buffer_acc_rel = }")
    print("----------------------------------")
    pma = assign_buffer_acc_rel.as_pw_multi_aff()
    def show(cond, ma):
        print(f"- {cond = }")
        print(f"- {ma = }")
        filter_acc_rel = assign_buffer_acc_rel.intersect_domain(cond)
        print(f"- {filter_acc_rel = }")
        print("")
    pma.foreach_piece(show)