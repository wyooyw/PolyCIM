import islpy as isl
from base_operator import BasicOperator
import utils
import itertools
from tqdm import tqdm

def multiply(factors):
    result = 1
    for factor in factors:
        result *= factor
    return result

def multi_level_tiling(operator, tiling_level, tiling_factors):
    """
    operator = Operator(
        domain=isl.BasicSet("{ [i,j]: 0 <= i < 8 and 0 <= j < 3 }"),
        acc_rel_out=isl.BasicMap("{ [i,j] -> C[i] }"), 
        acc_rel_lhs=isl.BasicMap("{ [i,j] -> A[i + j] }"), 
        acc_rel_rhs=isl.BasicMap("{ [i,j] -> B[i + j] }")
    )

    tiling_factors: 
    [
        [2,2],
        [1,4]
    ]
    """
    param_names = operator.domain.get_var_names(isl.dim_type.param)

    domain=operator.domain

    assert domain.is_box(), f"domain={domain} is not box"
    n_iter = domain.dim(isl.dim_type.set)
    assert len(tiling_factors)==n_iter, f"len(tiling_factors)={len(tiling_factors)} != n_iter={n_iter}"

    domain_shape = utils.get_static_box_shape(domain)

    # do checks
    for i in range(n_iter):
        assert len(tiling_factors[i])==tiling_level, f"len(tiling_factors[{i}])={len(tiling_factors[i])} != tiling_level={tiling_level}"
        dim_size = domain_shape[i]
        assert multiply(tiling_factors[i])==dim_size, f"multiply(tiling_factors[{i}])={multiply(tiling_factors[i])} != dim_size={dim_size}"

    # do tiling
    tiling_maps = []
    remain_factors = [*domain_shape]
    for l in range(tiling_level-1):
        n_keep_iters = l * n_iter
        keep_iter_names = [f"i{i}" for i in range(n_keep_iters)]
        change_iter_names = [f"i{i}" for i in range(n_keep_iters, n_keep_iters+n_iter)]

        outer_iters = []
        inner_iters = []
        for i in range(n_iter):
            factor = tiling_factors[i][l]
            remain_factors[i] = remain_factors[i] // factor
            outer_iters.append(f"floor({change_iter_names[i]}/{remain_factors[i]})")
            inner_iters.append(f"{change_iter_names[i]}%{remain_factors[i]}")

        tiling_map_def = f"[{','.join(param_names)}] -> {{ [{','.join(keep_iter_names + change_iter_names)}] -> [{','.join(keep_iter_names + outer_iters + inner_iters)}] }}"
        tiling_map = isl.BasicMap(tiling_map_def)
        # print(f"{tiling_map_def=}")
        tiling_maps.append(tiling_map)

    # import pdb; pdb.set_trace()
    # print("\n")
    tiling_map = tiling_maps[0].intersect_domain(domain)
    for _tiling_map in tiling_maps[1:]:
        tiling_map = tiling_map.apply_range(_tiling_map)

    tiling_map = tiling_map.intersect_domain(domain)
    
    new_operator =  operator.apply_schedule(tiling_map)
    new_operator.history_schedules.append({"tiling_factors":tiling_factors})
    return new_operator

def factorize(N, T, depth=1, path=None, results=None):
    if path is None:
        path = []
    
    if results is None:
        results = []

    if T == 1:
        results.append(path + [N])
        return
    for i in range(1, N+1):
        if N % i == 0:
            factorize(N // i, T - 1, i + 1, path + [i], results)
    return results

def filter_factors(factors):
    """
    filter factor like [1, 1, 4], actually this is not tiling
    """
    new_factors = []
    for factor in factors:
        is_one = [item==1 for item in factor]
        num_one = sum(is_one)
        if num_one >= len(factor) - 1:
            continue
        new_factors.append(factor)
    new_factors.append(None)
    return new_factors


def enumerate_tiling_factors(operator, tiling_factor):
    domain = operator.domain

    assert domain.is_box(), f"domain={domain} is not box"
    n_iter = domain.dim(isl.dim_type.set)

    domain_shape = utils.get_static_box_shape(domain)
    dim_factors = []
    for dim_size in domain_shape:
        factors = factorize(dim_size, tiling_factor)
        factors = filter_factors(factors)
        # factors = [factor for factor in factors if factor[-1]!=1 or max(factor)==1]
        # print(f"{len(factors)=}, {factors=}")
        dim_factors.append(factors)
    
    # exit()
    combination_list = list(itertools.product(*dim_factors))
    import pdb; pdb.set_trace()
    for combination in tqdm(combination_list):
        new_operator = multi_level_tiling(operator, tiling_factor, combination)
        yield new_operator

def pre_tiling_pass(op_list):
    new_op_list = []
    for op in op_list:
        new_op_list.append(op)
        
        for new_op in enumerate_tiling_factors(op, 2):
            new_op_list.append(new_op)
        import pdb; pdb.set_trace()
    # new_op_list = new_op_list[:40]
    # print(len(new_op_list))
    # exit()
    return new_op_list

def multi_level_tiling_outer(operator, tiling_level, tiling_factors, inner_level=5):
    """
    operator = Operator(
        domain=isl.BasicSet("{ [i,j]: 0 <= i < 8 and 0 <= j < 3 }"),
        acc_rel_out=isl.BasicMap("{ [i,j] -> C[i] }"), 
        acc_rel_lhs=isl.BasicMap("{ [i,j] -> A[i + j] }"), 
        acc_rel_rhs=isl.BasicMap("{ [i,j] -> B[i + j] }")
    )

    tiling_factors: 
    [
        [2,2],
        [1,4]
    ]
    """
    param_names = operator.domain.get_var_names(isl.dim_type.param)

    domain=operator.domain

    assert domain.is_box(), f"domain={domain} is not box"
    n_iter = domain.dim(isl.dim_type.set) - inner_level
    assert len(tiling_factors)==n_iter, f"len(tiling_factors)={len(tiling_factors)} != n_iter={n_iter}"

    domain_shape = utils.get_static_box_shape(domain)

    # do checks
    for i in range(n_iter):
        assert len(tiling_factors[i])==tiling_level, f"len(tiling_factors[{i}])={len(tiling_factors[i])} != tiling_level={tiling_level}"
        dim_size = domain_shape[i]
        assert multiply(tiling_factors[i])==dim_size, f"multiply(tiling_factors[{i}])={multiply(tiling_factors[i])} != dim_size={dim_size}"

    # do tiling
    tiling_maps = []
    remain_factors = [*domain_shape]
    for l in range(tiling_level-1):
        n_keep_iters = l * n_iter
        keep_iter_names = [f"i{i}" for i in range(n_keep_iters)]
        change_iter_names = [f"i{i}" for i in range(n_keep_iters, n_keep_iters+n_iter)]
        inner_keep_iter_names = [f"i{i}" for i in range(n_keep_iters+n_iter, n_keep_iters+n_iter+inner_level)]

        outer_iters = []
        inner_iters = []
        for i in range(n_iter):
            factor = tiling_factors[i][l]
            remain_factors[i] = remain_factors[i] // factor
            outer_iters.append(f"floor({change_iter_names[i]}/{remain_factors[i]})")
            inner_iters.append(f"{change_iter_names[i]}%{remain_factors[i]}")

        tiling_map_def = f"[{','.join(param_names)}] -> {{ [{','.join(keep_iter_names + change_iter_names + inner_keep_iter_names)}] -> [{','.join(keep_iter_names + outer_iters + inner_iters + inner_keep_iter_names)}] }}"
        tiling_map = isl.BasicMap(tiling_map_def)
        # print(f"{tiling_map_def=}")
        tiling_maps.append(tiling_map)

   
    # print("\n")
    tiling_map = tiling_maps[0].intersect_domain(domain)
    for _tiling_map in tiling_maps[1:]:
        tiling_map = tiling_map.apply_range(_tiling_map)

    tiling_map = tiling_map.intersect_domain(domain)
    
    new_operator =  operator.apply_schedule(tiling_map, skip_simplify=True)
    return new_operator

def enumerate_tiling_factors_outer(operator, tiling_factor, inner_level=5):
    domain = operator.domain

    assert domain.is_box(), f"domain={domain} is not box"
    n_iter = domain.dim(isl.dim_type.set)

    domain_shape = utils.get_static_box_shape(domain)[:n_iter-inner_level]
    dim_factors = []
    for dim_size in domain_shape:
        factors = factorize(dim_size, tiling_factor)
        # factors = filter_factors(factors)
        factors = [factor for factor in factors if factor[-1]!=1 or max(factor)==1]
        # print(f"{len(factors)=}, {factors=}")
        if len(factors) > 8:
            factors = factors[::4]
        dim_factors.append(factors)
    # import pdb; pdb.set_trace()
    # exit()
    # dim_factors = dim_factors[::4]
    for combination in itertools.product(*dim_factors):
        new_operator = multi_level_tiling_outer(operator, tiling_factor, combination, inner_level)
        yield new_operator

def memory_tiling_pass(op_list):
    new_op_list = []
    for op in op_list:
        new_op_list.append(op)
        for new_op in enumerate_tiling_factors_outer(op, 2, inner_level=5):
            new_op_list.append(new_op)

    return new_op_list