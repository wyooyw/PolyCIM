from collections import defaultdict

import numpy as np
from gurobipy import GRB, Model, quicksum


def make_decision_variables(model, n_level, operand_buffer_mappings):
    # Decision variables
    X = [
        [model.addVar(vtype=GRB.BINARY, name=f"X_{i}_{j}") for j in range(n_level)]
        for i in range(n_level)
    ]

    L = defaultdict(dict)
    for operand, buffer_list in operand_buffer_mappings.items():
        for buffer in buffer_list:
            _L = [
                model.addVar(vtype=GRB.BINARY, name=f"L_{operand}_{buffer}_{i}")
                for i in range(n_level)
            ]
            L[operand][buffer] = _L

    log_D = defaultdict(dict)
    for operand, buffer_list in operand_buffer_mappings.items():
        for buffer in buffer_list:
            _log_D = model.addVar(vtype=GRB.CONTINUOUS, name=f"D_{operand}_{buffer}")
            log_D[operand][buffer] = _log_D

    delta = defaultdict(dict)
    for operand, buffer_list in operand_buffer_mappings.items():
        for buffer in buffer_list:
            delta[operand][buffer] = model.addVar(vtype=GRB.BINARY, name=f"delta_{operand}_{buffer}")

    # Update model to integrate new variables
    model.update()
    return X, L, log_D, delta


def make_constants(
    n_level, sizes, buffer_sizes, operands_dominate, operand_base_buffer_size, buffer_bandwidth
):
    # Construct A^I
    A = dict()
    for operand, dominate in operands_dominate.items():
        A[operand] = [sizes[i] if dominate[i] else 1 for i in range(n_level)]

    log_A = {operand: np.log2(A[operand]) for operand in A.keys()}
    log_S = np.log2(sizes)
    log_B_max = {
        buffer: np.log2(buffer_sizes[buffer]) for buffer in buffer_sizes.keys()
    }
    log_operand_base_buffer_size = {
        operand: np.log2(operand_base_buffer_size[operand])
        for operand in operand_base_buffer_size.keys()
    }
    log_buffer_bandwidth = {
        buffer: np.log2(buffer_bandwidth[buffer]) for buffer in buffer_bandwidth.keys()
    }

    return A, log_A, log_S, log_B_max, log_operand_base_buffer_size, log_buffer_bandwidth


def add_constraints(model, X, L, n_level, operand_buffer_mappings, force_innermose_operand, operands_dominate):
    # Constraints for X
    for j in range(n_level):
        model.addConstr(quicksum(X[i][j] for i in range(n_level)) == 1)
    for i in range(n_level):
        model.addConstr(quicksum(X[i][j] for j in range(n_level)) == 1)

    for operand, buffer_list in operand_buffer_mappings.items():
        for buffer in buffer_list:
            _L = L[operand][buffer]
            for i in range(n_level - 1):
                model.addConstr(_L[i] <= _L[i + 1])
        for last_buffer, buffer in zip(buffer_list[:-1], buffer_list[1:]):
            _last_L = L[operand][last_buffer]
            _L = L[operand][buffer]
            for i in range(n_level):
                model.addConstr(_last_L[i] >= _L[i])

    iters_dominate = np.array(operands_dominate[force_innermose_operand]).nonzero()[0]
    L_force = list(L[force_innermose_operand].values())
    # import pdb; pdb.set_trace()
    assert len(L_force) == 1
    L_force = L_force[0]
    for d in iters_dominate:
        for i in range(0, n_level):
            model.addConstr(X[i][d] * L_force[i] == 0)


def set_objective(
    model,
    X,
    L,
    n_level,
    operand_buffer_mappings,
    reduce_levels,
    reduce_operand,
    reduce_sizes,
    reduce_bandwidth,
    log_A,
    S,
    log_S,
    log_B_max,
    log_operand_base_buffer_size,
    log_buffer_bandwidth,
    log_D,
    delta,
):
    # Objective function: Traf(I, local)
    obj = 0
    for operand, buffer_list in operand_buffer_mappings.items():
        for buffer in buffer_list:
            _log_A = log_A[operand]
            _L = L[operand][buffer]
            _log_B_max = log_B_max[buffer]
            log_B = (
                quicksum(
                    _log_A[j] * X[i][j] * _L[i]
                    for i in range(n_level)
                    for j in range(n_level)
                )
                + log_operand_base_buffer_size[operand]
            )
            model.addConstr(log_B <= _log_B_max, name="log_B_constraint")

            _log_W = log_buffer_bandwidth[buffer]
            x = log_B - _log_W
            _log_D = log_D[operand][buffer]
            y = _log_D
            _delta = delta[operand][buffer]
            M = 8192
            # y = max(x, 0)
            model.addConstr(x <= M * (1 - _delta), name="")
            model.addConstr(x >= - M * _delta, name="")
            model.addConstr(y >= 0, name="")
            model.addConstr(y >= x - M * _delta, name="")
            model.addConstr(y <= x + M * _delta, name="")
            model.addConstr(y <= M * (1 - _delta), name="")

            log_T = quicksum(
                log_S[j] * X[i][j] * (1 - _L[i])
                for i in range(n_level)
                for j in range(n_level)
            )
            _log_traffic = _log_D + log_T
            obj += _log_traffic


    # Constraint and objective of SIMD sum.
    # reduce_levels,
    # reduce_operand,
    # reduce_sizes,
    log_S_plus_1 = np.log2(np.array(S)+1)
    log_reduce_sizes = np.log2(reduce_sizes)
    log_reduce_bandwidth = np.log2(reduce_bandwidth)
    
    log_reduce_D = dict()
    for r in reduce_levels:
        _log_D = model.addVar(vtype=GRB.CONTINUOUS, name=f"reduce_D_{r}")
        log_reduce_D[r] = _log_D

    reduce_delta = dict()
    for r in reduce_levels:
        reduce_delta[r] = model.addVar(vtype=GRB.BINARY, name=f"reduce_delta_{r}")

    model.update()
    for i,r in enumerate(reduce_levels):
        R = np.zeros(n_level)
        R[r] = 1
        R_new = []
        for k in range(n_level):
            R_new.append(
                quicksum(
                    X[k][j] * R[j]
                    for j in range(n_level)
                )
            )
        L_reduce = [R_new[0]]
        for k in range(1, n_level):
            L_reduce.append(L_reduce[-1] + R_new[k])
            
        _log_A = log_A[reduce_operand]
        log_B_single = (
            quicksum(
                _log_A[j] * X[i][j] * L_reduce[i]
                for i in range(n_level)
                for j in range(n_level)
            )
            + log_operand_base_buffer_size[reduce_operand]
        )
        log_B_scale = log_S_plus_1[r]
        log_B = log_B_single + log_B_scale
        _log_B_max = log_reduce_sizes[i]
        model.addConstr(log_B <= _log_B_max, name="log_B_constraint")

        _log_W = log_reduce_bandwidth[i]
        x = log_B - _log_W
        _log_D = log_reduce_D[r]
        y = _log_D
        _delta = reduce_delta[r]
        M = 8192
        # y = max(x, 0)
        model.addConstr(x <= M * (1 - _delta), name="")
        model.addConstr(x >= - M * _delta, name="")
        model.addConstr(y >= 0, name="")
        model.addConstr(y >= x - M * _delta, name="")
        model.addConstr(y <= x + M * _delta, name="")
        model.addConstr(y <= M * (1 - _delta), name="")

        log_T_in = quicksum(
            log_S[j] * X[i][j] * (1 - L_reduce[i])
            for i in range(n_level)
            for j in range(n_level)
        )
        log_T_out = log_S[r]
        log_T = log_T_in + log_T_out
        _log_traffic = _log_D + log_T
        obj += _log_traffic

    # Set the objective
    model.setObjective(obj, GRB.MINIMIZE)

    # return obj


def extract_results(
    model,
    X,
    L,
    n_level,
    operand_buffer_mappings,
    log_A,
    S,
    log_S,
    log_B_max,
    log_operand_base_buffer_size,
    log_buffer_bandwidth,
    reduce_levels,
    reduce_operand,
    reduce_bandwidth,
):
    X_values = [[X[i][j].X for j in range(n_level)] for i in range(n_level)]
    X_values = np.array(X_values, dtype=int)
    L_values = defaultdict(dict)
    for operand, buffer_list in operand_buffer_mappings.items():
        for buffer in buffer_list:
            _L_values = [L[operand][buffer][i].X for i in range(n_level)]
            _L_values = np.array(_L_values, dtype=int)
            L_values[operand][buffer] = _L_values

    log_B_values = defaultdict(dict)
    log_D_values = defaultdict(dict)
    log_T_values = defaultdict(dict)
    log_traffic_values = defaultdict(dict)
    for operand, buffer_list in operand_buffer_mappings.items():
        for buffer in buffer_list:
            _log_A = log_A[operand]
            _L_values = L_values[operand][buffer]
            _log_B_value = sum(
                _log_A[j] * X_values[i][j] * _L_values[i]
                for i in range(n_level)
                for j in range(n_level)
            )
            _log_B_value = _log_B_value + log_operand_base_buffer_size[operand]
            log_B_values[operand][buffer] = _log_B_value

            log_W = log_buffer_bandwidth[buffer]
            _log_D_value = max(_log_B_value - log_W, 0)
            log_D_values[operand][buffer] = _log_D_value

            _log_T_value = sum(
                log_S[j] * X_values[i][j] * (1 - _L_values[i])
                for i in range(n_level)
                for j in range(n_level)
            )
            log_T_values[operand][buffer] = _log_T_value

            _log_traffic = _log_D_value + _log_T_value
            log_traffic_values[operand][buffer] = _log_traffic
    
    log_S_plus_1 = np.log2(np.array(S)+1)
    log_reduce_bandwidth = np.log2(reduce_bandwidth)
    log_reduce_B_values = dict()
    log_reduce_B_single_values = dict()
    log_reduce_B_scale_values = dict()
    log_reduce_D_values = dict()
    log_reduce_T_values = dict()
    log_reduce_traffic_values = dict()
    for i,r in enumerate(reduce_levels):
        R = np.zeros(n_level)
        R[r] = 1
        R_new = []
        for k in range(n_level):
            R_new.append(
                sum(
                    X_values[k][j] * R[j]
                    for j in range(n_level)
                )
            )
        L_reduce = [R_new[0]]
        for k in range(1, n_level):
            L_reduce.append(L_reduce[-1] + R_new[k])

        _log_A = log_A[reduce_operand]
        log_B_single = (
            sum(
                _log_A[j] * X_values[i][j] * L_reduce[i]
                for i in range(n_level)
                for j in range(n_level)
            )
            + log_operand_base_buffer_size[reduce_operand]
        )
        # import pdb; pdb.set_trace()
        log_B_scale = log_S_plus_1[r]
        log_B_value = log_B_single + log_B_scale
        log_reduce_B_single_values[r] = log_B_single
        log_reduce_B_scale_values[r] = log_B_scale
        log_reduce_B_values[r] = log_B_value

        log_W = log_reduce_bandwidth[i]
        log_D_value = max(log_B_value - log_W, 0)
        log_reduce_D_values[r] = log_D_value
        
        log_T_in = sum(
            log_S[j] * X_values[i][j] * (1 - L_reduce[i])
            for i in range(n_level)
            for j in range(n_level)
        )
        log_T_out = log_S[r]
        log_T = log_T_in + log_T_out
        log_reduce_T_values[r] = log_T

        log_reduce_traffic = log_D_value + log_T
        log_reduce_traffic_values[r] = log_reduce_traffic

    Traf_value = model.ObjVal
    
    return (
        X_values,
        L_values,
        log_B_values,
        log_D_values,
        log_T_values,
        log_traffic_values,

        log_reduce_B_single_values,
        log_reduce_B_scale_values,
        log_reduce_B_values,
        log_reduce_D_values,
        log_reduce_T_values,
        log_reduce_traffic_values,

        Traf_value,
    )


def show_result(
    X_values,
    L_values,
    log_B_values,
    log_T_values,
    log_traffic_values,
    Traf_value,
    operand_buffer_mappings,
    log_D_values,

    reduce_levels,
    log_reduce_B_single_values,
    log_reduce_B_scale_values,
    log_reduce_B_values,
    log_reduce_D_values,
    log_reduce_T_values,
    log_reduce_traffic_values, 
):
    print("X matrix:\n", X_values)
    print("Traf value:", Traf_value)
    # for each operand and buffer, print the log_B, log_T, and log_traffic values
    for operand, buffer_list in operand_buffer_mappings.items():
        print(f"{operand} buffer list:", buffer_list)
        for buffer in buffer_list:
            print(
                f"    {operand} {buffer} log(B) = {log_B_values[operand][buffer]}, B = {2**log_B_values[operand][buffer]}"
            )
            print(
                f"    {operand} {buffer} log(D) = {log_D_values[operand][buffer]}, D = {2**log_D_values[operand][buffer]}"
            )
            print(
                f"    {operand} {buffer} log(T) = {log_T_values[operand][buffer]}, T = {2**log_T_values[operand][buffer]}"
            )
            print(
                f"    {operand} {buffer} log(Traf) = {log_traffic_values[operand][buffer]}, Traf = {2**log_traffic_values[operand][buffer]}"
            )
    print(f"{reduce_levels=}")
    for i, r in enumerate(reduce_levels):
        print(f"reduce {i}: ")
        print(f"    log(B_single) = {log_reduce_B_single_values[r]}, B = {2**log_reduce_B_single_values[r]}")
        print(f"    log(B_scale) = {log_reduce_B_scale_values[r]}, B = {2**log_reduce_B_scale_values[r]}")
        print(f"    log(B) = {log_reduce_B_values[r]}, B = {2**log_reduce_B_values[r]}")
        print(f"    log(D) = {log_reduce_D_values[r]}, D = {2**log_reduce_D_values[r]}")
        print(f"    log(T) = {log_reduce_T_values[r]}, T = {2**log_reduce_T_values[r]}")
        print(f"    log(Traf) = {log_reduce_traffic_values[r]}, Traf = {2**log_reduce_traffic_values[r]}")


def solve_data_movement(
    n_level,
    sizes,
    buffer_sizes,
    buffer_bandwidth,
    operands_dominate,
    operand_buffer_mappings,
    operand_base_buffer_size,
    reduce_levels,
    reduce_operand,
    reduce_sizes,
    reduce_bandwidth,
    force_innermose_operand,
    show=False,
):
    """
    Example:

    X_values, L_values, log_B_values, log_T_values, log_traffic_values, Traf_value = solve_mip_gurobi(
        n_level = 6,
        sizes = [2, 2, 2, 2, 4, 4],
        buffer_sizes = {
            "local": 16,
            "global": 128,
            "in_reg": 1,
            "out_reg": 1,
        },
        operands_dominate = {
            "I": [True, True, False, False, True, True],
            "O": [False, False, True, True, True, True],
        },
        operand_buffer_mappings = {
            "I": ["local", "in_reg"],
            "O": ["local", "out_reg"],
        }
    )
    """
    # skip global memory
    operand_buffer_mappings = {
        key: value[1:] for key, value in operand_buffer_mappings.items()
    }

    # Create the model
    model = Model("Minimize_Traf")
    model.setParam("OutputFlag", 0)  # Suppress Gurobi output

    X, L, log_D, delta = make_decision_variables(model, n_level, operand_buffer_mappings)
    A, log_A, log_S, log_B_max, log_operand_base_buffer_size, log_buffer_bandwidth = make_constants(
        n_level, sizes, buffer_sizes, operands_dominate, operand_base_buffer_size, buffer_bandwidth
    )

    # Constraints for X
    add_constraints(model, X, L, n_level, operand_buffer_mappings, force_innermose_operand, operands_dominate)

    # Objective function: Traf(I, local)
    set_objective(
        model,
        X,
        L,
        n_level,
        operand_buffer_mappings,
        reduce_levels,
        reduce_operand,
        reduce_sizes,
        reduce_bandwidth,
        log_A,
        sizes,
        log_S,
        log_B_max,
        log_operand_base_buffer_size,
        log_buffer_bandwidth,
        log_D,
        delta,
    )

    # Solve the problem
    model.optimize()

    # Extract the results
    (
        X_values, L_values, log_B_values, log_D_values, log_T_values, log_traffic_values, 
        log_reduce_B_single_values,
        log_reduce_B_scale_values,
        log_reduce_B_values,
        log_reduce_D_values,
        log_reduce_T_values,
        log_reduce_traffic_values, 
        Traf_value
    ) = (
        extract_results(
            model,
            X,
            L,
            n_level,
            operand_buffer_mappings,
            log_A,
            sizes,
            log_S,
            log_B_max,
            log_operand_base_buffer_size,
            log_buffer_bandwidth,

            reduce_levels,
            reduce_operand,
            reduce_bandwidth,
        )
    )

    if show:
        show_result(
            X_values,
            L_values,
            log_B_values,
            log_T_values,
            log_traffic_values,
            Traf_value,
            operand_buffer_mappings,
            log_D_values,

            reduce_levels,
            log_reduce_B_single_values,
            log_reduce_B_scale_values,
            log_reduce_B_values,
            log_reduce_D_values,
            log_reduce_T_values,
            log_reduce_traffic_values, 
        )

    return (
        X_values,
        L_values,
        log_B_values,
        log_T_values,
        log_traffic_values,
        Traf_value,
    )


if __name__ == "__main__":
    # Example usage
    n_level = 6
    sizes = [2, 2, 2, 2, 4, 4]  # Example lengths
    B_max = [2]
    iter_dominates_I = [True, True, False, False, True, True]  # Example dominance

    operands_dominate = {
        "I": [True, True, False, False, True, True],
        "O": [False, False, True, True, True, True],
    }
    operand_buffer_mappings = {
        "I": ["local", "in_reg"],
        "O": ["local", "out_reg"],
    }
    X_values, L_values, log_B_values, log_T_values, log_traffic_values, Traf_value = (
        solve_data_movement(
            n_level=n_level,
            sizes=sizes,
            buffer_sizes={
                "local": 16,
                "global": 128,
                "in_reg": 1,
                "out_reg": 1,
            },
            buffer_bandwidth = {
                "local": 16,
                "global": 8,
                "in_reg": 32,
                "out_reg": 32,
            },
            operands_dominate=operands_dominate,
            operand_buffer_mappings=operand_buffer_mappings,
            operand_base_buffer_size = {
                "I": 1,
                "O": 1
            },
            reduce_levels=[2,3],
            reduce_operand="O",
            reduce_sizes=[16, 16],
            reduce_bandwidth=[32, 32],
            force_innermose_operand="I",
            show=True
        )
    )
    # X_values = np.array(X_values, dtype=int)
    # for operand, buffer_list in operand_buffer_mappings.items():
    #     for buffer in buffer_list:
    #         if buffer == "global":
    #             continue
    #         _L_values = L_values[operand][buffer]
    #         _L_values = np.array(_L_values, dtype=int)
    #         print(f"{operand} {buffer} L values:\n", _L_values)

    # print("X matrix:\n", X_values)
    # print("Traf value:", Traf_value)
    # # for each operand and buffer, print the log_B, log_T, and log_traffic values
    # for operand, buffer_list in operand_buffer_mappings.items():
    #     print(f"{operand} buffer list:", buffer_list)
    #     for buffer in buffer_list:
    #         if buffer == "global":
    #             continue
    #         print(
    #             f"    {operand} {buffer} log(B) = {log_B_values[operand][buffer]}, B = {2**log_B_values[operand][buffer]}"
    #         )
    #         print(
    #             f"    {operand} {buffer} log(T) = {log_T_values[operand][buffer]}, T = {2**log_T_values[operand][buffer]}"
    #         )
    #         print(
    #             f"    {operand} {buffer} log(Traf) = {log_traffic_values[operand][buffer]}, Traf = {2**log_traffic_values[operand][buffer]}"
    #         )
