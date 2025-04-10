from gurobipy import Model, GRB, quicksum
import numpy as np
from collections import defaultdict

def make_decision_variables(model, n_level):
    # Decision variables
    X = [[model.addVar(vtype=GRB.BINARY, name=f"X_{i}_{j}") for j in range(n_level)] for i in range(n_level)]
    
    L = defaultdict(dict)
    for operand, buffer_list in operand_buffer_mappings.items():
        for buffer in buffer_list:
            _L = [model.addVar(vtype=GRB.BINARY, name=f"L_{operand}_{buffer}_{i}") for i in range(n_level)]
            L[operand][buffer] = _L
    # Update model to integrate new variables
    model.update()
    return X, L

def make_constants(sizes, buffer_sizes, operands_dominate):
    # Construct A^I
    A = dict()
    for operand, dominate in operands_dominate.items():
        A[operand] = [sizes[i] if dominate[i] else 1 for i in range(n_level)]

    log_A = {operand: np.log2(A[operand]) for operand in A.keys()}
    log_S = np.log2(sizes)
    log_B_max = {buffer: np.log2(buffer_sizes[buffer]) for buffer in buffer_sizes.keys()}

    return A, log_A, log_S, log_B_max

def add_constraints(model, X, L, n_level, operand_buffer_mappings):
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

def set_objective(model, X, L, n_level, operand_buffer_mappings, log_A, log_S, log_B_max):
    # Objective function: Traf(I, local)
    obj = 0
    for operand, buffer_list in operand_buffer_mappings.items():
        for buffer in buffer_list:
            _log_A = log_A[operand]
            _L = L[operand][buffer]
            _log_B_max = log_B_max[buffer]
            log_B = quicksum(_log_A[j] * X[i][j] * _L[i] for i in range(n_level) for j in range(n_level))
            model.addConstr(log_B <= _log_B_max, name="log_B_constraint")
            
            log_T = quicksum(log_S[j] * X[i][j] * (1 - _L[i]) for i in range(n_level) for j in range(n_level))
            _log_traffic = log_B + log_T
            obj += _log_traffic
    
    # Set the objective
    model.setObjective(obj, GRB.MINIMIZE)

def extract_results(model, X, L, n_level, operand_buffer_mappings, log_A, log_S, log_B_max):
    X_values = [[X[i][j].X for j in range(n_level)] for i in range(n_level)]
    L_values = defaultdict(dict)
    for operand, buffer_list in operand_buffer_mappings.items():
        for buffer in buffer_list:
            _L_values = [L[operand][buffer][i].X for i in range(n_level)]
            L_values[operand][buffer] = _L_values
    
    log_B_values = defaultdict(dict)
    log_T_values = defaultdict(dict)
    log_traffic_values = defaultdict(dict)
    for operand, buffer_list in operand_buffer_mappings.items():
        for buffer in buffer_list:
            _log_A = log_A[operand]
            _L_values = L_values[operand][buffer]
            _log_B_value = sum(_log_A[j] * X_values[i][j] * _L_values[i] for i in range(n_level) for j in range(n_level))
            log_B_values[operand][buffer] = _log_B_value
            
            _log_T_value = sum(log_S[j] * X_values[i][j] * (1 - _L_values[i]) for i in range(n_level) for j in range(n_level))
            log_T_values[operand][buffer] = _log_T_value
            
            _log_traffic = _log_B_value + _log_T_value
            log_traffic_values[operand][buffer] = _log_traffic

    Traf_value = model.ObjVal
    return X_values, L_values, log_B_values, log_T_values, log_traffic_values, Traf_value

def solve_data_movement(
        n_level, 
        sizes, 
        buffer_sizes, 
        operands_dominate, 
        operand_buffer_mappings):  
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
      
    # Create the model
    model = Model("Minimize_Traf")
    model.setParam('OutputFlag', 0)  # Suppress Gurobi output
    
    X, L = make_decision_variables(model, n_level)
    A, log_A, log_S, log_B_max = make_constants(sizes, buffer_sizes, operands_dominate)
    
    # Constraints for X
    add_constraints(model, X, L, n_level, operand_buffer_mappings)
    
    # Objective function: Traf(I, local)
    set_objective(model, X, L, n_level, operand_buffer_mappings, log_A, log_S, log_B_max)
    
    # Solve the problem
    model.optimize()
    
    # Extract the results
    X_values, L_values, log_B_values, log_T_values, log_traffic_values, Traf_value = extract_results(model, X, L, n_level, operand_buffer_mappings, log_A, log_S, log_B_max)

    return X_values, L_values, log_B_values, log_T_values, log_traffic_values, Traf_value

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
    X_values, L_values, log_B_values, log_T_values, log_traffic_values, Traf_value = solve_mip_gurobi(
        n_level = n_level, 
        sizes = sizes, 
        buffer_sizes = {
            "local": 16,
            "global": 128,
            "in_reg": 1,
            "out_reg": 1,
        },
        operands_dominate = operands_dominate,
        operand_buffer_mappings = operand_buffer_mappings
    )
    X_values = np.array(X_values, dtype=int)
    for operand, buffer_list in operand_buffer_mappings.items():
        for buffer in buffer_list:
            _L_values = L_values[operand][buffer]
            _L_values = np.array(_L_values, dtype=int)
            print(f"{operand} {buffer} L values:\n", _L_values)

    print("X matrix:\n", X_values)
    print("Traf value:", Traf_value)
    # for each operand and buffer, print the log_B, log_T, and log_traffic values
    for operand, buffer_list in operand_buffer_mappings.items():
        print(f"{operand} buffer list:", buffer_list)
        for buffer in buffer_list:
            print(f"    {operand} {buffer} log(B) = {log_B_values[operand][buffer]}, B = {2**log_B_values[operand][buffer]}")
            print(f"    {operand} {buffer} log(T) = {log_T_values[operand][buffer]}, T = {2**log_T_values[operand][buffer]}")
            print(f"    {operand} {buffer} log(Traf) = {log_traffic_values[operand][buffer]}, Traf = {2**log_traffic_values[operand][buffer]}")
