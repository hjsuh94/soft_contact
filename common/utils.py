def snopt_status(val):
    if (val == 1):
        status = "optimality conditions satisfied"
    elif (val == 2):
        status = "feasible point found"
    elif (val == 13):
        status = "nonlinear infeasibilities minimized"
    elif (val == 14):
        status = "infeasibilities minimized"
    elif (val == 15):
        status = "infeasible linear constraints in QP subproblem"
    elif (val == 41):
        status = "current point cannot be improved"
    elif (val == 43):
        status = "cannot satisfy the general constraints"
    else:
        status = "unknown"

    return status
