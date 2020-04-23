import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt

T = 1000
h = 0.01

prog = gp.Model("MPCTest")
#prog.params.NonConvex = 2

q = prog.addMVar((T + 1, 1), vtype=GRB.CONTINUOUS, name='q')
v = prog.addMVar((T + 1, 1), vtype=GRB.CONTINUOUS, name='v')
u = prog.addMVar((T, 1), vtype=GRB.CONTINUOUS, name='u')

prog.addConstr(q[0,0] == 1)
prog.addConstr(v[0,0] == 0)
prog.addConstr(q[T,0] == 2)
prog.addConstr(v[T,0] == 0)

objective = 0

for t in range(T):
    objective += u[t] @ u[t]
    prog.addConstr(q[t+1] == (q[t] + h * v[t+1]))
    prog.addConstr(v[t+1] == (v[t] + h * u[t]))
    #prog.addConstr(u[t,0] <= 1.0)
    #prog.addConstr(u[t,0] >= -1.0)

prog.setObjective(objective, GRB.MINIMIZE)
prog.optimize()

plt.figure()
plt.subplot(1,3,1)
plt.plot(range(T+1), q.X)
plt.subplot(1,3,2)
plt.plot(range(T+1), v.X)
plt.subplot(1,3,3)
plt.plot(range(T), u.X)
plt.show()


