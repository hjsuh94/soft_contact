import simulator
import numpy as np
import matplotlib.pyplot as plt
from ctrl_lcp import LCPtrajopt

C = simulator.CartSimulator()
x = [2,3,4,0,0,0]
xd = [0.5,1.5,2.5,0,0,0]

lcp_options = {
    'initial_guess': None,
    'tol': 5.0,
    'solver': "snopt",
    'contact_max': 10,
    'input_max': 10}

traj = LCPtrajopt(C, x, xd, 500, lcp_options)

tol_list = [3.8, 2.6, 1.4, 0.5, 0.0]

for i in range(len(tol_list)):
    lcp_options['tol'] = tol_list[i]
    lcp_options['initial_guess'] = None
    print(lcp_options)
    traj = LCPtrajopt(C, x, xd, 500, lcp_options)

traj.plot()
print(traj.compute_time)
print(traj.cost)
C.animate(traj)


