import simulator
import numpy as np
import matplotlib.pyplot as plt
# from ctrl_lcp import LCPtrajopt
# from ctrl_smooth import Smoothtrajopt
from ctrl_sgd import SGDtrajopt

C = simulator.CartSimulator()
x = [1,3,4,0,0,0]
xd = [1.5,2.5,3.5,0,0,0]

lcp_options = {
    'initial_guess': None,
    'tol': 5.0,
    'solver': "snopt",
    'contact_max': None,
    'input_max': None}

smooth_options = {
    'initial_guess': None,
    'tol': 0.01,
    'solver': "snopt",
    'contact_max': None,
    'input_max': None}

sgd_options = {
	'initial_guess': None,
	'lr': None,
	'epochs': None,
	'u_lambda': None,
	'params': None,
	'input_max': None}


traj = None
sol = None
'''
tol_list = np.linspace(4.0, 0.0, 5)

for i in range(len(tol_list)):
    lcp_options['tol'] = tol_list[i]
    lcp_options['initial_guess'] = traj
    traj = LCPtrajopt(C, x, xd, 500, lcp_options)
    print(traj.compute_time)
'''
# # tol_list = [1e-3, 1e-4, 1e-5]

# for i in range(len(tol_list)):
#     smooth_options['tol'] = tol_list[i]
#     smooth_options['initial_guess'] = traj
#     traj = Smoothtrajopt(C, x, xd, 500, smooth_options)
#     print(traj.compute_time)


# inputs = np.ones((500, 2))
# inputs[:, 1] = -1


# traj = C.rollout(x, inputs)

traj = SGDtrajopt(C, x, xd, 500, sgd_options)

    
# traj.plot()
#C.animate(traj)

# lcp_options['tol'] = 0.0
# lcp_options['initial_guess'] = traj
# print(lcp_options)
# traj = LCPtrajopt(C, x, xd, 500, lcp_options)
# print(traj.compute_time)

C.animate(traj)
print(traj.compute_time)
print(traj.cost)
traj.plot()


