from simulator import FingerSimulator
import numpy as np
import sys, os

sys.path.insert(0, os.getcwd())
from common.trajectory import Trajectory
from ctrl_test import Testtrajopt

A = FingerSimulator()

T = 100
init_state = np.array([250e-3, 200e-3, 0.0, np.pi/3, -(np.pi + np.pi/3), 
                       0.0, 0.0, 0.0, 0.0, 0.0])

final_state = np.array([250e-3, 150e-3, 0.0, -np.pi/2, -np.pi/2,
                       0.0, 2.0, 0.0, 0.0, 0.0])

u = -0.0 * np.ones((500,2))
#u = np.hstack((-0.02 * np.ones((500,1)), 0.02 * np.ones((500,1))))

Traj = A.rollout(init_state, u)
Traj.plot()
A.animate(Traj)

'''
lcp_options = {
    'initial_guess': None,
    'tol': 0.0,
    'solver': 'snopt',
    'contact_max': None,
    'input_max': None}

Traj = Testtrajopt(A, init_state, final_state, T, options=lcp_options)
Traj.plot()
A.animate(Traj)
'''


