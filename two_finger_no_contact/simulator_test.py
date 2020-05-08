from simulator import FingerSimulator
import numpy as np
import sys, os

sys.path.insert(0, os.getcwd())
from common.trajectory import Trajectory
from ctrl_lcp import LCPtrajopt

A = FingerSimulator()

T = 500
init_state = np.array([np.pi/2, -np.pi/2, np.pi/2, np.pi/2,
                       0.0, 0.0, 0.0, 0.0])

final_state = np.array([np.pi/2, 0, np.pi/2, -np.pi/2,
                       0.0, 0.0, 0.0, 0.0])

u = np.hstack(
    (0.001 * np.ones((500,1)),
     0.001 * np.ones((500,1)), 
     -0.001 * np.ones((500,1)), 
     -0.001 * np.ones((500,1))))


lcp_options = {
    'initial_guess': None,
    'tol': 0.0,
    'solver': 'snopt',
    'contact_max': None,
    'input_max': 0.1}

print("Hi")

#Traj = A.rollout(init_state,u)
Traj = LCPtrajopt(A, init_state, final_state, T, options=lcp_options)
Traj.plot()
print(Traj.compute_time)
A.animate(Traj)



