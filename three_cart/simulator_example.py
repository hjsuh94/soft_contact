import simulator
import numpy as np
import matplotlib.pyplot as plt

C = simulator.CartSimulator()
traj = C.rollout([1,3,4,0,0,0], 5.0 * np.ones((500,2)))
traj.plot()
C.animate(traj)


