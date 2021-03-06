import simulator
import numpy as np
import matplotlib.pyplot as plt

C = simulator.CartSimulator(gui=True, video=True)
timesteps = 10000
init_state = np.array([1, 3, 4, 0, 0, 0]).astype(np.double)
x_trajectory = np.zeros((6, timesteps))
x_trajectory[:,0] = init_state

lambda_trajectory = np.zeros((2, timesteps))
C.set_state(init_state)

for t in range(timesteps):
    C.step([0.1, -0.1]) # Control input
    x_trajectory[:,t] = C.get_state()
    lambda_trajectory[:,t] = C.get_contact()

C.save_video("test.avi")

plt.figure()
plt.subplot(1,2,1)
plt.plot(range(timesteps), lambda_trajectory[0,:])
plt.plot(range(timesteps), lambda_trajectory[1,:])
plt.legend(['Contact 1', 'Contact 2'])
plt.subplot(1,2,2)
plt.plot(range(timesteps), x_trajectory[0,:])
plt.plot(range(timesteps), x_trajectory[2,:])
plt.plot(range(timesteps), x_trajectory[4,:])
plt.legend(['Cart 1', 'Cart 2', 'Cart 3'])
plt.show()
