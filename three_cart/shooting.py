import numpy as np
import importlib
import simulator as sim
np.set_printoptions(precision=3, suppress=True)
import matplotlib.pyplot as plt
import torch 
from torch.autograd import Variable
import torch.optim as optim
import time 
torch.set_printoptions(precision=3, sci_mode=False)


# tensorized version of cart simulator class
class CartSimTensorized(sim.CartSimulator):
    def __init__(self, timesteps, params):
        super(CartSimTensorized, self).__init__(gui=False, video=False)
        self.timesteps = timesteps
        self.goal_state = torch.zeros(6).double()
        self.states = torch.zeros(6, self.timesteps+1).double()
        self.forces = torch.zeros(2, self.timesteps).double()
        for key, value in params.items():
            self.set_parameters(key, value)

    def set_states(self, state, t):
        assert len(state) == 6
        self.states[:, t] = torch.tensor(state).double() #automatically creates a copy

    # Semi-implicit time-stepping for dynamics integration
    def forward_pass(self, inputs):
        # forward integrate to obtain a target state
        for t in range(self.timesteps):
            # Compute contact forces first
            lambda1 = torch.nn.ReLU()(-self.params['k'] * (
                self.states[1,t] - self.states[0,t] - self.params['d']))
            lambda2 = torch.nn.ReLU()(-self.params['k'] * (
                self.states[2,t] - self.states[1,t] - self.params['d']))
            self.forces[:,t] = torch.stack([lambda1, lambda2]).double().detach()

            # Propagate velocities ### EXAMINE HERE IF LAMBDA1 CAN BE USED
            self.states[3,t+1] = self.states[3,t] + self.params['h'] * \
                                (-self.params['c'] * self.states[3,t] - lambda1 + inputs[0,t])
            self.states[4,t+1] = self.states[4,t] + self.params['h'] * \
                                (-self.params['c'] * self.states[4,t] + lambda1 - lambda2)
            self.states[5,t+1] = self.states[5,t] + self.params['h'] * \
                                (-self.params['c'] * self.states[5,t] + lambda2 + inputs[1,t])

            # Propagate positions at t with velocities at t+1
            self.states[0:3,t+1] = self.states[0:3,t] + self.params['h'] * self.states[3:6,t+1]
            
        return self.states[:, -1] 


# simulate using inputs found by optimization
def simulate(simulator, init_state, inputs):
    # recorder trajectories
    timesteps = inputs.shape[1]
    x_trajectory = np.zeros((6, timesteps))
    x_trajectory[:,0] = init_state
    lambda_trajectory = np.zeros((2, timesteps))
    # forward simulate
    simulator.set_state(init_state)
    for t in range(inputs.shape[1]):
        simulator.step(inputs[:, t]) # Control input
        x_trajectory[:,t] = simulator.get_state()
        lambda_trajectory[:,t] = simulator.get_contact()
    return x_trajectory, lambda_trajectory



# hyper params 
timesteps = 10000
init_state = np.array([1, 3, 4, 0, 0, 0]).astype(np.double)
inputs = np.array([[10, -10] for i in range(timesteps)]).T
params = {'m': 1.,  # mass
          'c': 3.0,  # viscosity
          'k': 100.0,  # elasticity
          'd': 0.5,    # cart length
          'w': 0.5,    # cart height (just for visualization) 
          'h': 0.01,   # time step
          't': 0.0}   # sleep time before redrawing on canvas


# set initial initial state, initial guess input traj, and goal state
init_state = np.array([1, 3, 4, 0, 0, 0]).astype(np.double)
guess_inputs = np.array([[0, 0] for i in range(timesteps)]).T 
center_cart_goal_x = 1.5
goal_state = torch.tensor([center_cart_goal_x - params['d'], 0,
						   center_cart_goal_x,               0,
						   center_cart_goal_x + params['d'], 0])


# training hyper params
lr = 50
epochs = 5
input_loss_weight = 0.000005

# training stepup
guess_inputs = torch.tensor(guess_inputs, requires_grad=True, dtype=torch.double)
optimizer = optim.SGD([guess_inputs], lr=lr)
losses = np.zeros(epochs)
#record time
since = time.time()

# start training 
for epoch in range(epochs):
    CSGD = CartSimTensorized(timesteps, params)
    CSGD.set_states(init_state, 0)
    final_state = CSGD.forward_pass(guess_inputs)
    # terminal loss
    terminal_loss = ((final_state - goal_state)**2).mean()
    # input L2 regulation
    input_loss = torch.sum(guess_inputs * guess_inputs)
    # total loss
    loss = terminal_loss + input_loss_weight*input_loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    losses[epoch] = loss.data.numpy()
    if True :
        print('loss: ', '{:.3f}'.format(loss.data.numpy()))
time_elapsed = time.time() - since
print('Training takes in {:.0f}m {:.0f}s'.format(time_elapsed // 60,
                                                time_elapsed % 60))

guess_inputs = guess_inputs.cpu()


# visualize simulation
C = sim.CartSimulator(gui=True, video=True)
for key, value in params.items():
    C.set_parameters(key, value)
_, _ = simulate(C, init_state, guess_inputs)
C.save_video("new_sim.avi")


# now plot the trajectory
plt.figure(figsize=(15, 10))
plt.subplot(2,2,1)
plt.plot(range(timesteps), CSGD.forces[0,:])
plt.plot(range(timesteps), CSGD.forces[1,:])
plt.legend(['Contact 1', 'Contact 2'])
plt.subplot(2,2,2)
plt.plot(range(timesteps+1), CSGD.states[0,:].detach())
plt.plot(range(timesteps+1), CSGD.states[2,:].detach())
plt.plot(range(timesteps+1), CSGD.states[4,:].detach())
plt.legend(['Cart 1', 'Cart 2', 'Cart 3'])
plt.subplot(2,2,3)
plt.plot(range(timesteps), guess_inputs[0,:].detach())
plt.plot(range(timesteps), guess_inputs[1,:].detach())
plt.legend(['input 1', 'input 2'])
plt.subplot(2,2,4)
plt.plot(range(epochs), losses)
plt.legend(['loss evolution'])
plt.show()
