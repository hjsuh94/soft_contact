import time 
import torch 
import numpy as np
import simulator as sim
import matplotlib.pyplot as plt
import torch.optim as optim
from common.trajectory import Trajectory 
torch.set_printoptions(precision=3, sci_mode=False)
np.set_printoptions(precision=3, suppress=True)



def SGDtrajopt(sim, xi, xf, T, options=None):
    '''
    Function that does trajectory optimization with SGD.

    Options should be a dictionary containing the following:
    - initial_guess : initial guess to be used (torch tensor of size (T, 2))
    - lr            : learning rate, default 50
    - epochs        : epoch number, default 5
    - u_lambda      : weight on u traj costs, defaut 0.000005
    - input_max     : maximum value for input. Defaults to None. NOT IMPLEMENTED YET
    - params        : params to update sim params
    '''
    #---------------------------------------------------------------------------
    # Define helper functions here
    #---------------------------------------------------------------------------

    # build a tensor graph of forward integration
    def forward(xi, u_traj, T):
        x_traj = [xi]
        for t in range(T):
            # Compute contact forces first
            x = x_traj[t] # current state (x0, x1, x2, v0, v1, v2)
            f1 = torch.nn.ReLU()(-sim.params['k'] * (x[1] - x[0] - sim.params['d']))
            f2 = torch.nn.ReLU()(-sim.params['k'] * (x[2] - x[1] - sim.params['d']))

            # Propagate velocities 
            v0 = x[3] + sim.params['h'] * (-sim.params['c'] * x[3] - f1 + u_traj[0, t])
            v1 = x[4] + sim.params['h'] * (-sim.params['c'] * x[4] + f1 - f2)
            v2 = x[5] + sim.params['h'] * (-sim.params['c'] * x[5] + f2 + u_traj[1, t])

            # Propagate positions at t with velocities at t+1
            x0 = x[0] + sim.params['h'] * v0
            x1 = x[1] + sim.params['h'] * v1
            x2 = x[2] + sim.params['h'] * v2

            # repack states
            x = torch.stack([x0, x1, x2, v0, v1, v2]).view(6, 1)
            x_traj.append(x)

        return x_traj     

    def sgd(xi, xf, u_traj, u_lambda, T, lr, epochs):
        # training stepup
        u_traj = u_traj.clone().detach()
        u_traj.requires_grad = True
        optimizer = optim.SGD([u_traj], lr=lr)
        costs = np.zeros(epochs)
        #record time
        since = time.time()

        # start training 
        for epoch in range(epochs):
            x_traj = forward(xi.clone().detach(), u_traj, T)
            assert len(x_traj) == T+1
            # terminal L1 cost
            x_cost = ((x_traj[-1] - xf)**2).mean()
            # input L2 regulation
            u_cost = torch.sum(u_traj * u_traj)
            # total cost
            cost = x_cost + u_lambda*u_cost
            cost.backward()
            optimizer.step()
            optimizer.zero_grad()
            costs[epoch] = cost.data.numpy()
            if False :
                print('cost: ', '{:.3f}'.format(cost.data.numpy()))
        time_elapsed = time.time() - since
        print('Training lasts {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        return u_traj, time_elapsed

    #---------------------------------------------------------------------------
    # Set default parameters options.
    #---------------------------------------------------------------------------

    if options['initial_guess']: # assume to be (T, 2) numpy array
        u_traj = torch.tensor(options['initial_guess']).view(T, 2).double().T
    else:
        u_traj = torch.zeros(2, T).double() 

    if options['lr']:
        lr = options['lr']
    else:
        lr = 500

    if options['epochs']:
        epochs = options['epochs']
    else:
        epochs = 50

    if options['u_lambda']:
        u_lambda = options['u_lambda']
    else: 
        u_lambda = 0.00005

    if options['input_max']:
        input_max = options['input_max']
    else:
        input_max = None

    if options['params']:
        for key, value in options['params'].items():
            print(key, value)
            sim.set_parameters(key, value)

    #---------------------------------------------------------------------------
    # Solve the optimization
    #---------------------------------------------------------------------------
    xi = torch.tensor(xi).view(6, 1).double() # initial state
    xf = torch.tensor(xf).view(6, 1).double() # goal state
    u_traj, time_elapsed = sgd(xi, xf, u_traj, u_lambda, T, lr, epochs)
    # plt.plot(range(epochs), costs)
    # plt.show()

    traj = sim.rollout(xi.flatten().numpy(), u_traj.detach().numpy().T)
    traj.compute_time = time_elapsed

    return traj



