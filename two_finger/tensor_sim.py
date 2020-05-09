from tkinter import *
import time, os, io
import torch
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from PIL import Image, ImageTk, ImageDraw
import torch.optim as optim
import cv2

class FinSimTensorized():
    def __init__(self, gui=True, video=False):

        self.gui = gui
        self.video = video

        self.xb = torch.zeros((6, 1)).double() # [xb, yb, thetab, dotxb, dotyb, dotthetab]
        self.xl = torch.zeros((4, 1)).double() # [q1, q2, dotq1, dotq2]
        self.xr = torch.zeros((4, 1)).double() # [q1, q2, dotq1, dotq2]
        self.x = torch.cat((self.xb, self.xl, self.xr))

        self.pb = torch.zeros((2, 1)).double() # stores body coordinates, repetitive with xb
        self.pb_l = torch.zeros((2, 1)).double() # stores base joint coordinate                
        self.p1_l = torch.zeros((2, 1)).double() # stores first joint coordinate of left
        self.p2_l = torch.zeros((2, 1)).double() # stores second joint coordinate of left
        self.pb_r = torch.zeros((2, 1)).double() # stores base joint coordinate
        self.p1_r = torch.zeros((2, 1)).double() # stores first joint coordinate of right
        self.p2_r = torch.zeros((2, 1)).double() # stores second joint coordinate of right

        self.p = torch.cat((self.pb, self.p1_l, self.p2_l, self.p1_r, self.p2_r))
        
        self.contact_l = torch.zeros((2, 1)).double() # [lambda_x, lambda_y]
        self.contact_r = torch.zeros((2, 1)).double() # [lambda_x, lambda_y]
        self.contact = torch.cat((self.contact_l, self.contact_r))

        # follows SI units
        self.params = {'m': 0.1,  # mass of each finger
                       'mb': 0.5, # mass of the manipuland
                       'cb': 1.0, # damping on the manipuland
                       'R': 50e-3, # radius of the manipuland
                       'Lx': 130e-3, # length between center frame and base of finger
                       'Ly': 0, # length between center frame and base of finger
                       'Cx': 250e-3, # center frame location
                       'Cy': 250e-3, # center frame location
                       'l': 100e-3, # length of each finger
                       'w': 20e-3, # width of each finger
                       'k': 1e2, # spring constant on the fingertip
                       'h': 0.01, # time step
                        'g': 0.0 } # gravity

        self.update_params()
        
        self.step_count = 0

        if(self.gui):
            # GUI Parameters
            self.width = 500
            self.height = 500
            self.tk = Tk()
            self.tk.title("2 Finger Simulation")
            self.canvas = Canvas(self.tk, width=self.width, height=self.height, bg='white')
            self.canvas.pack()

            self.img_array = []
            
    def get_state(self, body=None):
        if (body == 'b'):
            return self.xb
        elif (body == 'l'):
            return self.xl
        elif (body == 'r'):
            return self.xr
        else:
            return self.x

    def get_contact(self):
        return self.contact

    def repack_state(self):
        assert self.xb.size() == (6,1)
        assert self.xl.size() == (4,1)
        assert self.xr.size() == (4,1)
        assert self.pb.size() == (2,1)
        assert self.pb_l.size() == (2,1)
        assert self.p1_l.size() == (2,1)
        assert self.p2_l.size() == (2,1)
        assert self.pb_r.size() == (2,1)
        assert self.p1_r.size() == (2,1)
        assert self.p2_r.size() == (2,1)
        assert self.contact_l.size() == (2,1)
        assert self.contact_r.size() == (2,1)
        self.x = torch.cat((self.xb, self.xl, self.xr))
        self.p = torch.cat((self.pb, self.p1_l, self.p2_l, self.p1_r, self.p2_r))        
        self.contact = torch.cat((self.contact_l, self.contact_r))

    def update_params(self):
        # auto-generated self parameters based on manual input
        self.params['r'] = 0.5 * self.params['w']
        self.params['rl'] = 0.5 * self.params['l']
        self.params['I'] = (1/12) * self.params['m'] * (
            self.params['l'] ** 2.0 + self.params['w'] ** 2.0)
        self.params['Ib'] = (np.pi / 2) * self.params['mb'] * self.params['R'] ** 2.0
        self.params['alpha'] = 2. * self.params['I'] + 6. * self.params['m'] * (
            self.params['rl'] ** 2.0)
        self.params['beta'] = self.params['m'] * self.params['l'] * self.params['rl']
        self.params['delta'] = self.params['I'] + self.params['m'] * (self.params['rl'] ** 2.0)

        self.pb_l = torch.tensor([[self.params['Cx'] - self.params['Lx']],
                                [self.params['Cy'] + self.params['Ly']]]).double()
        self.pb_r = torch.tensor([[self.params['Cx'] + self.params['Lx']],
                                [self.params['Cy'] + self.params['Ly']]]).double()

    def set_state(self, x, body=None):
        if (body == 'l'):
            self.xl = x
        elif (body == 'r'):
            self.xr = x
        elif (body == 'b'):
            self.xb = x
        else:
            self.x = x
            self.xb = x[0:6].view((6,1))
            self.xl = x[6:10].view((4,1))
            self.xr = x[10:14].view((4,1))
        self.repack_state()

    def set_parameters(self, param_name, param_value):
        self.params[param_name] = param_value
        self.update_params()

    #---------------------------------------------------------------------------
    # Dynamics Section
    #---------------------------------------------------------------------------

    def fkin(self, q, body):
        ''' Receives a vector of joints and returns forward kinematics.
            [NOTE]: This function also stores joint positions into position storage. 
                    DO NOT call for simple calculation of forward kinematics, 
                    access self.p instead.'''

        q1,q2 = q # Unpack joints 

        # assume q1, q2 are torch tensors
        x1 = self.params['l'] * (torch.cos(q1))
        y1 = self.params['l'] * (torch.sin(q1))
        x2 = x1 + self.params['l'] * torch.cos(q1 + q2)
        y2 = y1 + self.params['l'] * torch.sin(q1 + q2)

        y1 = y1 + self.params['Cy'] + self.params['Ly']
        y2 = y2 + self.params['Cy'] + self.params['Ly']
        
        if (body == 'l'):
            x1 = x1 + self.params['Cx'] - self.params['Lx']
            x2 = x2 + self.params['Cx'] - self.params['Lx']       
            self.p1_l = torch.stack((x1, y1))
            self.p2_l = torch.stack((x2, y2))
        elif (body == 'r'):
            x1 = x1 + self.params['Cx'] + self.params['Lx']
            x2 = x2 + self.params['Cx'] + self.params['Lx']            
            self.p1_r = torch.stack((x1, y1))
            self.p2_r = torch.stack((x2, y2))
            
        else:
            raise ValueError('Body should only be "l" or "r"')
        
        return torch.stack((x2,y2))

    def jacobian(self, q):
        ''' Receives a vector of joints and returns the Jacobian by lienarizing
            the forward kinematics '''
        q1,q2 = q # Unpack Joints
        J = torch.tensor([[-self.params['l'] * (torch.sin(q1) + torch.sin(q1 + q2)),
                      -self.params['l'] * (torch.sin(q1 + q2))],
                     [self.params['l'] * (torch.cos(q1) + torch.cos(q1 + q2)),
                      self.params['l'] * (torch.cos(q1 + q2))]]).double()

        assert J.size() == (2,2)
        return J #np.squeeze(J)

    def manipulator_params(self, x):
        ''' Receives a vector of joints and joint velocities, and returns the 
            matrices in the manipulator equation'''
        q1,q2,q1dot,q2dot = x # Unpack joints

        M = torch.tensor([[self.params['alpha'] + 2 * self.params['beta'] * torch.cos(q2),
                       self.params['delta'] + self.params['beta'] * torch.cos(q2)],
                      [self.params['delta'] + self.params['beta'] * torch.cos(q2),
                       self.params['delta']]]).double()

        C = torch.tensor([[-self.params['beta'] * torch.sin(q2) * q2dot,
                       -self.params['beta'] * torch.sin(q2) * (q1dot + q2dot)],
                      [self.params['beta'] * torch.sin(q2) * q1dot,
                       0]]).double()
        assert M.size() == (2,2)
        assert C.size() == (2,2) 
        return M, C #np.squeeze(M), np.squeeze(C)

    def step(self, tau):
        ''' Semi-implicit time-stepping for dynamics integration. Does the following:
            1. Compute contact forces at time (t) from joint configuration at time (t)
            2. Compute velocities at time (t+1) from contact forces at time (t)
            3. Compute joint configuration at time (t+1) from velocities at time (t+1) '''
        
        # Unpack Joints and torques--------------------------------------------
        x, y, theta, vx, vy, omega = self.xb
        q1_l, q2_l, q1dot_l, q2dot_l = self.xl
        q1_r, q2_r, q1dot_r, q2dot_r = self.xr
        tau1_l, tau2_l, tau1_r, tau2_r = tau
        # print(tau1_l.size(), tau2_r.size())
        # print(tau)

        # Compute forward kinematics--------------------------------------------
        pl = self.fkin([q1_l, q2_l], 'l')
        pr = self.fkin([q1_r, q2_r], 'r')
        pb = torch.stack((x,y)).double()
        self.pb = pb

        # Compute contact forces------------------------------------------------
        dist_l = torch.norm(pl - pb) 
        dist_r = torch.norm(pr - pb)

        # print(pl.size(), pr.size(), pb.size())

        # Note: these contact forces are applied on the ball. Take negative to apply to finger.
        lambda_l = -self.params['k'] * ((pl - pb) / dist_l) * \
                   torch.nn.ReLU()(self.params['R'] + self.params['r'] - dist_l).double()
        lambda_r = -self.params['k'] * ((pr - pb) / dist_r) * \
                   torch.nn.ReLU()(self.params['R'] + self.params['r'] - dist_r).double()

        self.contact_l = lambda_l
        self.contact_r = lambda_r
        
        # Propagate Velocities--------------------------------------------------

        # Update manipuland velocity 
        # v = torch.cat((vx, vy, omega))
        # print(lambda_l.size(), lambda_r.size())
        self.xb[3:5] = self.xb[3:5] + self.params['h'] * (1. / self.params['mb']) * (
            lambda_l + lambda_r - self.params['cb'] * self.xb[3:5])
        self.xb[5] = self.xb[5] + self.params['h'] * (1./self.params['Ib']) * 0.
        # add gravity
        self.xb[4] = self.xb[4] + self.params['g'] * self.params['h']
        
        # Update left manipulator velocity
        Ml, Cl = self.manipulator_params(self.xl)
        Jl = self.jacobian(torch.cat((q1_l, q2_l)))
        # print(tau1_l.size(), tau2_l.size())
        # print(torch.mm(Cl, self.xl[2:4]).size(), 
                # torch.stack((tau1_l, tau2_l)).size(),  
                # torch.mm(torch.transpose(Jl, 0, 1), lambda_l).size())
        self.xl[2:4] = self.xl[2:4] + self.params['h'] * torch.inverse(Ml).mm(
            -torch.mm(Cl, self.xl[2:4]) + torch.stack((tau1_l, tau2_l))
            -torch.mm(torch.transpose(Jl, 0, 1), lambda_l))

        # Update right manipulator velocity 
        Mr, Cr = self.manipulator_params(self.xr)
        Jr = self.jacobian(torch.cat((q1_r, q2_r)))
        self.xr[2:4] = self.xr[2:4] + self.params['h'] * torch.inverse(Mr).mm(
            -torch.mm(Cr, self.xr[2:4]) + torch.stack((tau1_r, tau2_r))
            -torch.mm(torch.transpose(Jr, 0, 1), lambda_r))

        # Propagate positions---------------------------------------------------
        self.xb[0:3] = self.xb[0:3] + self.params['h'] * self.xb[3:6]
        self.xl[0:2] = self.xl[0:2] + self.params['h'] * self.xl[2:4]
        self.xr[0:2] = self.xr[0:2] + self.params['h'] * self.xr[2:4]

        self.repack_state()

        self.step_count = self.step_count + 1
        # print(self.x.size(), self.p.size(), self.contact.size())
        return self.x, self.p, self.contact


def simulate(simulator, init_state, inputs):
    # recorder trajectories
    timesteps = inputs.shape[1]
    x_trajectory = np.zeros((14, timesteps))
    x_trajectory[:,0] = init_state
    lambda_trajectory = np.zeros((4, timesteps))
    # forward simulate
    simulator.set_state(init_state)
    for t in range(inputs.shape[1]):
        simulator.step(inputs[:, t]) # Control input
        x_trajectory[:,t] = simulator.get_state()
        lambda_trajectory[:,t] = simulator.get_contact()
    return x_trajectory, lambda_trajectory


torch.autograd.set_detect_anomaly(True)

# hyper params set 1
timesteps = 1000
init_state = torch.tensor([250e-3, 200e-3, 0.0, 0.0, 0.0, 0, # manipuland pose 
                       0, -3*np.pi/2, 0, 0, # left finger joint pose
                       np.pi/2, np.pi/2, 0, 0]).double() # right finger joint pose
goal_state = torch.tensor([250e-3, 250e-3]).view(2,1) # manipuland deisired location


# training hyper parameters
lr = 50
epochs = 10
input_loss_weight = 0 #0.000005

# training stepup
guess_inputs = np.zeros((4, timesteps))
guess_inputs = torch.tensor(guess_inputs, requires_grad=True, dtype=torch.double)
optimizer = optim.SGD([guess_inputs], lr=lr)
losses = np.zeros(epochs)
#record time
since = time.time()

# start training 
for epoch in range(epochs):
    CSGD = FinSimTensorized(timesteps)
    CSGD.set_state(init_state)

    for i in range(guess_inputs.size(1)):
        x, p, contact = CSGD.step(guess_inputs[:, [i]])
    # terminal loss
    final_state = p[:2]
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



























# # Example usage. Will be gone soon and separated into another file.
# C = FingerSimulator(gui=True, video=True)
# timesteps = 1000
# init_state = np.array([250e-3, 200e-3, 0.0, 0.0, 0.0, 0, # manipuland pose 
#                        0, -3*np.pi/2, 0, 0, # left finger joint pose
#                        np.pi/2, np.pi/2, 0, 0]).astype(np.double) # right finger joint pose
# x_trajectory = np.zeros((14, timesteps))
# x_trajectory[:,0] = init_state
# lambda_trajectory = np.zeros((4, timesteps))
# C.set_state(init_state)

# # Try out a simple Cartesian PD Controller!
# kp = 3.0
# kd = 1.0
# p_l_past = 0.0
# p_r_past = 0.0

# for t in range(timesteps):
#     x_trajectory[:,t] = C.get_state().ravel()
#     lambda_trajectory[:,t] = C.get_contact().ravel()

#     pd_l = np.array([[220e-3], [300e-3 - t * 3e-5]])
#     pd_r = np.array([[280e-3], [300e-3 - t * 3e-5]])

#     q1_l, q2_l, q1dot_l, q2dot_l = C.get_state(body='l')
#     q1_r, q2_r, q1dot_r, q2dot_r = C.get_state(body='r')

#     J_l = C.jacobian(np.array([q1_l, q2_l]))
#     J_r = C.jacobian(np.array([q1_r, q2_r]))
#     p_l = C.fkin(np.array([q1_l, q2_l]), body='l')
#     p_r = C.fkin(np.array([q1_r, q2_r]), body='r')

#     ul = np.dot(np.transpose(J_l), (-kp * (p_l - pd_l) - kd * (p_l - p_l_past) / C.params['h']))
#     ur = np.dot(np.transpose(J_r), (-kp * (p_r - pd_r) - kd * (p_r - p_r_past) / C.params['h']))

#     u = np.array([ul[0], ul[1], ur[0], ur[1]])

#     # Simulate saturation
#     max_torque = 5.0
#     u[u > max_torque] = max_torque
    
#     C.step(u)

#     p_l_past = p_l
#     p_r_past = p_r

#     if t % 1000 == 0:
#         print(p_l)
#         # print(C.get_state('r'))
# # print(C.get_state('b'))

# C.save_video("test.avi")
