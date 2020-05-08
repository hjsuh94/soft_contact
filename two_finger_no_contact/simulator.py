from tkinter import *
import time, os, io, sys
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from PIL import Image, ImageTk, ImageDraw
import cv2

sys.path.insert(0, os.getcwd())
from common.trajectory import Trajectory

class FingerSimulator():
    '''
    Finger Simulator without contact. 

    State definition: [q1_l, q2_l, q2_r, q2_r]
    Input definition: [u1_l, u2_l, u1_r, u2_r]

    where _b subscript denotes ball coordinates (SE(2) pose and velocity)
          _l subscript denotes left manipulator coordinate (joint position and velocity)
          _r subscript denotes right manipulator coordinate (joint position and velocity)

    NOTE: Simulator does not contain any 'internal states'. 
    '''
    
    def __init__(self, gui=True):

        self.gui = gui

        self.nq = 4
        self.nv = 4 # Alias for self.nq
        self.nu = 4
        self.nf = 0

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
                       'h': 0.01} # time step

        self.update_params()

    '''
    Parameter Section---------------------------------------------------------------
    '''


    def set_parameter(self, param_name, param_value):
        '''
        Set parameters of the simulator according to param_name and value
        '''
        self.params[param_name] = param_value
        self.update_params()

    def set_parameters(self, param_dict):
        '''
        Set parameters of the simulator with a dict. Useful for changing params with
        entire dictionary
        '''
        for key, value in param_dict.items():
            self.set_parameter(key, value)
            
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

        # Cartesian position of base joints for visualization
        self.pb_l = np.array([[self.params['Cx'] - self.params['Lx']],
                              [self.params['Cy'] + self.params['Ly']]]).astype(np.double)
        self.pb_r = np.array([[self.params['Cx'] + self.params['Lx']],
                              [self.params['Cy'] + self.params['Ly']]]).astype(np.double)
    '''
    Dynamics Section---------------------------------------------------------------
    '''

    def fkin(self, q, body):
        ''' 
        Receives a vector of joints and returns forward kinematics.
        body can be one of the following:
          - l1: return first joint of left arm
          - l2: return second joint of left arm
          - r1: return first joint of right arm
          - r2: return second joint of right arm
        '''

        q1,q2,_,_ = q # Unpack joints

        if (body[0] == 'l'):
            x0 = self.params['Cx'] - self.params['Lx']
        elif (body[0] == 'r'):
            x0 = self.params['Cx'] + self.params['Lx']
        else:
            raise ValueError('Body should only be "l0,l1,l2,r0,r1,r2"')

        y0 = self.params['Cy'] + self.params['Ly']

        if (body[1] == '0'):
            return np.array([x0, y0])
        
        elif (body[1] == '1'):
            x1 = x0 + self.params['l'] * (np.cos(q1))
            y1 = y0 + self.params['l'] * (np.sin(q1))

            return np.array([x1, y1])
        
        elif (body[1] == '2'):
            x1 = x0 + self.params['l'] * (np.cos(q1))
            y1 = y0 + self.params['l'] * (np.sin(q1))
            x2 = x1 + self.params['l'] * np.cos(q1 + q2)
            y2 = y1 + self.params['l'] * np.sin(q1 + q2)
            
            return np.array([x2, y2])
        else:
            raise ValueError('Body should only be "l0,l1,l2,r0,r1,r2"')
        
    def jacobian(self, x):
        ''' Receives a vector of joints and returns the Jacobian by lienarizing
            the forward kinematics '''
        q1,q2,_,_ = x # Unpack Joints
        J = np.array([[-self.params['l'] * (np.sin(q1) + np.sin(q1 + q2)),
                       -self.params['l'] * (np.sin(q1 + q2))],
                      [self.params['l'] * (np.cos(q1) + np.cos(q1 + q2)),
                       self.params['l'] * (np.cos(q1 + q2))]]).astype(np.double)

        return np.squeeze(J)

    def manipulator_params(self, x):
        ''' Receives a vector of joints and joint velocities, and returns the 
            matrices in the manipulator equation'''
        q1,q2,q1dot,q2dot = x # Unpack joints

        M = np.array([[self.params['alpha'] + 2 * self.params['beta'] * np.cos(q2),
                       self.params['delta'] + self.params['beta'] * np.cos(q2)],
                      [self.params['delta'] + self.params['beta'] * np.cos(q2),
                       self.params['delta']]]).astype(np.double)

        C = np.array([[-self.params['beta'] * np.sin(q2) * q2dot,
                       -self.params['beta'] * np.sin(q2) * (q1dot + q2dot)],
                      [self.params['beta'] * np.sin(q2) * q1dot,
                       0]]).astype(np.double)

        return np.squeeze(M), np.squeeze(C)

    def unpack_states(self, x):
        '''
        Unpack the total state of the system to decoupled representation.
        Returns state of body, state of left finger, state of right finger.
        '''

        q1_l, q2_l, q1_r, q2_r = x[0:self.nq]
        q1dot_l, q2dot_l, q1dot_r, q2dot_r = x[self.nq:(2*self.nq)]

        x_l = np.array([q1_l, q2_l, q1dot_l, q2dot_l])
        x_r = np.array([q1_r, q2_r, q1dot_r, q2dot_r])

        return x_l, x_r

    def pack_states(self, x_l, x_r):
        ''' 
        Packs decoupled states of individual bodies into total system state-space.
        Argument works from state of body, left finger, right finger
        '''
        q1_l, q2_l, q1dot_l, q2dot_l = x_l
        q1_r, q2_r, q1dot_r, q2dot_r = x_r

        x = np.array([q1_l, q2_l, q1_r, q2_r,
                      q1dot_l, q2dot_l, q1dot_r, q2dot_r])

        return x

    def step(self, x, u):
        ''' 
        Semi-implicit time-stepping for dynamics integration. Does the following:
        1. Compute contact forces at time (t) from joint configuration at time (t)
        2. Compute velocities at time (t+1) from contact forces at time (t)
        3. Compute joint configuration at time (t+1) from velocities at time (t+1) 

        Inputs:
          x : (nq + nv) np array
          u : (nu) np array
        Outputs:
          x : (nq + nv) np array
        '''

        # Unpack Joints and torques--------------------------------------------
        
        x_l, x_r = self.unpack_states(x)
        tau1_l, tau2_l, tau1_r, tau2_r = u

        # Compute contact forces------------------------------------------------

        # Update left manipulator velocity
        Ml, Cl = self.manipulator_params(x_l)
        Jl = self.jacobian(x_l)

        x_l[2:4] += self.params['h'] * np.linalg.inv(Ml).dot(
            -np.dot(Cl, x_l[2:4]) + np.array([tau1_l, tau2_l]))

        # Update right manipulator velocity 
        Mr, Cr = self.manipulator_params(x_r)
        Jr = self.jacobian(x_r)
        x_r[2:4] += self.params['h'] * np.linalg.inv(Mr).dot(
            -np.dot(Cr, x_r[2:4]) + np.array([tau1_r, tau2_r]))

        # Propagate positions---------------------------------------------------
        x_l[0:2] += self.params['h'] * x_l[2:4]
        x_r[0:2] += self.params['h'] * x_r[2:4]

        # Repack states and return----------------------------------------------

        return self.pack_states(x_l, x_r)

    def rollout(self, initial_state, inputs):
        '''
        Rollout of dynamics from the initial state and input, according to sim parameters.
        Inputs:
            initial state: (nq + nv) np array
            inputs: T x nu np array 
        Output:
            Trajectory class with timesteps of T + 1 
        '''

        T = inputs.shape[0]

        x_traj = np.zeros((T + 1, 2 * self.nq))
        f_traj = None
        u_traj = inputs
        init_time = time.time()

        x_traj[0] = initial_state

        for t in range(T):
            # 1. Compute forces and store them
            # 2. Rollout dynamics
            x_traj[t+1] = self.step(x_traj[t], u_traj[t])

        compute_time = time.time() - init_time

        return Trajectory(self, T, x_traj, f_traj, u_traj, compute_time = compute_time)

    '''
    Animation Section---------------------------------------------------------------
    '''

    def draw_circle(self, pos, radius, color, fillcolor, width=2.0):
        '''Draw on Tkinter. Used for rendering GUI'''
        bbox = np.array([pos[0] - radius,
                         pos[1] - radius,
                         pos[0] + radius,
                         pos[1] + radius]).astype(np.double)
        bbox = (1e3 * bbox).astype(np.int) # Scale up and round!
        #bbox = np.squeeze(bbox, 1)
        self.canvas.create_oval(bbox[0], bbox[1], bbox[2], bbox[3],
                                fill=fillcolor, outline=color, width=width)

    def draw_circle_image(self, draw, pos, radius, color, fillcolor, width=2):
        '''Draw on ImageDraw. Used for rendering video'''
        bbox = np.array([pos[0] - radius,
                         pos[1] - radius,
                         pos[0] + radius,
                         pos[1] + radius]).astype(np.double)
        bbox = (1e3 * bbox).astype(np.int)
        #bbox = np.squeeze(bbox, 1)
        draw.ellipse((bbox[0], bbox[1], bbox[2], bbox[3]),
                     fill=fillcolor, outline=color, width=width)

    def draw_line(self, pos1, pos2, color, width=15.0):
        '''Draw on Tkinter. Used for rendering GUI'''
        pos1 = np.squeeze((1e3 * pos1).astype(np.int))
        pos2 = np.squeeze((1e3 * pos2).astype(np.int))
        
        self.canvas.create_line(pos1[0], pos1[1], pos2[0], pos2[1],
                                fill=color, width=width)

    def draw_line_image(self, draw, pos1, pos2, color, width=15):
        '''Draw on ImageDraw. Used for rendering video'''
        pos1 = np.squeeze((1e3 * pos1).astype(np.int))
        pos2 = np.squeeze((1e3 * pos2).astype(np.int))
        draw.line((pos1[0], pos1[1], pos2[0], pos2[1]), fill=color, width=width)

    def animate(self, traj, video_name="run_finger.avi", frame_rate=300.):
        '''
        Animates the trajectory with gui and video.
        Inputs:
            Trajectory class to animate, video name, and frame rate of video.
        '''
        # GUI Parameters
        
        self.width = 500
        self.height = 500
        self.tk = Tk()
        self.tk.title("2 Finger Simulation")
        self.canvas = Canvas(self.tk, width=self.width, height=self.height, bg='white')
        self.canvas.pack()
        self.img_array = []

        for t in range(traj.T + 1):
            self.render(traj.x_traj[t])
            time.sleep(1./frame_rate)

        self.save_video(os.path.join("videos/", video_name), frame_rate)

    def render(self, x):
        ''' Horrible piece of code for rendering, but gets the job done.'''
        self.canvas.delete(ALL)

        # 1. Unpack states
        x_l, x_r = self.unpack_states(x)

        pb_l = self.fkin(x_l, 'l0')
        p1_l = self.fkin(x_l, 'l1')
        p2_l = self.fkin(x_l, 'l2')
        pb_r = self.fkin(x_r, 'r0')
        p1_r = self.fkin(x_r, 'r1')
        p2_r = self.fkin(x_r, 'r2')

        self.draw_line(pb_l, p1_l, 'black')
        self.draw_line(p1_l, p2_l, 'black')
        self.draw_line(pb_r, p1_r, 'black')
        self.draw_line(p1_r, p2_r, 'black')                
        
        self.draw_circle(p1_l, self.params['w'] / 2, 'black', 'white')
        self.draw_circle(p2_l, self.params['w'] / 2, 'black', 'blue')
        self.draw_circle(p1_r, self.params['w'] / 2, 'black', 'white')
        self.draw_circle(p2_r, self.params['w'] / 2, 'black', 'blue')
        
        self.draw_circle(pb_l, self.params['w'] / 2 + 5e-3, 'black', 'white')
        self.draw_circle(pb_l, self.params['w'] / 2, 'black', 'white')
        self.draw_circle(pb_r, self.params['w'] / 2 + 5e-3, 'black', 'white')
        self.draw_circle(pb_r, self.params['w'] / 2, 'black', 'white')

        self.tk.update()

        image = Image.new("RGB", (self.width, self.height), (255, 255, 255))
        draw = ImageDraw.Draw(image)

        self.draw_line_image(draw, pb_l, p1_l, 'black')
        self.draw_line_image(draw, p1_l, p2_l, 'black')
        self.draw_line_image(draw, pb_r, p1_r, 'black')
        self.draw_line_image(draw, p1_r, p2_r, 'black')                

        self.draw_circle_image(draw, p1_l, self.params['w'] / 2, 'black', 'white')
        self.draw_circle_image(draw, p2_l, self.params['w'] / 2, 'black', 'blue')
        self.draw_circle_image(draw, p1_r, self.params['w'] / 2, 'black', 'white')
        self.draw_circle_image(draw, p2_r, self.params['w'] / 2, 'black', 'blue')

        self.draw_circle_image(draw, pb_l, self.params['w'] / 2 + 5e-3, 'black', 'white')
        self.draw_circle_image(draw, pb_l, self.params['w'] / 2, 'black', 'white')
        self.draw_circle_image(draw, pb_r, self.params['w'] / 2 + 5e-3, 'black', 'white')
        self.draw_circle_image(draw, pb_r, self.params['w'] / 2, 'black', 'white')

        cv_image = np.array(image)
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        self.img_array.append(cv_image)

    def save_video(self, video_name, frame_rate):
        out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'MJPG'), frame_rate,
                              (self.width, self.height))
        for i in range(len(self.img_array)):
            out.write(self.img_array[i])
        out.release
