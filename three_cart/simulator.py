from tkinter import *
import time, os, io, sys
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from PIL import Image, ImageTk, ImageDraw
import cv2

sys.path.insert(0, os.getcwd())
from common.trajectory import Trajectory

class CartSimulator():
    '''
    Cart Simulator. Contains functions for the three cart system.

    State definition: [q1, q2, q3, v1, v2, v3]
    Input definition: [u1, u2] 
    Force definition: [f1, f2]

    NOTE: Simulator does not contain any 'internal states'. It is up to the user to
          keep track of states.
    '''
    def __init__(self, gui=True):

        self.gui = gui

        self.nq = 3
        self.nv = 3
        self.nu = 2
        self.nf = 2
        
        self.params = {'m': 1.,  # mass
                       'c': 5.0,  # viscosity
                       'k': 10.0,  # elasticity
                       'd': 0.5,    # cart length
                       'w': 0.5,    # cart height (just for visualization) 
                       'h': 0.01,   # time step
                       't': 0.0}   # sleep time before redrawing on canvas
        

    def set_parameter(self, param_name, param_value):
        '''
        Set parameters of the simulator according to param_name and value. 
        '''
        self.params[param_name] = param_value

    def set_parameters(self, param_dict):
        '''
        Set parameters of the simulator with a dict. Useful for changing params with 
        entire dictionary. 
        '''
        for key, value in param_dict.items():
            self.set_parameter(key, value)
            
    def compute_contact(self, x):
        '''
        Computes contact forces from the current states. Both state and configuration
        as input is supported.
        '''
        lambda1 = np.maximum(-self.params['k'] * (x[1] - x[0] - self.params['d']), 0)
        lambda2 = np.maximum(-self.params['k'] * (x[2] - x[1] - self.params['d']), 0)

        return np.array([lambda1, lambda2])

    def step(self, x, u):
        '''
        Semi-implicit time stepping propagation of dynamics. 
        Inputs:
          x : (nq + nv) np array 
          u : (nu) np array
        Outputs:
          x : (nq + nv) np array 
        '''
        
        # Compute contact forces first
        contact = self.compute_contact(x)
        lambda1, lambda2 = contact
        
        # Propagate velocities
        x[3] += self.params['h'] * (-self.params['c'] * x[3] - lambda1 + u[0])
        x[4] += self.params['h'] * (-self.params['c'] * x[4] + lambda1 - lambda2)
        x[5] += self.params['h'] * (-self.params['c'] * x[5] + lambda2 + u[1])

        # Propagate positions
        x[0] += self.params['h'] * x[3]
        x[1] += self.params['h'] * x[4]
        x[2] += self.params['h'] * x[5]

        return x

    def rollout(self, initial_state, inputs):
        '''
        Rollout of dynamics from initial state and input, according to sim parameters.
        Inputs:
          initial_state: (nq + nv) np array
          inputs: T x nu np array
        Output:
          Trajectory class with timesteps of T + 1

        NOTE: how many rollouts depends on the size of the input array given. 
        '''

        T = inputs.shape[0]

        x_traj = np.zeros((T + 1, 2 * self.nq))
        f_traj = np.zeros((T + 1, self.nf))
        u_traj = inputs  # must be T x nu 
        init_time = time.time()

        x_traj[0] = initial_state

        for t in range(T):
            # 1. Compute forces and store them.
            f_traj[t] = self.compute_contact(x_traj[t])
            # 2. Rollout dynamics
            x_traj[t+1] = self.step(x_traj[t], u_traj[t])

        compute_time = time.time() - init_time

        return Trajectory(self, T, x_traj, f_traj, u_traj, compute_time = compute_time)

    def animate(self, traj, video_name="run.avi", frame_rate=300.):
        '''
        Animates the trajectory with gui and video. 
        Inputs:
          Trajectory class to animate, video name, and frame rate of video.

        TODO: It is possible to reduce video size by saving every n frames, but this
              hasn't been implemented.
        '''
        # GUI Parameters
        self.width = 500
        self.height = 300
        self.tk = Tk()
        self.tk.title("3 Cart Simulation")
        self.canvas = Canvas(self.tk, width=self.width, height=self.height, bg='white')
        self.canvas.pack()
        self.img_array = []

        for t in range(traj.T + 1):
            self.render(traj.x_traj[t])
            time.sleep(1./frame_rate)
            
        self.save_video(os.path.join("videos/", video_name), frame_rate)

    def eval_traj(self, traj, xd):
        '''
        TODO:
        Evaluates the trajectory on several criteria:
          1. What is the cost of the trajectory?
          2. Does the trajectory respect the physics defined by the simulator?
          3. Is the trajectory stable with respect to given final coordinate xd?
          4. How much time did it take to generate this trajectory?
        '''
        pass
            
    def render(self, x):
        
        vis_x0 = int(x[0] * 1e2)
        vis_x1 = int(x[1] * 1e2)
        vis_x2 = int(x[2] * 1e2)
        vis_d = int(self.params['d'] * 1e2)
        vis_w = int(self.params['w'] * 1e2)
        
        bbox_1 = [vis_x0 - vis_d / 2,
                  self.height / 2 - vis_w / 2,
                  vis_x0 + vis_d / 2,
                  self.height / 2 + vis_w / 2]
        bbox_2 = [vis_x1 - vis_d / 2,
                  self.height / 2 - vis_w / 2,
                  vis_x1 + vis_d / 2,
                  self.height / 2 + vis_w / 2]
        bbox_3 = [vis_x2 - vis_d / 2,
                  self.height / 2 - vis_w / 2,
                  vis_x2 + vis_d / 2,
                  self.height / 2 + vis_w / 2]        
        self.canvas.delete(ALL)
        self.canvas.create_rectangle(
            bbox_1[0], bbox_1[1], bbox_1[2], bbox_1[3], fill='', outline='red')
        self.canvas.create_rectangle(
            bbox_3[0], bbox_3[1], bbox_3[2], bbox_3[3], fill='', outline='red')
        self.canvas.create_rectangle(
            bbox_2[0], bbox_2[1], bbox_2[2], bbox_2[3], fill='', outline='green')
        self.tk.update()

        image = Image.new("RGB", (self.width, self.height), (255, 255, 255))
        draw = ImageDraw.Draw(image)

        draw.rectangle([bbox_1[0], bbox_1[1], bbox_1[2], bbox_1[3]], outline='blue')
        draw.rectangle([bbox_2[0], bbox_2[1], bbox_2[2], bbox_2[3]], outline='green')
        draw.rectangle([bbox_3[0], bbox_3[1], bbox_3[2], bbox_3[3]], outline='blue')
        cv_image = np.array(image)
        self.img_array.append(cv_image)

    def save_video(self, video_name, frame_rate):
        out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'MJPG'), 100,
                              (self.width, self.height))
        for i in range(len(self.img_array)):
            out.write(self.img_array[i])
        out.release
