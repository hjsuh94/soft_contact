from tkinter import *
import time, os, io
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from PIL import Image, ImageTk, ImageDraw
import cv2

class FingerSimulator():
    def __init__(self, gui=True, video=False):

        self.gui = gui
        self.video = video

        self.xb = np.zeros((6, 1)).astype(np.double) # [xb, yb, thetab, dotxb, dotyb, dotthetab]
        self.xl = np.zeros((4, 1)).astype(np.double) # [q1, q2, dotq1, dotq2]
        self.xr = np.zeros((4, 1)).astype(np.double) # [q1, q2, dotq1, dotq2]
        self.x = np.vstack((self.xb, self.xl, self.xr))

        self.pb = np.zeros((2, 1)).astype(np.double) # stores body coordinates, repetitive with xb
        self.pb_l = np.zeros((2, 1)).astype(np.double) # stores base joint coordinate                
        self.p1_l = np.zeros((2, 1)).astype(np.double) # stores first joint coordinate of left
        self.p2_l = np.zeros((2, 1)).astype(np.double) # stores second joint coordinate of left
        self.pb_r = np.zeros((2, 1)).astype(np.double) # stores base joint coordinate
        self.p1_r = np.zeros((2, 1)).astype(np.double) # stores first joint coordinate of right
        self.p2_r = np.zeros((2, 1)).astype(np.double) # stores second joint coordinate of right

        self.p = np.vstack((self.pb, self.p1_l, self.p2_l, self.p1_r, self.p2_r))
        
        self.contact_l = np.zeros((2, 1)).astype(np.double) # [lambda_x, lambda_y]
        self.contact_r = np.zeros((2, 1)).astype(np.double) # [lambda_x, lambda_y]
        self.contact = np.vstack((self.contact_l, self.contact_r))

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
                       'h': 0.001} # time step

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
        self.x = np.vstack((self.xb, self.xl, self.xr))
        self.p = np.vstack((self.pb, self.p1_l, self.p2_l, self.p1_r, self.p2_r))        
        self.contact = np.vstack((self.contact_l, self.contact_r))

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

        self.pb_l = np.array([[self.params['Cx'] - self.params['Lx']],
                              [self.params['Cy'] + self.params['Ly']]]).astype(np.double)
        self.pb_r = np.array([[self.params['Cx'] + self.params['Lx']],
                              [self.params['Cy'] + self.params['Ly']]]).astype(np.double)

    def set_state(self, x, body=None):
        if (body == 'l'):
            self.xl = x
        elif (body == 'r'):
            self.xr = x
        elif (body == 'b'):
            self.xb = x
        else:
            self.x = x
            self.xb = x[0:6].reshape((6,1))
            self.xl = x[6:10].reshape((4,1))
            self.xr = x[10:14].reshape((4,1))
        self.repack_state()

    def set_parameters(self, param_name, param_value):
        self.parameters[param_name] = param_value
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

        x1 = self.params['l'] * (np.cos(q1))
        y1 = self.params['l'] * (np.sin(q1))
        x2 = x1 + self.params['l'] * np.cos(q1 + q2)
        y2 = y1 + self.params['l'] * np.sin(q1 + q2)

        y1 += self.params['Cy'] + self.params['Ly']
        y2 += self.params['Cy'] + self.params['Ly']
        
        if (body == 'l'):
            x1 += self.params['Cx'] - self.params['Lx']
            x2 += self.params['Cx'] - self.params['Lx']            
            self.p1_l = np.vstack((x1, y1))
            self.p2_l = np.vstack((x2, y2))
        elif (body == 'r'):
            x1 += self.params['Cx'] + self.params['Lx']
            x2 += self.params['Cx'] + self.params['Lx']            
            self.p1_r = np.vstack((x1, y1))
            self.p2_r = np.vstack((x2, y2))
            
        else:
            raise ValueError('Body should only be "l" or "r"')
        
        return np.vstack((x2,y2))

    def jacobian(self, q):
        ''' Receives a vector of joints and returns the Jacobian by lienarizing
            the forward kinematics '''
        q1,q2 = q # Unpack Joints
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

        # Compute forward kinematics--------------------------------------------
        pl = self.fkin([q1_l, q2_l], 'l')
        pr = self.fkin([q1_r, q2_r], 'r')
        pb = np.vstack((x,y))
        self.pb = pb

        # Compute contact forces------------------------------------------------
        dist_l = np.linalg.norm(pl - pb, 2) 
        dist_r = np.linalg.norm(pr - pb, 2)

        # Note: these contact forces are applied on the ball. Take negative to apply to finger.
        lambda_l = -self.params['k'] * ((pl - pb) / dist_l) * \
                   np.maximum(self.params['R'] + self.params['r'] - dist_l, 0)
        lambda_r = -self.params['k'] * ((pr - pb) / dist_r) * \
                   np.maximum(self.params['R'] + self.params['r'] - dist_r, 0)

        self.contact_l = lambda_l
        self.contact_r = lambda_r
        
        # Propagate Velocities--------------------------------------------------

        # Update manipuland velocity 
        v = np.vstack((vx, vy, omega))
        self.xb[3:5] += self.params['h'] * (1. / self.params['mb']) * (
            lambda_l + lambda_r - self.params['cb'] * self.xb[3:5])
        self.xb[5] += self.params['h'] * (1./self.params['Ib']) * 0.
        
        # Update left manipulator velocity
        Ml, Cl = self.manipulator_params(self.xl)
        Jl = self.jacobian(np.vstack((q1_l, q2_l)))
        
        self.xl[2:4] += self.params['h'] * np.linalg.inv(Ml).dot(
            -np.dot(Cl, self.xl[2:4]) + np.vstack((tau1_l, tau2_l))
            -np.dot(np.transpose(Jl), lambda_l))

        # Update right manipulator velocity 
        Mr, Cr = self.manipulator_params(self.xr)
        Jr = self.jacobian(np.vstack((q1_r, q2_r)))
        self.xr[2:4] += self.params['h'] * np.linalg.inv(Mr).dot(
            -np.dot(Cr, self.xr[2:4]) + np.vstack((tau1_r, tau2_r))
            -np.dot(np.transpose(Jr), lambda_r))

        # Propagate positions---------------------------------------------------
        self.xb[0:3] += self.params['h'] * self.xb[3:6]
        self.xl[0:2] += self.params['h'] * self.xl[2:4]
        self.xr[0:2] += self.params['h'] * self.xr[2:4]

        self.repack_state()

        self.step_count += 1

        if(self.gui):
            self.render()

    def draw_circle(self, pos, radius, color, fillcolor, width=2.0):
        '''Draw on Tkinter. Used for rendering GUI'''
        bbox = np.array([pos[0] - radius,
                         pos[1] - radius,
                         pos[0] + radius,
                         pos[1] + radius]).astype(np.double)
        bbox = (1e3 * bbox).astype(np.int) # Scale up and round!
        bbox = np.squeeze(bbox, 1)
        self.canvas.create_oval(bbox[0], bbox[1], bbox[2], bbox[3],
                                fill=fillcolor, outline=color, width=width)

    def draw_circle_image(self, draw, pos, radius, color, fillcolor, width=2):
        '''Draw on ImageDraw. Used for rendering video'''
        bbox = np.array([pos[0] - radius,
                         pos[1] - radius,
                         pos[0] + radius,
                         pos[1] + radius]).astype(np.double)
        bbox = (1e3 * bbox).astype(np.int)
        bbox = np.squeeze(bbox, 1)
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
            
    def render(self):
        ''' Horrible piece of code for rendering, but gets the job done.'''
        self.canvas.delete(ALL)

        self.draw_circle(self.pb, self.params['R'], 'black', 'red')

        self.draw_line(self.pb_l, self.p1_l, 'black')
        self.draw_line(self.p1_l, self.p2_l, 'black')
        self.draw_line(self.pb_r, self.p1_r, 'black')
        self.draw_line(self.p1_r, self.p2_r, 'black')                
        
        self.draw_circle(self.p1_l, self.params['w'] / 2, 'black', 'white')
        self.draw_circle(self.p2_l, self.params['w'] / 2, 'black', 'blue')
        self.draw_circle(self.p1_r, self.params['w'] / 2, 'black', 'white')
        self.draw_circle(self.p2_r, self.params['w'] / 2, 'black', 'blue')
        
        self.draw_circle(self.pb_l, self.params['w'] / 2 + 5e-3, 'black', 'white')
        self.draw_circle(self.pb_l, self.params['w'] / 2, 'black', 'white')
        self.draw_circle(self.pb_r, self.params['w'] / 2 + 5e-3, 'black', 'white')
        self.draw_circle(self.pb_r, self.params['w'] / 2, 'black', 'white')

        self.tk.update()

        if (self.video) and (np.remainder(self.step_count, 10) == 0):
            image = Image.new("RGB", (self.width, self.height), (255, 255, 255))
            draw = ImageDraw.Draw(image)

            self.draw_circle_image(draw, self.pb, self.params['R'], 'black', 'red')

            self.draw_line_image(draw, self.pb_l, self.p1_l, 'black')
            self.draw_line_image(draw, self.p1_l, self.p2_l, 'black')
            self.draw_line_image(draw, self.pb_r, self.p1_r, 'black')
            self.draw_line_image(draw, self.p1_r, self.p2_r, 'black')                

            self.draw_circle_image(draw, self.p1_l, self.params['w'] / 2, 'black', 'white')
            self.draw_circle_image(draw, self.p2_l, self.params['w'] / 2, 'black', 'blue')
            self.draw_circle_image(draw, self.p1_r, self.params['w'] / 2, 'black', 'white')
            self.draw_circle_image(draw, self.p2_r, self.params['w'] / 2, 'black', 'blue')

            self.draw_circle_image(draw, self.pb_l, self.params['w'] / 2 + 5e-3, 'black', 'white')
            self.draw_circle_image(draw, self.pb_l, self.params['w'] / 2, 'black', 'white')
            self.draw_circle_image(draw, self.pb_r, self.params['w'] / 2 + 5e-3, 'black', 'white')
            self.draw_circle_image(draw, self.pb_r, self.params['w'] / 2, 'black', 'white')

            cv_image = np.array(image)
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            self.img_array.append(cv_image)

    def save_video(self, video_name):
        out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'MJPG'), 300,
                              (self.width, self.height))
        for i in range(len(self.img_array)):
            out.write(self.img_array[i])
        out.release

# Example usage. Will be gone soon and separated into another file.
C = FingerSimulator(gui=True, video=True)
timesteps = 12000
init_state = np.array([250e-3, 200e-3, 0.0, 0.0, 0.0, 0, # manipuland pose 
                       np.pi/2, -np.pi/2, 0, 0, # left finger joint pose
                       np.pi/2, np.pi/2, 0, 0]).astype(np.double) # right finger joint pose
x_trajectory = np.zeros((14, timesteps))
x_trajectory[:,0] = init_state
lambda_trajectory = np.zeros((4, timesteps))
C.set_state(init_state)

# Try out a simple Cartesian PD Controller!
kp = 3.0
kd = 1.0
p_l_past = 0.0
p_r_past = 0.0

for t in range(timesteps):
    x_trajectory[:,t] = C.get_state().ravel()
    lambda_trajectory[:,t] = C.get_contact().ravel()

    pd_l = np.array([[200e-3], [400e-3 - t * 3e-5]])
    pd_r = np.array([[300e-3], [400e-3 - t * 3e-5]])

    q1_l, q2_l, q1dot_l, q2dot_l = C.get_state(body='l')
    q1_r, q2_r, q1dot_r, q2dot_r = C.get_state(body='r')

    J_l = C.jacobian(np.array([q1_l, q2_l]))
    J_r = C.jacobian(np.array([q1_r, q2_r]))
    p_l = C.fkin(np.array([q1_l, q2_l]), body='l')
    p_r = C.fkin(np.array([q1_r, q2_r]), body='r')

    ul = np.dot(np.transpose(J_l), (-kp * (p_l - pd_l) - kd * (p_l - p_l_past) / C.params['h']))
    ur = np.dot(np.transpose(J_r), (-kp * (p_r - pd_r) - kd * (p_r - p_r_past) / C.params['h']))

    u = np.array([ul[0], ul[1], ur[0], ur[1]])

    # Simulate saturation
    max_torque = 5.0
    u[u > max_torque] = max_torque
    
    C.step(u)

    p_l_past = p_l
    p_r_past = p_r

C.save_video("test.avi")
