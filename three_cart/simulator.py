from tkinter import *
import time, os, io
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from PIL import Image, ImageTk, ImageDraw
import cv2
import time

class CartSimulator():
    def __init__(self, gui=True, video=False):

        self.gui = gui
        self.video = video

        # State Variables defined as: [x1, x2, x3, v1, v2, v3]
        self.x = np.zeros(6).astype(np.double)
        self.contact = np.zeros(2).astype(np.double)
        self.params = {'m': 1.,  # mass
<<<<<<< HEAD
                       'c': 3.0,  # viscosity
                       'k': 10.0,  # elasticity
                       'd': 80.,    # cart length
                       'w': 50.,    # cart height (just for visualization) 
                       'h': 0.01   # time step
                       't': 0.0}   # sleep time before redrawing on canvas
=======
                       'c': 1.0,  # viscosity
                       'k': 500.0,  # elasticity
                       'd': 0.5,    # cart length
                       'w': 0.5,    # cart height (just for visualization) 
                       'h': 0.01} # time step
>>>>>>> d6cc1663c7457e6c120fa21d3c78b352c01ada53

        self.step_count = 0

        if(self.gui):
            # GUI Parameters
            self.width = 500
            self.height = 300
            self.tk = Tk()
            self.tk.title("3 Cart Simulation")
            self.canvas = Canvas(self.tk, width=self.width, height=self.height, bg='white')
            self.canvas.pack()

            self.img_array = []
            
    def get_state(self):
        return self.x

    def get_contact(self):
        return self.contact

    def set_state(self, x):
        self.x = x.copy()

    def set_parameters(self, param_name, param_value):
        self.params[param_name] = param_value

    # Semi-implicit time-stepping for dynamics integration
    def step(self, u):
        # Compute contact forces first
        lambda1 = np.maximum(-self.params['k'] * (self.x[1] - self.x[0] - self.params['d']), 0)
        lambda2 = np.maximum(-self.params['k'] * (self.x[2] - self.x[1] - self.params['d']), 0)

        self.contact = np.array([lambda1, lambda2]).astype(np.double)

        # Propagate velocities
        self.x[3] += self.params['h'] * (-self.params['c'] * self.x[3] - lambda1 + u[0])
        self.x[4] += self.params['h'] * (-self.params['c'] * self.x[4] + lambda1 - lambda2)
        self.x[5] += self.params['h'] * (-self.params['c'] * self.x[5] + lambda2 + u[1])

        # Propagate positions
        self.x[0] += self.params['h'] * self.x[3]
        self.x[1] += self.params['h'] * self.x[4]
        self.x[2] += self.params['h'] * self.x[5]

        self.step_count += 1

        if(self.gui):
            self.render()
            time.sleep(self.params['t'])
            
    def render(self):
        # sry for this mess....

        vis_x0 = int(self.x[0] * 1e2)
        vis_x1 = int(self.x[1] * 1e2)
        vis_x2 = int(self.x[2] * 1e2)
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

        if(self.video) and (np.remainder(self.step_count, 4) == 0):
            image = Image.new("RGB", (self.width, self.height), (255, 255, 255))
            draw = ImageDraw.Draw(image)

            draw.rectangle([bbox_1[0], bbox_1[1], bbox_1[2], bbox_1[3]], outline='blue')
            draw.rectangle([bbox_2[0], bbox_2[1], bbox_2[2], bbox_2[3]], outline='green')
            draw.rectangle([bbox_3[0], bbox_3[1], bbox_3[2], bbox_3[3]], outline='blue')
            cv_image = np.array(image)
            self.img_array.append(cv_image)



    def save_video(self, video_name):
        out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'MJPG'), 30,
                              (self.width, self.height))
        for i in range(len(self.img_array)):
            out.write(self.img_array[i])
        out.release




        

        



    
        

        
