import numpy as np
import matplotlib.pyplot as plt

class Trajectory():
    ''' 
    Trajectory class for interfacing. The objective of this class is to facilitate:
    1. Easy interfacing between different solvers and initial guesses
    2. Evaluation for different metrics such as cost, stability, physical accuracy, etc.
    '''
    def __init__(self, sim, T, x_traj, f_traj, u_traj, compute_time=None):
        '''
        Initial arguments:
        - sim    : what is this a trajectory for?
        -   T    : how long is this trajectory?
        - x_traj : (T + 1) x (2 * nq) numpy array of system configuration and velocities
        - f_traj : (T + 1) x nf numpy array of system contact forces
        - u_traj : T x nu numpy array of system inputs 

        Optional arguments:
        - compute_time : stores how long it took to obtain this trajectory.
        '''
        self.T = T
        self.sim = sim
        self.x_traj = x_traj
        self.f_traj = f_traj
        self.u_traj = u_traj

        self.cost = self.compute_cost()
        self.compute_time = compute_time

    def compute_cost(self):
        '''
        Compute cost (currently computes input norm cost)
        '''
        cost = 0.0
        for i in range(self.T):
            cost += self.u_traj[i].dot(self.u_traj[i])
        return cost

    def plot(self):
        ''' 
        Method to monitor (plot) the trajectory. More finer plots
        (such as adding legends / axis) should be done on simulator class 
        or manually.
        '''
        plt.figure()
        
        plt.subplot(2,2,1)
        plt.title('System Configurations')
        for i in range(self.sim.nq):
            plt.plot(range(self.T+1), self.x_traj[:,i])
            
        plt.subplot(2,2,2)
        plt.title('System Velocities')
        for i in range(self.sim.nv):
            plt.plot(range(self.T+1), self.x_traj[:,self.sim.nq + i])
            
        plt.subplot(2,2,3)
        plt.title('System Forces')
        for i in range(self.sim.nf):
            plt.plot(range(self.T+1), self.f_traj[:,i])
            
        plt.subplot(2,2,4)
        plt.title('System Inputs')
        for i in range(self.sim.nu):
            plt.plot(range(self.T), self.u_traj[:,i])
            
        plt.show()
            
        
            


        

    
            
            
        
