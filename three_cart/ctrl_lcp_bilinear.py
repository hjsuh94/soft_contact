import simulator
import numpy as np
import matplotlib.pyplot as plt
import time

import gurobipy as gp
from gurobipy import GRB

class LCPCtrl():
    def __init__(self, sim):

        self.sim = sim
        self.params = self.sim.params
        self.solver = "snopt"

        self.contact_max = 10
        self.input_max = 10

        # Ctrl Parameters
        self.T = 100 # Number of timesteps to optimize for
        self.nq = 3 # Number of joints
        self.nu = 3 # Number of inputs
        self.nf = 2 # Number of contact forces
        self.xd = np.array([0, 0, 0, 0, 0, 0]).astype(np.double)

    def set_xd(self, xd):
        self.xd = xd

    def compute_input(self, x, xd, initial_guess=None, tol=0.0):

        prog = gp.Model("LCPCtrl")
        prog.params.NonConvex = 2

        # Joint configuration states & Contact forces
        q = prog.addVars(self.T + 1, self.nq, name='q')
        v = prog.addVars(self.T + 1, self.nq, name='v')
        u = prog.addVars(self.T, self.nu, name='u')
        #contact = prog.addVars(self.T, self.nf, name='f')

        # Add Objective
        objective = 0.0

        # Add Initial & Terminal Condition Constraint
        for i in range(3):
            prog.addConstr(q[0,i] == x[i])
            prog.addConstr(v[0,i] == x[3 + i])
            prog.addConstr(q[self.T,i] == xd[i])
            prog.addConstr(v[self.T,i] == xd[3 + i])
        
        # Add Dynamics Constraints
        for t in range(self.T):
            # Add Dynamics Constraints
            for i in range(3):
                prog.addConstr(q[t+1,i] == (
                    q[t,i] + self.sim.params['h'] * v[t+1,i]))

            prog.addConstr(v[t+1,0] == (v[t,0] + self.sim.params['h'] * (
                -self.sim.params['c'] * v[t,0] + u[t,0])))
            prog.addConstr(v[t+1,1] == (v[t,1] + self.sim.params['h'] * (
                -self.sim.params['c'] * v[t,1] + u[t,1])))
            prog.addConstr(v[t+1,2] == (v[t,2] + self.sim.params['h'] * (
                -self.sim.params['c'] * v[t,2] + u[t,2])))

            '''                
            prog.addConstr(v[t+1,0] == (v[t,0] + self.sim.params['h'] * (
                -self.sim.params['c'] * v[t,0] - contact[t,0] + u[t,0])))
            prog.addConstr(v[t+1,1] == (v[t,1] + self.sim.params['h'] * (
                -self.sim.params['c'] * v[t,1] + contact[t,0] - contact[t,1])))
            prog.addConstr(v[t+1,2] == (v[t,2] + self.sim.params['h'] * (
                -self.sim.params['c'] * v[t,2] + contact[t,1] + u[t,1])))
            '''

            '''
            # Add Contact Constraints
            prog.addConstr(contact[t,0] >= 0)
            prog.addConstr(contact[t,1] >= 0)
            prog.addConstr(contact[t,0] + self.sim.params['k'] * (
                q[t,1] - q[t,0] - self.sim.params['d']) >= 0)
            prog.addConstr(contact[t,1] + self.sim.params['k'] * (
                q[t,2] - q[t,1] - self.sim.params['d']) >= 0)
            

            # Complementarity constraints. Start with relaxed version and start constraining.

            prog.addConstr(contact[t,0] * (contact[t,0] + self.sim.params['k'] * (
                q[t,1] - q[t,0] - self.sim.params['d'])) <= tol)
            prog.addConstr(contact[t,1] * (contact[t,1] + self.sim.params['k'] * (
                q[t,2] - q[t,1] - self.sim.params['d'])) <= tol)
            '''

            
            # Add Input Constraints and Contact Constraints
            '''
            for i in range(2):
                prog.addConstr(contact[t,i] <= self.contact_max)
                prog.addConstr(contact[t,i] >= -self.contact_max)
                prog.addConstr(u[t,i] <= self.input_max)
                prog.addConstr(u[t,i] >= -self.input_max)
            '''

            #objective += u[t,0] * u[t,0] + u[t,1] * u[t,1]

        prog.setObjective(0, GRB.MAXIMIZE)
        prog.update()

        '''
        # Set Initial Guess as empty. Otherwise, start from last solver iteration.
        if (type(initial_guess) == type(None)):
            initial_guess = np.empty(prog.num_vars())

            # Populate initial guess by linearly interpolating between initial
            # and final states
            #qinit = np.linspace(x[0:3], xd[0:3], self.T + 1)
            qinit = np.tile(np.array(x[0:3]), (self.T + 1, 1))
            vinit = np.tile(np.array(x[3:6]), (self.T + 1, 1))
            uinit = np.tile(np.array([0,0]), (self.T, 1))
            finit = np.tile(np.array([0,0]), (self.T, 1))

            prog.SetDecisionVariableValueInVector(q, qinit, initial_guess)
            prog.SetDecisionVariableValueInVector(v, vinit, initial_guess)
            prog.SetDecisionVariableValueInVector(u, uinit, initial_guess)
            prog.SetDecisionVariableValueInVector(contact, finit, initial_guess)        
        '''

        prog.params.NonConvex = -1 # Needed for bilinear optimization
        prog.optimize()

        #prog.printAttr('q')


        sol = result.GetSolution()
        q_opt = result.GetSolution(q)
        v_opt = result.GetSolution(v)
        u_opt = result.GetSolution(u)
        f_opt = result.GetSolution(contact)
            
        return sol, q_opt, v_opt, u_opt, f_opt

    def step(self):
        ''' Observes the simulator state and do control based on those states'''
        # 1. Get the current state from the simulator.
        x = self.sim.get_state() # [x1, v1, x2, v2, x3, v3]
        u = self.compute_input(x)
        self.sim.step(u)

    def snopt_status(self, val):
        if (val == 1):
            status = "optimality conditions satisfied"
        elif (val == 2):
            status = "feasible point found"
        elif (val == 13):
            status = "nonlinear infeasibilities minimized"
        elif (val == 14):
            status = "infeasibilities minimized"
        elif (val == 15):
            status = "infeasible linear constraints in QP subproblem"
        elif (val == 41):
            status = "current point cannot be improved"
        elif (val == 43):
            status = "cannot satisfy the general constraints"
        else:
            status = "unknown"

        return status
                     

if __name__ == '__main__':
    S = simulator.CartSimulator(gui=True)
    init_state = np.array([2, 3, 4, 0, 0, 0]).astype(np.double)
    des_state = np.array([0.5, 3, 4, 0, 0, 0]).astype(np.double)
    S.set_state(init_state)

    Ctrl = LCPCtrl(S)
    initial_guess = None

    # Successive relaxation
    # Do this once to see if initially converges.
    # Ctrl.compute_input(init_state, des_state, initial_guess, 0.0)
    
    tol_list = np.linspace(4.0, 0.0, 5)
    for i in range(len(tol_list)):
        print("Iteration " + str(i) + ". Current Tolerance: " + str(tol_list[i]))
        sol, q_opt, v_opt, u_opt, f_opt  = Ctrl.compute_input(
            init_state, des_state, initial_guess, tol_list[i])
        
        initial_guess = sol

    plt.figure()
    plt.plot(range(Ctrl.T + 1), q_opt[:,0])
    plt.plot(range(Ctrl.T + 1), q_opt[:,1])
    plt.plot(range(Ctrl.T + 1), q_opt[:,2])
    plt.legend(['Cart 1', 'Cart 2', 'Cart 3'])
    plt.show()

    plt.figure()
    plt.plot(range(Ctrl.T), u_opt[:,0])
    plt.plot(range(Ctrl.T), u_opt[:,1])
    plt.show()

    plt.figure()
    plt.plot(range(Ctrl.T), f_opt[:,0])
    plt.plot(range(Ctrl.T), f_opt[:,1])
    plt.show()

    # Open Loop Visualization

    for t in range(Ctrl.T):
        S.step(u_opt[t,:])
        time.sleep(0.01)

    S.save_video("test.avi")

    


    
    
        

        

        

        

        
        

        
