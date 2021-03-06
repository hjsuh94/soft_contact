import simulator
import numpy as np
import matplotlib.pyplot as plt
import time

from pydrake.solvers.mathematicalprogram import MathematicalProgram, Solve
from pydrake.all import eq, le, ge
from pydrake.solvers.ipopt import IpoptSolver
from pydrake.solvers.snopt import SnoptSolver

class LCPCtrl():
    def __init__(self, sim):

        self.sim = sim
        self.params = self.sim.params
        self.solver = "snopt"

        self.contact_max = 5
        self.input_max = 10

        # Ctrl Parameters
        self.T = 500 # Number of timesteps to optimize for
        self.nq = 3 # Number of joints
        self.nu = 2 # Number of inputs
        self.nf = 2 # Number of contact forces
        self.xd = np.array([0, 0, 0, 0, 0, 0]).astype(np.double)

    def set_xd(self, xd):
        self.xd = xd

    def compute_input(self, x, xd, initial_guess=None, tol=0.0):
        prog = MathematicalProgram()

        # Joint configuration states & Contact forces
        q = prog.NewContinuousVariables(rows=self.T + 1, cols=self.nq, name='q')
        v = prog.NewContinuousVariables(rows=self.T + 1, cols=self.nq, name='v')
        u = prog.NewContinuousVariables(rows=self.T, cols=self.nu, name='u')
        contact = prog.NewContinuousVariables(rows=self.T, cols=self.nf, name='lambda')

        # 
        alpha = prog.NewContinuousVariables(rows=self.T, cols=2, name='alpha')
        beta = prog.NewContinuousVariables(rows=self.T, cols=2, name='beta')

        # Add Initial Condition Constraint
        prog.AddConstraint(eq(q[0], np.array(x[0:3])))
        prog.AddConstraint(eq(v[0], np.array(x[3:6])))

        # Add Final Condition Constraint
        prog.AddConstraint(eq(q[self.T], np.array(xd[0:3])))
        prog.AddConstraint(eq(v[self.T], np.array(xd[3:6])))        
        
        # Add Dynamics Constraints
        for t in range(self.T):
            # Add Dynamics Constraints
            prog.AddConstraint(eq(q[t+1], (q[t] + self.sim.params['h'] * v[t+1])))
            
            prog.AddConstraint(v[t+1,0] == (v[t,0] + self.sim.params['h'] * (
                -self.sim.params['c'] * v[t,0] - contact[t,0] + u[t,0])))
            prog.AddConstraint(v[t+1,1] == (v[t,1] + self.sim.params['h'] * (
                -self.sim.params['c'] * v[t,1] + contact[t,0] - contact[t,1])))
            prog.AddConstraint(v[t+1,2] == (v[t,2] + self.sim.params['h'] * (
                -self.sim.params['c'] * v[t,2] + contact[t,1] + u[t,1])))

            # Add Contact Constraints
            prog.AddConstraint(ge(alpha[t], 0))
            prog.AddConstraint(ge(beta[t], 0))
            prog.AddConstraint(alpha[t,0] == contact[t,0])
            prog.AddConstraint(alpha[t,1] == contact[t,1])
            prog.AddConstraint(beta[t,0] == (contact[t,0] + self.sim.params['k'] * (
                q[t,1] - q[t,0] - self.sim.params['d'])))
            prog.AddConstraint(beta[t,1] == (contact[t,1] + self.sim.params['k'] * (
                q[t,2] - q[t,1] - self.sim.params['d'])))

            # Complementarity constraints. Start with relaxed version and start constraining.
            prog.AddConstraint(alpha[t,0] * beta[t,0] <= tol)
            prog.AddConstraint(alpha[t,1] * beta[t,1] <= tol)            
            
            # Add Input Constraints and Contact Constraints
            prog.AddConstraint(le(contact[t], self.contact_max))
            prog.AddConstraint(ge(contact[t], -self.contact_max))
            prog.AddConstraint(le(u[t], self.input_max))
            prog.AddConstraint(ge(u[t], -self.input_max))
            
            # Add Costs
            prog.AddCost(u[t].dot(u[t]))

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

        # Solve the program
        if (self.solver == "ipopt"):
            solver = IpoptSolver()
        elif (self.solver == "snopt"):
            solver = SnoptSolver()
            
        result = solver.Solve(prog, initial_guess)

        if (self.solver == "ipopt"):
            print("Ipopt Solver Status: ", result.get_solver_details().status,
              ", meaning ", result.get_solver_details().ConvertStatusToString())
        elif (self.solver == "snopt"):
            val = result.get_solver_details().info
            status = self.snopt_status(val)
            print("Snopt Solver Status: ", result.get_solver_details().info,
                  ", meaning ", status)

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
    S = simulator.CartSimulator(gui=True, video=True)
    init_state = np.array([2, 3, 4, 0, 0, 0]).astype(np.double)
    des_state = np.array([0.5, 1.5, 2.5, 0, 0, 0]).astype(np.double)
    S.set_state(init_state)

    Ctrl = LCPCtrl(S)
    initial_guess = None

    # Successive relaxation
    # Do this once to see if initially converges.
    # Ctrl.compute_input(init_state, des_state, initial_guess, 0.0)
    
    tol_list = np.linspace(4.0, 0.0, 4)
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

    


    
    
        

        

        

        

        
        

        
