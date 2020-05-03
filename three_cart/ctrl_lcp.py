import simulator
import numpy as np
import matplotlib.pyplot as plt
import time, os, sys

from pydrake.solvers.mathematicalprogram import MathematicalProgram, Solve
from pydrake.all import eq, le, ge
from pydrake.solvers.ipopt import IpoptSolver
from pydrake.solvers.snopt import SnoptSolver

sys.path.insert(0, os.getcwd())
from common.trajectory import Trajectory
from common.utils import snopt_status

def LCPtrajopt(sim, xi, xf, T, options=None):
    ''' 
    Function that does trajectory optimization with LCP Approaches. 

    Options should be a dictionary containing the following:
    - initial_guess : initial guess to be used (Trajectory Class)
    - tol           : tolerance for lcp inequality constraint (semipositive float). Default 0.0
    - solver        : solver to be used. Support "snopt" or "ipopt", default is "snopt"
    - contact_max   : maximum value for contact. Defaults to None. 
    - input_max     : maximum value for input. Defaults to None.
    '''
    #---------------------------------------------------------------------------
    # Define helper functions here
    #---------------------------------------------------------------------------
    def pack_trajectory(sim, q_traj, v_traj, f_traj, u_traj, compute_time):
        '''
        Pack trajectory from q,v,f,u to Trajectory class
        '''
        T = u_traj.shape[0]
        x_traj = np.hstack((q_traj, v_traj))
        traj = Trajectory(sim, T, x_traj, f_traj, u_traj, compute_time)
        return traj

    def unpack_trajectory(sim, traj):
        '''
        Unpack trajectory from Trajectory class fo q,v,f,u notation
        '''
        x_traj = traj.x_traj
        q_traj = x_traj[:,0:sim.nq]
        v_traj = x_traj[:,sim.nq:(2*sim.nq)]
        f_traj = traj.f_traj
        u_traj = traj.u_traj
        return q_traj, v_traj, f_traj, u_traj
    
    def generate_initial_traj(sim, xi, xf, T):
        ''' 
        Use default initial guess with stationary assumption
        '''
        qinit = np.tile(np.array(xi[0:3]), (T + 1, 1))
        vinit = np.tile(np.array(xi[3:6]), (T + 1, 1))
        uinit = np.tile(np.array([0,0]), (T, 1))
        finit = np.tile(np.array([0,0]), (T, 1))

        traj = pack_trajectory(sim, qinit, vinit, uinit, finit, None)
        return traj
        

    #---------------------------------------------------------------------------
    # 1. Set default parameters options.
    #---------------------------------------------------------------------------

    if (options == None) or (options['tol'] == None):
        tol = 0.0
    else:
        tol = options['tol']
    if (options == None) or (options['solver'] == None):
        solver = "snopt"
    else:
        solver = options['solver']
    if (options == None) or (options['contact_max'] == None):
        contact_max = None
    else:
        contact_max = options['contact_max']
    if (options == None) or (options['input_max'] == None):
        input_max = None
    else:
        input_max = options['input_max']
    if (options == None) or (options['initial_guess'] == None):
        initial_guess = generate_initial_traj(sim, xi, xf, T)
    else:
        initial_guess = options['initial_guess']

    qinit, vinit, finit, uinit = unpack_trajectory(sim, initial_guess)

    #---------------------------------------------------------------------------
    # 2. Define program and constraints
    #---------------------------------------------------------------------------
    prog = MathematicalProgram()

    # Joint configuration states & Contact forces
    q = prog.NewContinuousVariables(rows=T + 1, cols=sim.nq, name='q')
    v = prog.NewContinuousVariables(rows=T + 1, cols=sim.nq, name='v')
    u = prog.NewContinuousVariables(rows=T, cols=sim.nu, name='u')
    contact = prog.NewContinuousVariables(rows=T, cols=sim.nf, name='lambda')
    alpha = prog.NewContinuousVariables(rows=T, cols=2, name='alpha')
    beta = prog.NewContinuousVariables(rows=T, cols=2, name='beta')

    # Add Initial Condition Constraint
    prog.AddConstraint(eq(q[0], np.array(xi[0:3])))
    prog.AddConstraint(eq(v[0], np.array(xi[3:6])))

    # Add Final Condition Constraint
    prog.AddConstraint(eq(q[T], np.array(xf[0:3])))
    prog.AddConstraint(eq(v[T], np.array(xf[3:6])))        

    # Add Dynamics Constraints
    for t in range(T):
        # Add Dynamics Constraints
        prog.AddConstraint(eq(q[t+1], (q[t] + sim.params['h'] * v[t+1])))

        prog.AddConstraint(v[t+1,0] == (v[t,0] + sim.params['h'] * (
            -sim.params['c'] * v[t,0] - contact[t,0] + u[t,0])))
        prog.AddConstraint(v[t+1,1] == (v[t,1] + sim.params['h'] * (
            -sim.params['c'] * v[t,1] + contact[t,0] - contact[t,1])))
        prog.AddConstraint(v[t+1,2] == (v[t,2] + sim.params['h'] * (
            -sim.params['c'] * v[t,2] + contact[t,1] + u[t,1])))

        # Add Contact Constraints
        prog.AddConstraint(ge(alpha[t], 0))
        prog.AddConstraint(ge(beta[t], 0))
        prog.AddConstraint(alpha[t,0] == contact[t,0])
        prog.AddConstraint(alpha[t,1] == contact[t,1])
        prog.AddConstraint(beta[t,0] == (contact[t,0] + sim.params['k'] * (
            q[t,1] - q[t,0] - sim.params['d'])))
        prog.AddConstraint(beta[t,1] == (contact[t,1] + sim.params['k'] * (
            q[t,2] - q[t,1] - sim.params['d'])))

        # Complementarity constraints. Start with relaxed version and start constraining.
        prog.AddConstraint(alpha[t,0] * beta[t,0] <= tol)
        prog.AddConstraint(alpha[t,1] * beta[t,1] <= tol)            

        # Add Input Constraints and Contact Constraints
        if (contact_max is not None):
            prog.AddConstraint(le(contact[t], contact_max))
            prog.AddConstraint(ge(contact[t], -contact_max))
        if (input_max is not None):
            prog.AddConstraint(le(u[t], input_max))
            prog.AddConstraint(ge(u[t], -input_max))

        # Add Costs
        prog.AddCost(u[t].dot(u[t]))

    #---------------------------------------------------------------------------
    # 3. Set Initial guess of the program
    #---------------------------------------------------------------------------

    initial_guess = np.empty(prog.num_vars())
    prog.SetDecisionVariableValueInVector(q, qinit, initial_guess)
    prog.SetDecisionVariableValueInVector(v, vinit, initial_guess)
    prog.SetDecisionVariableValueInVector(u, uinit, initial_guess)
    #prog.SetDecisionVariableValueInVector(contact, finit, initial_guess)

    #---------------------------------------------------------------------------
    # 4. Solve the program
    #---------------------------------------------------------------------------
    
    # Solve the program
    if (solver == "ipopt"):
        solver_cls = IpoptSolver()
    elif (solver == "snopt"):
        solver_cls = SnoptSolver()

    init_time = time.time()
    result = solver_cls.Solve(prog, initial_guess)
    solver_time = time.time() - init_time

    if (solver == "ipopt"):
        print("Ipopt Solver Status: ", result.get_solver_details().status,
          ", meaning ", result.get_solver_details().ConvertStatusToString())
    elif (solver == "snopt"):
        val = result.get_solver_details().info
        status = snopt_status(val)
        print("Snopt Solver Status: ", result.get_solver_details().info,
              ", meaning ", status)



    #---------------------------------------------------------------------------
    # 5. Pack the solution and return trajectory.
    #---------------------------------------------------------------------------

    q_opt = result.GetSolution(q)
    v_opt = result.GetSolution(v)
    f_opt = result.GetSolution(contact)    
    u_opt = result.GetSolution(u)
    traj_opt = pack_trajectory(sim, q_opt, v_opt, f_opt, u_opt, solver_time)
    
    return traj_opt


    


    
    
        

        

        

        

        
        

        
