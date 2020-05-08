import simulator
import numpy as np
import matplotlib.pyplot as plt
import time, os, sys

from pydrake.solvers.mathematicalprogram import MathematicalProgram, Solve
from pydrake.all import eq, le, ge
from pydrake.solvers.ipopt import IpoptSolver
from pydrake.solvers.snopt import SnoptSolver
import pydrake.symbolic as ps

sys.path.insert(0,os.getcwd())
from common.trajectory import Trajectory
from common.utils import snopt_status

def LCPtrajopt(sim, xi, xf, T, options=None):
    '''
    Function that does trajectory optimization with LCP approaches.
    
    Options should be a dictionary containing the following:
    - initial_guess : initial guess to be used (Trajectory class)
    - tol           : tolerance for lcp inequality constraint (semipositive float). Default 0.0
    - solver        : solver to be used. Support "snopt" or "ipopt", default is "snopt"
    - contact_max   : maximum value for contact. Defaults to None (unbounded). 
    - input_max     : maximum value for input. Defaults to None (unbounded).
    '''

    '''
    -------------------------------------------------------------------------
    Helper functions
    -------------------------------------------------------------------------
    '''

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
        Unpack trajectory from trajectory class to q,v,f,u notation
        '''
        x_traj = traj.x_traj
        q_traj = x_traj[:,0:sim.nq]
        v_traj = x_traj[:,sim.nq:(2*sim.nq)]
        f_traj = traj.f_traj
        u_traj = traj.u_traj
        return q_traj, v_traj, f_traj, u_traj

    def generate_initial_traj(sim, xi, xf, T, method='stationary'):
        '''
        Use default initial guess with stationary assumption
        '''

        if (method == 'stationary'):
            qinit = np.tile(np.array(xi[0:sim.nq]), (T + 1, 1))
            vinit = np.tile(np.array(xi[sim.nq:(2*sim.nq)]), (T + 1, 1))
        elif (method == 'linear'):
            qinit = np.linspace(xi[0:sim.nq], xf[0:sim.nq], T+1)
            vinit = np.linspace(xi[sim.nq:(2*sim.nq)], xf[sim.nq:(2*sim.nq)], T+1)

        uinit = np.tile(np.array([0] * sim.nu), (T, 1))
        finit = np.tile(np.array([0] * sim.nf), (T, 1))

        traj = pack_trajectory(sim, qinit, vinit, finit, uinit, None)
        return traj

    def manipulator_params(x):
        q1, q2, q1dot, q2dot = x

        M = np.array([[sim.params['alpha'] + 2 * sim.params['beta'] * ps.cos(q2),
                       sim.params['delta'] + sim.params['beta'] * ps.cos(q2)],           
                      [sim.params['delta'] + sim.params['beta'] * ps.cos(q2),              
                       sim.params['delta']]])
        C = np.array([[-sim.params['beta'] * ps.sin(q2) * q2dot,                            
                       -sim.params['beta'] * ps.sin(q2) * (q1dot + q2dot)],                
                      [sim.params['beta'] * ps.sin(q2) * q1dot,                         
                       0]])

        return np.squeeze(M), np.squeeze(C)

    def jacobian(x):
        q1,q2,_,_ = x # Unpack Joints                                            
        J = np.array([[-sim.params['l'] * (ps.sin(q1) + ps.sin(q1 + q2)),                 
                       -sim.params['l'] * (ps.sin(q1 + q2))],                            
                      [sim.params['l'] * (ps.cos(q1) + ps.cos(q1 + q2)),                  
                       sim.params['l'] * (ps.cos(q1 + q2))]])

        return np.squeeze(J)               

    def fkin(q, body):
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
            x0 = sim.params['Cx'] - sim.params['Lx']
        elif (body[0] == 'r'):
            x0 = sim.params['Cx'] + sim.params['Lx']
        else:
            raise ValueError('Body should only be "l0,l1,l2,r0,r1,r2"')

        y0 = sim.params['Cy'] + sim.params['Ly']

        if (body[1] == '0'):
            return np.array([x0, y0])
        
        elif (body[1] == '1'):
            x1 = x0 + sim.params['l'] * (ps.cos(q1))
            y1 = y0 + sim.params['l'] * (ps.sin(q1))

            return np.array([x1, y1])
        
        elif (body[1] == '2'):
            x1 = x0 + sim.params['l'] * (ps.cos(q1))
            y1 = y0 + sim.params['l'] * (ps.sin(q1))
            x2 = x1 + sim.params['l'] * ps.cos(q1 + q2)
            y2 = y1 + sim.params['l'] * ps.sin(q1 + q2)
            
            return np.array([x2, y2])
        else:
            raise ValueError('Body should only be "l0,l1,l2,r0,r1,r2"')
    
    ''' 
    -------------------------------------------------------------------------
    Set parameter options
    -------------------------------------------------------------------------
    '''


    if (options == None) or (options['tol'] == None):
        tol = 0.0
    else:
        tol = options['tol']

    if (options == None) or (options['solver'] == None):
        solver = 'snopt'
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
        initial_guess = generate_initial_traj(sim, xi, xf, T, method='linear')
    else:
        initial_guess = options['initial_guess']

    qinit, vinit, finit, uinit = unpack_trajectory(sim, initial_guess)

    ''' 
    -------------------------------------------------------------------------
    Define mathematical Program
    -------------------------------------------------------------------------
    '''

    prog = MathematicalProgram()

    q = prog.NewContinuousVariables(rows=T+1, cols=sim.nq, name='q')
    v = prog.NewContinuousVariables(rows=T+1, cols=sim.nq, name='v')
    u = prog.NewContinuousVariables(rows=T, cols=sim.nu, name='u')
    #contact = prog.NewContinuousVariables(rows=T, cols=sim.nf, name='lambda')

    ''' 
    alpha = prog.NewContinuousVariables(rows=T, cols=sim.nf, name='alpha')
    beta = prog.NewContinuousVariables(rows=T, cols=sim.nf, name='beta')
    '''

    q_b = q[:,0:3]
    q_l = q[:,3:5]
    q_r = q[:,5:7]

    v_b = v[:,0:3]
    v_l = v[:,3:5]
    v_r = v[:,5:7]

    u_l = u[:,0:2]
    u_r = u[:,2:4]

    x_b = np.hstack((q_b, v_b))
    x_l = np.hstack((q_l, v_l))
    x_r = np.hstack((q_r, v_r))

    #lambda_l = contact[:,0:2]
    #lambda_r = contact[:,2:4]

    # Add initial Condition Constraint
    prog.AddConstraint(eq(q[0], np.array(xi[0:sim.nq])))
    prog.AddConstraint(eq(v[0], np.array(xi[sim.nq:(2*sim.nq)])))

    # Add Final condition constraint
    prog.AddConstraint(eq(q[T], np.array(xf[0:sim.nq])))
    prog.AddConstraint(eq(v[T], np.array(xf[sim.nq:(2*sim.nq)])))

    # Add Dynamics Constraints
    for t in range(T):
        # Velocity to Forces Constraints
        prog.AddConstraint(eq(q[t+1], (q[t] + sim.params['h'] * v[t+1])))

        # Forces to Velocity Constraints
        # 1. Add Constraints for manipuland
        
        '''
        prog.AddConstraint(eq(v_b[t+1,0:2], (v_b[t,0:2] + sim.params['h'] * (
            (1. / sim.params['mb']) * (
                lambda_l[t] + lambda_r[t] - sim.params['cb'] * v_b[t,0:2])))))
        '''

        prog.AddConstraint(eq(v_b[t+1,0:2], (v_b[t,0:2] + sim.params['h'] * (
            (1. / sim.params['mb']) * (
                - sim.params['cb'] * v_b[t,0:2])))))


        # Skip and just set to zero since rotation won't change
        prog.AddConstraint(v_b[t+1,2] == v_b[t,2] + sim.params['h'] * 0.)

        # 2. Add Constraints for left manipulator (implicit)

        M_l, C_l = manipulator_params(x_l[t])
        J_l = jacobian(x_l[t])

        # M(q_t)v_{t+1} = M(q_t)v_t - C(q_t,v_t)v_t + tau_t + J(q_t)^T \lambda_t
        # Sign on multiplier term negated due to contact force definition.
        
        prog.AddConstraint(eq(M_l.dot(v_l[t+1]), \
                             (M_l.dot(v_l[t]) + sim.params['h'] * (- C_l.dot(v_l[t]) +
                              u_l[t])))) #- np.transpose(J_l).dot(lambda_l[t])))))

        # 3. Add Constraints for right manipulator

        M_r, C_r = manipulator_params(x_l[t])
        J_r = jacobian(x_r[t])

        prog.AddConstraint(eq(M_r.dot(v_r[t+1]),\
                             (M_r.dot(v_r[t]) + sim.params['h'] * (- C_r.dot(v_r[t]) +
                              u_r[t])))) #- np.transpose(J_r).dot(lambda_r[t])))))

        # 4. Add Contact Constraints

        '''
        p_l = fkin(x_l[t], 'l2')
        p_r = fkin(x_r[t], 'r2')
        p_b = x_b[t,0:2]

        p_lb = p_l - p_b
        p_rb = p_r - p_b

        d_l = np.sqrt(p_lb[0] ** 2. + p_lb[1] ** 2.)
        d_r = np.sqrt(p_rb[0] ** 2. + p_rb[1] ** 2.)

        # Following Posa's trick, add alpha and beta variables.

        prog.AddConstraint(ge(alpha[t], 0))
        prog.AddConstraint(ge(beta[t], 0))
        prog.AddConstraint(eq(alpha[t,0:2], lambda_l[t]))
        prog.AddConstraint(eq(alpha[t,2:4], lambda_r[t]))

        prog.AddConstraint(eq(beta[t,0:2], lambda_l[t] + sim.params['k'] * (
            (p_lb / d_l) * (sim.params['R'] + sim.params['r'] - d_l))))
        prog.AddConstraint(eq(beta[t,2:4], lambda_r[t] + sim.params['k'] * (
            (p_rb / d_r) * (sim.params['R'] + sim.params['r'] - d_r))))

        # Complementarity Constraints.
        prog.AddConstraint(alpha[t,0] * beta[t,0] <= tol)
        prog.AddConstraint(alpha[t,1] * beta[t,1] <= tol)
        prog.AddConstraint(alpha[t,2] * beta[t,2] <= tol)
        prog.AddConstraint(alpha[t,3] * beta[t,3] <= tol)
        '''

        if (contact_max is not None):
            prog.AddConstraint(le(contact[t], contact_max))
            prog.AddConstraint(ge(contact[t], -contact_max))
        if (input_max is not None):
            prog.AddConstraint(le(u[t], input_max))
            prog.AddConstraint(ge(u[t], -input_max))

        # Add Costs
        prog.AddCost(u[t].dot(u[t]))

    ''' 
    -------------------------------------------------------------------------
    Set up initial guess of the program
    -------------------------------------------------------------------------
    '''

    initial_guess = np.empty(prog.num_vars())

    prog.SetDecisionVariableValueInVector(q, qinit, initial_guess)
    prog.SetDecisionVariableValueInVector(v, vinit, initial_guess)
    prog.SetDecisionVariableValueInVector(u, uinit, initial_guess)
    #prog.SetDecisionVariableValueInVector(contact, finit, initial_guess)

    '''
    # Set Alpha and beta to satisfy the equality constraints defined by program
    alphainit = np.zeros((T,4))
    alphainit[:,0:2] = finit[:,0:2]
    alphainit[:,2:4] = finit[:,2:4]

    betainit = np.zeros((T,4))

    for t in range(T):
        x_b, x_l, x_r = sim.unpack_states(np.concatenate((qinit[t], vinit[t])))

        p_l = sim.fkin(x_l, 'l2')
        p_r = sim.fkin(x_r, 'r2')
        p_b = x_b[0:2]

        p_lb = p_l - p_b
        p_rb = p_r - p_b

        d_l = np.sqrt(p_lb[0] ** 2. + p_lb[1] ** 2.)
        d_r = np.sqrt(p_rb[0] ** 2. + p_rb[1] ** 2.)

        betainit[t,0:2] = finit[t,0:2] + sim.params['k'] * (
            p_lb / d_l) * (sim.params['R'] + sim.params['r'] - d_l)
        betainit[t,2:4] = finit[t,2:4] + sim.params['k'] * (
            p_rb / d_r) * (sim.params['R'] + sim.params['r'] - d_r)

    prog.SetDecisionVariableValueInVector(alpha, alphainit, initial_guess)
    prog.SetDecisionVariableValueInVector(beta, betainit, initial_guess)
    '''

    ''' 
    -------------------------------------------------------------------------
    Solve the Program
    -------------------------------------------------------------------------
    '''

    if (solver == 'ipopt'):
        solver_cls = IpoptSolver()
    elif (solver == 'snopt'):
        solver_cls = SnoptSolver()
    else:
        raise ValueError("Solver is not supported! Use ipopt or snopt")

    init_time = time.time()
    result = solver_cls.Solve(prog, initial_guess)
    solver_time = time.time() - init_time

    if (solver == 'ipopt'):
        print("Ipopt Solver Status: ", result.get_solver_details().status,
              ", meaing ", result.get_solver_details().ConvertStatusToString())
    elif (solver == 'snopt'):
        val = result.get_solver_details().info
        status = snopt_status(val)
        print("Snopt Solver Status: ", result.get_solver_details().info,
              ", meaning ", status)

        
    ''' 
    -------------------------------------------------------------------------
    Pack and Return
    -------------------------------------------------------------------------
    '''

    sol = result.GetSolution()
    q_opt = result.GetSolution(q)
    v_opt = result.GetSolution(v)
    f_opt = result.GetSolution(contact)
    u_opt = result.GetSolution(u)

    traj_opt = pack_trajectory(sim, q_opt, v_opt, f_opt, u_opt, solver_time)

    return traj_opt

    
    
        
    








        
                        




    

