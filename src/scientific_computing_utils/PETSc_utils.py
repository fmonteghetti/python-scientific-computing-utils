# -*- coding: utf-8 -*-
"""
Utility functions for petsc4py.

"""
import numpy as np
import scipy.sparse as sp
from petsc4py import PETSc

def convert_scipy_to_PETSc(A,comm=None):
    """
    Convert from scipy sparse to PETSc sparse.

    Parameters
    ----------
    A : scipy.sparse matrix
        
    comm : MPI communicator PETSc.Comm, optional
        
    Returns
    -------
    A_petsc : PETSc.Mat
    
    """
    if A.format!="csr":
        A_csr = A.tocsr(copy=False)
    else:
        A_csr = A
        
    A_petsc = PETSc.Mat().createAIJ(size=A_csr.shape,
                                   csr=(A_csr.indptr, A_csr.indices,
                                       A_csr.data),comm=comm)
    return A_petsc 
    

def convert_PETSc_to_scipy(A):
    """
    Convert from PETSc.Mat to scipy sparse (csr).

    Parameters
    ----------
    A : PETSc.Mat


    Returns
    -------
    A_scipy : scipy.sparse.csr.csr_matrix
        

    """
    (indptr,indices,val) = A.getValuesCSR()
    A_scipy = sp.csr_matrix((val, indices, indptr), shape=A.size)
    return A_scipy

def set_diagonal_entries(A):
    """
    Set all diagonal entries of A without modifying existing entries. Use in
    last resort, as this is an extremely slow function.

    Parameters
    ----------
    A : PETSc.Mat

    """

    x=A.createVecRight(); x.setArray(0)
    A.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR,False)
    A.setDiagonal(x,addv=True)
    A.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR,True)

def get_cleared_options_db():
    """ Clear the option database and return a pointer to it. """
    OptDB = PETSc.Options() # pointer to THE option database
    for key in OptDB.getAll(): OptDB.delValue(key) # Clear database
    return OptDB


def options_db_clear_prefix(OptDB,prefix):
    """ Remove parameters with given prefix from options database. 
    prefix: string (e.g. 'ts', 'ksp') """
    P = len(prefix)
    idx = slice(0,P)
    for key in OptDB.getAll():
        if (len(key)>=P) and key[idx]==prefix:
            OptDB.delValue(key)

def options_db_set_all(OptDB,entries):
    """ Set all entires in dictionary entries in options database. """
    for key,val in entries.items():
        OptDB.setValue(key,val)


######################### PETSc TS ###########################################
# 
# DAE time-integration guidelines
# --------------------------------
# 
#   1 General guidelines
#   ---------------------
# - DAE can be viewed as an 'infinitely stiff' ODE. Suitable schemes:
#   -> A-Stable methods (stability region = entire half-plane)
#   -> L-Stable and/or stiffly accurate = subset of A-Stable methods, well-suited
#   for DAE.
# - Low-order implicit schemes (be/cn/theta) are less sensitive to non-consistent
# initial conditions (typically, due to non-consistent algebraic variables),
# which implies:
#   -> Used with constant time step, they can yield stable and accurate results,
#   albeit as a large cost for long-time computation.
#   -> They can be used to compute approximately consistent initial state.
# - To use high-order implicit scheme with time adaption:
#   -> Use consistent initial state (very sensitive)
#   -> Lagrange parameters (or even all algebraic variables) should be excluded
# from Local Truncation Error, otherwise time adaption fails.
#   -> Ensure ts_rtol and ts_atol have sensible values.
# - Rosenberg-W schemes such as 'ra34pw2' gives excellent performance 
# and accuracy for DAE, see 
# [S. Abhyankar, E. Constantinescu, and A. Flueck. 2017. Variable-step 
# multi-stage integration methods for fast and accurate power system dynamics 
# simulation. IREP'17.]
# 
#   2 Relevant TS and SNES options
#   ------------------------------   
    # Newton iteration (nonlinear solve)
# OptDB["snes_atol"] = 1e-6
# OptDB["snes_rtol"] = 1e-6
# OptDB["snes_stol"] = 1e-6
# OptDB["snes_max_it"] = 1 
# OptDB["snes_max_funcs"] = 10
# OptDB["snes_lag_jacobian"] = 5
# OptDB["snes_lag_jacobian_persists"] = True
    # Time-stepping tolerance
# OptDB["ts_rtol"] = 1e-4
# OptDB["ts_atol"] = 1e-4
    # Time step adaption
# OptDB["ts_max_snes_failures"] = -1 # avoids failure with stiff problem
# OptDB["ts_max_reject"] = 5 # -1 for unlimited
# OptDB["ts_adapt_type"] = "basic"/"dsp"/"none"
    # Low-order implicit scheme
# OptDB["ts_type"] = "cn", "beuler"
        # BDF
# OptDB["ts_type"] = "bdf"
# OptDB["ts_bdf_order"] = 2
        # ARKIMEX schemes
# OptDB["ts_type"] = PETSc.TS.Type.ARKIMEX
# OptDB["ts_equation_type"] = PETSc.TS.EquationType.IMPLICIT
# OptDB["ts_type_arkimex"] = 'bpr3'
        # Rosenberg-W linear implicit schemes
# OptDB["ts_type"] = PETSc.TS.Type.ROSW
# OptDB["ts_equation_type"] = PETSc.TS.EquationType.IMPLICIT
# OptDB["ts_type_rosw"] = "2p", "rodas3", "ra34pw2"
    # Linear solver: GMRES
# OptDB["ksp_type"] = "gmres" # preonly, gmres
# OptDB["pc_type"] = "jacobi" # none, jacobi, ilu, icc
     # Linear solver: direct sparse solver
# OptDB["ksp_type"] = "preonly"
# OptDB["pc_type"] = "lu"
# OptDB["pc_factor_mat_solver_type"] = "mumps"

import time

class Fully_implicit_DAE(object):
    """ DAE under a fully implicit form F(t,y,yd) = 0 suitable for
    integration with PETSc TS fully implicit schemes. """ 
    
    def __init__(self,N,name,idx_alg=None):
        self.name = name        # str: name
        self.N = N              # int: number of unknowns
        self.idx_alg = idx_alg  # list(int): algebraic variables
        self.F = 0              # PETSc.Vec(): vector used during time-integration
        self.J = 0              # PETSc.Mat(): matrix used during time-integration
           
    def init(self):
        """ Initialize. Call before new time integration. """
            # History
        self.init_history()

    def IFunction(self, ts, t, y, yd, F):
        """ Evaluate residual vector F(t,y,yd)."""
        pass
        
    def IJacobian(self, ts, t, y, yd, c, Amat, Pmat):
        """ Evaluate jacobian matrix J = c*J_yd + J_y."""
        pass
    
    def init_history(self):
        """ Initialize history structure. """
        self.history = dict(t=list(),y=list())
        self.t_start = 0

    def monitor(self, ts, i, t, y, dt_export=1):
        """ Monitor to use during iterations. """
        if self.history['t']:
            lastt  = self.history['t'][-1]            
        else:
            lastt = t-2*dt_export
            self.t_start = time.time()
        if (t > lastt + dt_export) or (i==-1):
            print(f"i={i:8d} t={t:8g} * dt={ts.getTimeStep():8g} ({int(time.time()-self.t_start)}s)")
            self.history['t'].append(t)
            self.history['y'].append(y.copy())
        else: # (i<10): 
            print(f"i={i:8d} t={t:8g}   dt={ts.getTimeStep():8g} ({int(time.time()-self.t_start)}s)")
            
    def get_vector(self):
        """ Get new residual-size vector. """
        x = PETSc.Vec().createWithArray(np.zeros(self.N,))
        return x

def TS_exclude_var_from_lte(dae,ts):
    """ Exclude algebraic variables from local truncation error. 
    
    Inputs
    -------
    dae: Fully_implicit_DAE
    
    ts: PETSc.TS
    
    """
    atol = ts.getTolerances()[1]
    atol_v = atol*np.ones((dae.N,))
    atol_v[dae.idx_alg] = np.inf # inf for algebraic variables
    atol_v_petsc = dae.get_vector()
    atol_v_petsc.setArray(atol_v)
    ts.setTolerances(atol=atol_v_petsc)




def TS_integration_dae(dae,x0,dt,tf,dt_export=None,
                        cinit=False,cinit_dt=1e-2,cinit_nstep=1):
    """ Time-integration of DAE under fully implicit form F(t,y,yd)=0.
    
    Inputs
    -------
    dae: Fully_implicit_DAE
    
    x0: PETSc.Vec
        Initial condition. This vector is used throughout the time integration.
        
    dt,tf: real
        Initial time step and final time.
    
    dt_export: real (Optional)
        Time step for saving solution.
        
    cinit, cinit_dt, cinit_nstep: bool, real, int
        Parameters related to consistent initialization.
    
    """
    tc = 0
    if dt_export is None:
        dt_export = dt
    dae.init()
    monitor = lambda ts, i, t, x: dae.monitor(ts, i, t, x, dt_export=dt_export)
    OptDB = PETSc.Options()
    import time
        # -- Consistent initialization
    if cinit==True:
            # Clear options related to ts and snes
        OptDB_bak = OptDB.getAll()
        options_db_clear_prefix(OptDB,"ts")
        options_db_clear_prefix(OptDB,"snes")
        ts = PETSc.TS().create()
        ts.setMonitor(monitor)
        ts.setIFunction(dae.IFunction, dae.F)
        ts.setIJacobian(dae.IJacobian, dae.J)
        ts.setMaxSNESFailures(-1)
        ts.setFromOptions()
        ts.setTime(0.0)
        ts.setMaxTime(cinit_nstep*cinit_dt)
        ts.setTimeStep(cinit_dt)
        ts.setMaxSteps(cinit_nstep)
        ts.setExactFinalTime(PETSc.TS.ExactFinalTime.MATCHSTEP)
        ts.setType(PETSc.TS.Type.BEULER)
        print(f"Consistent initialization: {cinit_nstep} step(s) of {ts.getType()}.")
        start = time.time()
        ts.solve(x0)
        tc = ts.getTime()
        print(f"Done (t={tc:8g}).\nElapsed time: {time.time()-start:1.4g}s")
        ts.destroy()
        del ts
            # Restore user-defined options
        options_db_set_all(OptDB,OptDB_bak)
        # -- Main integration
    ts = PETSc.TS().create()
    ts.setMonitor(monitor)
    ts.setIFunction(dae.IFunction, dae.F)
    ts.setIJacobian(dae.IJacobian, dae.J)
            # Default options
    ts.setTime(tc)
    ts.setMaxTime(tf)
    OptDB['ts_dt'] = dt # initial time step
    ts.setMaxSNESFailures(-1) # avoid unecessary divergence
    ts.setExactFinalTime(PETSc.TS.ExactFinalTime.MATCHSTEP)
            # User-defined options
    ts.setFromOptions()
    snes = ts.getSNES()
            # Print some options    
    print(f"Scheme: {ts.getType()}")
    print(f"TS atol: {ts.atol:1.1e}, rtol:{ts.rtol:1.1e}")
    print(f"SNES atol: {snes.atol:1.1e}, rtol:{snes.rtol:1.1e}, stol: {snes.stol}")
    print(f"SNES max_it: {snes.max_it:d}")
    TS_exclude_var_from_lte(dae,ts)
    start = time.time()    
    ts.solve(x0)
    print(f"Elapsed time: {time.time()-start:1.4g}s")
    print(f"Steps: {ts.getStepNumber()} ({ts.getStepRejections()} rejected, {ts.getSNESFailures()} Nonlinear solver failures)")
    print(f"Nonlinear iterations: {ts.getSNESIterations()}, Linear iterations: {ts.getKSPIterations()}")
    ts.reset()
    ts.destroy()