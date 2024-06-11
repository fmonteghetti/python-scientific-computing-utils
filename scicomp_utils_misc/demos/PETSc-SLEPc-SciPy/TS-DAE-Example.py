# Time integration of high-dimensional index-1 DAE usign PETSc TS
# State y = (y_diff,y_alg)
# dy_diff/dt = -y_diff + src_diff
# y_a = src_alg

# Improvement ideas:
# - Compute Jacobian only once: compute it in constructor, give it to
# setIJacobian, then don't do anything in IJacobian.
# - Monitor directly computes an array

import numpy as np
import matplotlib.pyplot as plt
import time

import sys, petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
Opt = PETSc.Options() # pointer to THE option database
for key in Opt.getAll(): Opt.delValue(key) # Clear database

class DAE_description(object):
    def __init__(self,comm,src_diff,src_alg,name):
        self.comm = comm # not used
        self.N_diff = src_diff.size
        self.N_alg = src_alg.size
        self.src_diff = src_diff
        self.src_alg = src_alg
        self.algvar = np.ones((self.N_diff+self.N_alg,),dtype="bool")
        self.algvar[self.N_diff+np.array(range(self.N_alg))] = False
        self.name = name
        self.history = []
            # Jacobian matrix setup
        self.Amat = PETSc.Mat().create()
        self.Amat.setSizes([N_diff+N_alg,N_diff+N_alg])
        self.Amat.setType('aij')
        self.Amat.setPreallocationNNZ(1) # number of non-zero entries per row
        self.Arow = np.r_[0:(self.N_diff+self.N_alg+1)].astype('int32')
        self.Acol = self.Arow[:-1].astype('int32')
        self.mask_diff = self.Acol<self.N_diff
            # Residual Vector setup
        self.Fvec = self.Amat.createVecRight()
    def IFunction(self, ts, t, y, yd, f):
        # y, yd, f: PETSc.Vec (already allocated)
        res_diff = yd.array[self.algvar] + y.array[self.algvar] - self.src_diff
        res_alg = y.array[~self.algvar] - self.src_alg
        f.setArray(np.concatenate((res_diff,res_alg)))
    def IJacobian(self, ts, t, y, yd, a, Amat, Pmat):
        # y, yd: PETSc.Vec (already allocated)
        # Amat, Pmat: PETSc.Mat (already allocated)
            # Build method 1
        # for i in range(self.N_diff): Pmat[i,i] = 1+a
        # for i in range(self.N_alg): Pmat[N_diff+i,N_diff+i] = 1
        # Pmat.assemblyBegin(); Pmat.assemblyEnd()
            # Build method 2 (fast)
        Pmat.setValuesCSR(self.Arow,
                          self.Acol,
                          (1+a)*self.mask_diff+(1)*(~self.mask_diff))
        Pmat.assemblyBegin(); Pmat.assemblyEnd()
        if Amat != Pmat:
            print("Operator different from preconditioning")
            Amat.assemble()
    def monitor(self, ts, i, t, x, dt_export=1):
        if self.history:
            lasti, lastt, lastx = self.history[-1]
            if (i>10) and (t < lastt + dt_export): return
        print(f"i={i} t={t:2.3g}")
#        self.history_x = np.vstack((history_x,x.array))
        self.history.append((i, t, x.array.flatten()))
# ----------- User input
    # Dimension
N_diff = int(1e5)
N_alg = int(1e5)
    # Source term
src_diff = np.ones((N_diff,))
src_alg = 1*(np.random.rand(N_alg))
    # Initial condition (Consistency of algebraic part required)
y_diff_0 = 4*np.ones((N_diff,))
y_alg_0 = src_alg
    # Solver options
OptDB = PETSc.Options() # Pointer to opSet up options
    # Numerical scheme
OptDB["ts_type"] = "bdf" # bdf, beuler, cn
        # ARKIMEX schemes
# OptDB["ts_type"] = PETSc.TS.Type.ARKIMEX
# OptDB["ts_type_arkimex"] = PETSc.TS.ARKIMEXType.ARKIMEX3
        # ROSW Linear implicit schemes
# OptDB["ts_type"] = PETSc.TS.Type.ROSW
# OptDB["ts_type_rosw"] = "ra34pw2"
    # Linear solver
#OptDB["ksp_type"] = "gmres" # preonly, gmres
    # Preconditioner
# OptDB["pc_type"] = "ilu" # none, jacobi, ilu, icc
            # Algebraic multigrid
# OptDB["pc_type"] = "gamg"
# OptDB['mg_coarse_pc_type'] = 'svd'
# OptDB['mg_levels_pc_type'] = 'sor'
            # Direct sparse solver
# OptDB["ksp_type"] = "preonly"
# OptDB["pc_type"] = "lu"
# OptDB["pc_factor_mat_solver_type"] = "mumps"
# -----------
OptDB.view()
ode = DAE_description(PETSc.COMM_WORLD,src_diff,src_alg,name = 'High-dim index-1 DAE')
    # Create Time-Stepper (TS)
ts = PETSc.TS().create(comm=ode.comm)
    # Monitoring (store time steps)
ts.setMonitor(lambda ts, i, t, x: ode.monitor(ts, i, t, x, dt_export=0.01))
    # DAE F(t,x,x_dot) = 0
ts.setIFunction(ode.IFunction, ode.Fvec)
ts.setIJacobian(ode.IJacobian, ode.Amat)
    # Nonlinear iterations
        # Tolerances
ts.setMaxSNESFailures(-1)       # allow an unlimited number of failures (step will be rejected and retried)
snes = ts.getSNES()             # Nonlinear solver
snes.setTolerances(max_it=10)   # Stop nonlinear solve after 10 iterations (TS will retry with shorter step)
    # Time options
tf = 5
ts.setTime(0.0)
ts.setMaxTime(tf)
ts.setTimeStep(tf/100)
#ts.setMaxSteps(100)
ts.setExactFinalTime(PETSc.TS.ExactFinalTime.INTERPOLATE)
    # Apply run-time options, e.g. -ts_adapt_monitor -ts_type arkimex -snes_converged_reason
ts.setFromOptions()
start = time.time()
x = ode.Fvec.duplicate()
x.setArray(np.concatenate((y_diff_0,y_alg_0)))
ts.solve(x)
print(f"Elapsed time: {time.time()-start:1.4g}s")
print(f"Steps: {ts.getStepNumber()} ({ts.getStepRejections()} rejected, {ts.getSNESFailures()} Nonlinear solver failures)")
print(f"Nonlinear iterations: {ts.getSNESIterations()}, Linear iterations: {ts.getKSPIterations()}")
    # Convert history to plottable array
N = len(ode.history); tp = np.zeros((N,)); yp = np.zeros((N,ode.history[0][2].size))
for i in range(len(ode.history)):
    tp[i] = ode.history[i][1]
    yp[i,:] = ode.history[i][2]

i_p1 = N_diff-1; i_p2 = N_diff
fig = plt.figure()
ax = fig.add_subplot(2,1,1)
ax.plot(tp.flatten(),yp[:,i_p1],label=f"ODE")
ax.set_xlabel('t')
ax.set_ylabel('x (ODE part)')
ax.set_title(ode.name)
ax = fig.add_subplot(2,1,2)
ax.plot(tp.flatten(),yp[:,i_p2],label=f"DAE")
ax.set_xlabel('t')
ax.set_ylabel('x (DAE part)')
#%% Individually test functions for debug purposes
N_diff = int(1e1)
N_alg = int(1e1)
src_diff = np.ones((N_diff,))
src_alg = 1*(np.random.rand(N_alg))
imp_mod = DAE_description(PETSc.COMM_WORLD,src_diff,src_alg,name = 'High-dim index-1 DAE')
t = 0
ts = []
y = PETSc.Vec().createWithArray(np.ones((N_diff+N_alg,)))
yd = PETSc.Vec().createWithArray(2*np.ones((N_diff+N_alg,)))
f = imp_mod.Fvec
imp_mod.IFunction(ts, t, y, yd, f)
c = 0.1
Amat = imp_mod.Amat
Pmat = Amat
imp_mod.IJacobian(ts, t, y, yd, c, Amat, Pmat)