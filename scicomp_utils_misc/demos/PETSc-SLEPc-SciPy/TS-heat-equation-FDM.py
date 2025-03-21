# Example adapted from petsc4py provided ones.
# Solves a Heat equation on a periodic domain, discretized using FDM.
# - Formulated as implicit problem: F(t,x,xdot)=0
# - Analytical Jacobian provided
# - Monitor to store solution
# - Linear solver customized
# - Uses raw VecScatter

from __future__ import division

import numpy

import sys, petsc4py

petsc4py.init(sys.argv)
from petsc4py import PETSc
from mpi4py import MPI

# Clear options database
Opt = PETSc.Options()  # pointer to THE option database
for key in Opt.getAll():
    Opt.delValue(key)
print(Opt.view())


class Heat(object):
    def __init__(self, comm, N):
        self.comm = comm
        self.N = N  # global problem size
        self.h = 1 / N  # grid spacing on unit interval
        self.n = N // comm.size + int(
            comm.rank < (N % comm.size)
        )  # owned part of global problem
        self.start = comm.exscan(self.n)
        if comm.rank == 0:
            self.start = 0
        gindices = (
            numpy.arange(self.start - 1, self.start + self.n + 1, dtype=int) % N
        )  # periodic
        self.mat = PETSc.Mat().create(comm=comm)
        size = (self.n, self.N)  # local and global sizes
        self.mat.setSizes((size, size))
        self.mat.setFromOptions()
        self.mat.setPreallocationNNZ(
            (3, 1)
        )  # Conservative preallocation for 3 "local" columns and one non-local

        # Allow matrix insertion using local indices [0:n+2]
        lgmap = PETSc.LGMap().create(list(gindices), comm=comm)
        self.mat.setLGMap(lgmap, lgmap)

        # Global and local vectors
        self.gvec = self.mat.createVecRight()
        self.lvec = PETSc.Vec().create(comm=PETSc.COMM_SELF)
        self.lvec.setSizes(self.n + 2)
        self.lvec.setUp()
        # Configure scatter from global to local
        isg = PETSc.IS().createGeneral(list(gindices), comm=comm)
        self.g2l = PETSc.Scatter().create(self.gvec, isg, self.lvec, None)

        self.tozero, self.zvec = PETSc.Scatter.toZero(self.gvec)
        self.history = []

        if False:  # Print some diagnostics
            print(
                "[%d] local size %d, global size %d, starting offset %d"
                % (comm.rank, self.n, self.N, self.start)
            )
            self.gvec.setArray(numpy.arange(self.start, self.start + self.n))
            self.gvec.view()
            self.g2l.scatter(self.gvec, self.lvec, PETSc.InsertMode.INSERT)
            for rank in range(comm.size):
                if rank == comm.rank:
                    print("Contents of local Vec on rank %d" % rank)
                    self.lvec.view()
                comm.barrier()

    def get_initial_condition(self, t, x):
        assert t == 0.0, "only for t=0.0"
        coord = numpy.arange(self.start, self.start + self.n) / self.N
        x.setArray((numpy.abs(coord - 0.5) < 0.1) * 1.2)

    def IFunction(self, ts, t, x, xdot, f):
        # Compute F(t,U,U_t) where F() = 0 is the DAE to be solved.
        # t 	- time at step/stage being solved
        # x 	- state vector
        # x_dot - time derivative of state vector
        # F 	- function vector
        self.g2l.scatter(
            x, self.lvec, PETSc.InsertMode.INSERT
        )  # lvec is a work vector
        h = self.h
        with self.lvec as u, xdot as udot:
            f.setArray(
                udot * h + 2 * u[1:-1] / h - u[:-2] / h - u[2:] / h
            )  # Scale equation by volume element

    def IJacobian(self, ts, t, x, xdot, a, Amat, Pmat):
        # Compute Jacobian matrix dF/dU + a*dF/dU_t
        #   t   - time at step/stage being solved
        # 	x 	- state vector
        # 	x_dot 	- time derivative of state vector
        # 	a 	- shift
        # 	Amat 	- Jacobian matrix
        # 	Pmat 	- matrix used for constructing preconditioner,
        #             usually the same as Amat
        # Common practice is to build analytical Jacobian in Pmat
        h = self.h
        for i in range(self.n):
            lidx = i + 1
            gidx = self.start + i
            Pmat.setValuesLocal(
                [lidx],
                [lidx - 1, lidx, lidx + 1],
                [-1 / h, a * h + 2 / h, -1 / h],
            )
        Pmat.assemble()
        if Amat != Pmat:
            print(f"Operator different from preconditioning")
            Amat.assemble()
        return True  # same nonzero pattern

    def monitor(self, ts, i, t, x, dt_export=1):
        if self.history:
            lasti, lastt, lastx = self.history[-1]
            if (i > 10) and (t < lastt + dt_export):
                return
        print(f"i={i} t={t}")
        self.tozero.scatter(x, self.zvec, PETSc.InsertMode.INSERT)
        xx = self.zvec[:].tolist()
        self.history.append((i, t, xx))

    def plotHistory(self):
        try:
            from matplotlib import pylab, rcParams
        except ImportError:
            print("matplotlib not available")
            raise SystemExit
        rcParams.update({"text.usetex": True, "figure.figsize": (10, 6)})
        # rc('figure', figsize=(600,400))
        pylab.title("Heat: TS \\texttt{%s}" % ts.getType())
        x = numpy.arange(self.N) / self.N
        for i, t, u in self.history:
            pylab.plot(x, u, label="step=%d t=%8.2g" % (i, t))
        pylab.xlabel("$x$")
        pylab.ylabel("$u$")
        # pylab.legend(loc='upper right')


#        pylab.savefig('heat-history.png')
# pylab.show()

# Pointer to option database
OptDB = PETSc.Options()
# Numerical scheme
OptDB["ts_type"] = "bdf"  # bdf, beuler, cn
# ARKIMEX schemes
# OptDB["ts_type"] = PETSc.TS.Type.ARKIMEX
# OptDB["ts_type_arkimex"] = PETSc.TS.ARKIMEXType.ARKIMEX3
# ROSW Linear implicit schemes
# OptDB["ts_type"] = PETSc.TS.Type.ROSW
# OptDB["ts_type_rosw"] = "ra34pw2"
# Linear solver
OptDB["ksp_type"] = "gmres"  # preonly, gmres
# Preconditioner
OptDB["pc_type"] = "ilu"  # none, jacobi, ilu, icc
# Algebraic multigrid
# OptDB["pc_type"] = "gamg"
# OptDB['mg_coarse_pc_type'] = 'svd'
# OptDB['mg_levels_pc_type'] = 'sor'
# Direct sparse solver
# OptDB["ksp_type"] = "preonly"
# OptDB["pc_type"] = "lu"
# OptDB["pc_factor_mat_solver_type"] = "mumps"
# Create implicit ODE problem
ode = Heat(MPI.COMM_WORLD, OptDB.getInt("n", int(1e4)))
# Create Time-Stepper (TS)
ts = PETSc.TS().create(comm=ode.comm)
# Monitoring (store time steps)
ts.setMonitor(lambda ts, i, t, x: ode.monitor(ts, i, t, x, dt_export=0.01))
# DAE F(t,x,x_dot) = 0
ts.setIFunction(ode.IFunction, ode.gvec)
ts.setIJacobian(ode.IJacobian, ode.mat)
# Nonlinear iterations
# Tolerances
ts.setMaxSNESFailures(
    -1
)  # allow an unlimited number of failures (step will be rejected and retried)
snes = ts.getSNES()  # Nonlinear solver
snes.setTolerances(
    max_it=10
)  # Stop nonlinear solve after 10 iterations (TS will retry with shorter step)
# Linear solver
ksp = snes.getKSP()
# Preconditioner
pc = ksp.getPC()
# Time options
x = ode.gvec.duplicate()
ode.get_initial_condition(0.0, x)
ts.setTime(0.0)
ts.setTimeStep(ode.h**2)
ts.setMaxTime(1)
# ts.setMaxSteps(100)
ts.setExactFinalTime(PETSc.TS.ExactFinalTime.INTERPOLATE)
# Apply run-time options, e.g. -ts_adapt_monitor -ts_type arkimex -snes_converged_reason
ts.setFromOptions()
import time

start = time.time()
ts.solve(x)
print(f"Elapsed time: {time.time() - start:1.4g}s")
# Post-processing
if ode.comm.rank == 0:
    print(
        f"Steps: {ts.getStepNumber()} ({ts.getStepRejections()} rejected, {ts.getSNESFailures()} Nonlinear solver failures)"
    )
    print(
        f"Nonlinear iterations: {ts.getSNESIterations()}, Linear iterations: {ts.getKSPIterations()}"
    )

if OptDB.getBool("plot_history", True) and ode.comm.rank == 0:
    ode.plotHistory()
# ts.view()
