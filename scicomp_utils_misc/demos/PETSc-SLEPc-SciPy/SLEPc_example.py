# -*- coding: utf-8 -*-
"""
Script that demonstrates the use of slepc in python.

This is a temporary script file.
"""

import sys, slepc4py

slepc4py.init(sys.argv)

from petsc4py import PETSc
from slepc4py import SLEPc
import numpy
from scicomp_utils_misc import SLEPc_utils

# -- PETSc matrix assembly
opts = PETSc.Options()
n = opts.getInt("n", 150)

A = PETSc.Mat()
A.create()
A.setSizes([n, n])
A.setFromOptions()
A.setUp()

rstart, rend = A.getOwnershipRange()

# first row
if rstart == 0:
    A[0, :2] = [2, -1]
    rstart += 1
# last row
if rend == n:
    A[n - 1, -2:] = [-1, 2]
    rend -= 1
# other rows
for i in range(rstart, rend):
    A[i, i - 1 : i + 2] = [-1, 2, -1]

A.assemble()

# -- Solving with SLEPc
# Build an Eigenvalue Problem Solver object
E = SLEPc.EPS()
E.create(comm=None)
E.setOperators(A)
# HEP = Hermitian
# NHEP = Npn-Hermitian
E.setProblemType(SLEPc.EPS.ProblemType.NHEP)
# set the number of eigenvalues requested
E.setDimensions(nev=5)

history = []


def monitor(eps, its, nconv, eig, err):
    history.append(err[nconv])
    # print(its, rnorm)


E.setMonitor(monitor)
E.setFromOptions()
E.view()  # print all options of EPS object

E.solve()

Print = PETSc.Sys.Print

Print()
Print("******************************")
Print("*** SLEPc Solution Results ***")
Print("******************************")
Print()

its = E.getIterationNumber()
Print("Number of iterations of the method: %d" % its)

eps_type = E.getType()
Print("Solution method: %s" % eps_type)

nev, ncv, mpd = E.getDimensions()
Print("Number of requested eigenvalues: %d" % nev)

tol, maxit = E.getTolerances()
Print("Stopping condition: tol=%.4g, maxit=%d" % (tol, maxit))

nconv = E.getConverged()
Print("Number of converged eigenpairs %d" % nconv)

if nconv > 0:
    # Create the results vectors
    vr, vi = A.createVecs()
    #
    Print()
    Print("        k          ||Ax-kx||/||kx|| ")
    Print("----------------- ------------------")
    for i in range(nconv):
        k = E.getEigenpair(i, vr, vi)
        error = E.computeError(i)
        if k.imag != 0.0:
            Print(" %9f%+9f j %12g" % (k.real, k.imag, error))
        else:
            Print(" %12f       %12g" % (k.real, error))
    Print()

(eigval, eigvec_r, eigvec_i) = SLEPc_utils.EPS_get_spectrum(E)
