#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test PETSc_utils.kron_vector().

The script can be run on one or multiple threads:
    python ./MPI-test.py (one thread)
    mpirun -n N ./MPI-test.py (N threads).
The latter can be executed from the ipython prompt using:
    ! mpirun -n N ./MPI-test.py.

"""
#%%
import time
from mpi4py import MPI
comm = MPI.COMM_WORLD
from petsc4py import PETSc
import numpy as np
from scientific_computing_utils import PETSc_utils

def _create_vector(N):
    """ Create distributed vector. """
    L = PETSc.Vec()
    L.create(comm)
    L.setSizes((PETSc.DECIDE,N))
    L.setFromOptions()
    rows=L.getOwnershipRange()
    rows_idx = np.arange(rows[0],rows[1],dtype=PETSc.IntType)
    L.setValues(rows_idx,rows_idx)
    L.assemblyBegin()
    L.assemblyEnd()
    return L

def _compute_exact_solution(N):
    """ Exact kronecker product L*L^T. """
    A_ex = np.kron(np.arange(0,N),np.arange(0,N)).reshape((N,-1))
    return A_ex

def _compute_error(A,A_ex):
    """ Compute ||A-Aex|| in the infinity norm."""
    I = A.getOwnershipRange()
    error = 0
    for i in np.arange(I[0],I[1]):
        for j in np.arange(0,N):
            error = np.max([error,np.abs(A[i,j]-A_ex[i,j])])
    return error

N = 1234 
L=_create_vector(N)
t0=time.process_time_ns()
A = PETSc_utils.kron_vector(L)
comm.Barrier()
if comm.rank ==0:
    print(f"Elapsed time: {1e-9*(time.process_time_ns()-t0):1.2g}s")
A_ex = _compute_exact_solution(N)
error = _compute_error(A,A_ex)
print(f"[{comm.rank}] Error={error}")

# Test reuse of matrix A
A.zeroEntries()
t0=time.process_time_ns()
PETSc_utils.kron_vector(L,result=A)
comm.Barrier()
if comm.rank ==0:
    print(f"Elapsed time (re-use): {1e-9*(time.process_time_ns()-t0):1.2g}s")
error = _compute_error(A,A_ex)
print(f"[{comm.rank}] Error={error}")