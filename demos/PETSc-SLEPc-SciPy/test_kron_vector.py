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

def _get_integer_test(N):
    """ Create a distributed integer vector L and the matrix L*L^T."""
    L = PETSc.Vec()
    L.create(comm)
    L.setSizes((PETSc.DECIDE,N))
    L.setFromOptions()
    rows=L.getOwnershipRange()
    rows_idx = np.arange(rows[0],rows[1],dtype=PETSc.IntType)
    L.setValues(rows_idx,rows_idx)
    L.assemblyBegin()
    L.assemblyEnd()

    A_ex = np.kron(np.arange(0,N),np.arange(0,N)).reshape((N,-1))
    return (L,A_ex)

def _get_real_test(N):
    """ Create a distributed real vector L and the matrix L*L^T."""
    x = np.linspace(-10,10,num=N)
    f = lambda x: np.exp(x) * np.sin(2*x) + 1.23 
    x = f(x)
    A_ex = np.kron(x,x).reshape(N,-1)
    L = PETSc.Vec()
    L.create(comm)
    L.setSizes((PETSc.DECIDE,N))
    L.setFromOptions()
    rows=L.getOwnershipRange()
    rows_idx = np.arange(rows[0],rows[1],dtype=PETSc.IntType)
    L.setValues(rows_idx,x[rows_idx])
    L.assemblyBegin()
    L.assemblyEnd()
    return (L,A_ex)

def _get_complex_test(N):
    """ Create a distributed complex vector L and the matrix L*L^T."""
    x = np.linspace(-10,10,num=N)
    f = lambda x: np.exp(x) * np.sin(2*x) + 1.23 
    x = f(x + 1j)
    A_ex = np.kron(x,x).reshape(N,-1)

    L = PETSc.Vec()
    L.create(comm)
    L.setSizes((PETSc.DECIDE,N))
    L.setFromOptions()
    rows=L.getOwnershipRange()
    rows_idx = np.arange(rows[0],rows[1],dtype=PETSc.IntType)
    L.setValues(rows_idx,x[rows_idx])
    L.assemblyBegin()
    L.assemblyEnd()
    return (L,A_ex)

def _compute_error(A,A_ex):
    """ Compute ||A-Aex|| in the infinity norm."""
    I = A.getOwnershipRange()
    error = 0
    for i in np.arange(I[0],I[1]):
        for j in np.arange(0,A_ex.shape[1]):
            error = np.max([error,np.abs(A[i,j]-A_ex[i,j])])
    return error

N = 239
test_cases = [_get_integer_test(N+27),
              _get_real_test(N+28)]
if PETSc.ScalarType==np.complex128:
    test_cases.append(_get_complex_test(N+33))

for (L,A_ex) in test_cases:
    t0=time.process_time_ns()
    A = PETSc_utils.kron_vector(L)
    comm.Barrier()
    if comm.rank ==0:
        print(f"Elapsed time: {1e-9*(time.process_time_ns()-t0):1.2g}s")
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
# Print memory allocation info
info = A.getInfo()
if comm.rank==0:
    print(f"[{comm.rank}] nz_allocated/nz_unneeded/mallocs: {info['nz_allocated']}/{info['nz_unneeded']}/{info['mallocs']}.")