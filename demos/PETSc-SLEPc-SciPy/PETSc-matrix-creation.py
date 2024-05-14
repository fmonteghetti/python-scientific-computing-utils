#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Creation of PETSc matrix in parallel.

The script can be run on one or multiple threads:
    python ./MPI-test.py (one thread)
    mpirun -n N ./MPI-test.py (N threads).
The latter can be executed from the ipython prompt using:
    ! mpirun -n N ./MPI-test.py.

"""
#%%
from mpi4py import MPI
comm = MPI.COMM_WORLD
from petsc4py import PETSc
import numpy as np

N, M = 5, 5

# Create a distributed matrix
A = PETSc.Mat()
A.create(comm)
# A is an (N,M) matrix
# Each MPI process stores a (nxM) submatrix,
# where n is determined by PETSc
A.setSizes(((PETSc.DECIDE,N),(PETSc.DECIDE,M)))
A.setFromOptions()
    # Pre-allocate a number of non-zero element per row.
    # When n is PETSc.DECIDE, nz must be a scalar
    # Otherwise, nnz can be an integer vector of length n.
nnz = 2
A.setPreallocationNNZ(nnz)
    # Rows owned by MPI process
    # When n is PETSc.DECIDE, this must be called after setPreallocationNNZ()
rows = A.getOwnershipRange()
    # Set Values using global numering
A.setValues(rows[0],0,comm.rank,addv=False)
A.setValues(rows[0],1,comm.rank,addv=False)
A.assemblyBegin()
A.assemblyEnd()
print(f"[{comm.rank}] A local and global sizes: {A.getSizes()[0]}")
print(f"[{comm.rank}] Rows owned: {A.getOwnershipRange()}")
info = A.getInfo()
print(f"[{comm.rank}] nz_allocated/nz_unneeded: "+ \
        f"{info['nz_allocated']}/{info['nz_unneeded']}, "+\
        f"mallocs: {info['mallocs']}")
comm.Barrier()
A.view()

# Create a distributed vector
N = 5
L = PETSc.Vec()
L.create(comm)
L.setSizes((PETSc.DECIDE,N))
L.setFromOptions()
rows=L.getOwnershipRange()
rows_idx = np.arange(rows[0],rows[1],dtype=PETSc.IntType)
L.setValues(rows_idx,rows_idx)
L.assemblyBegin()
L.assemblyEnd()
L.view()