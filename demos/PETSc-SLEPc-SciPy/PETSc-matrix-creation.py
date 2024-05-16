#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Elementary creation and manipulation of PETSc vectors and matrix in parallel.

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
    # When running in parallel, diagonal and off-process blocks are
    # pre-allocated separately.
d_nnz = 2 # nnz element per row of diagonal (n,m) submatrix
o_nnz = 1 # nnz element per row of off-process (n,M-m) submatrix
A.setPreallocationNNZ((d_nnz,o_nnz))
    # Rows owned by MPI process
    # When n is PETSc.DECIDE, this must be called after setPreallocationNNZ()
rows = A.getOwnershipRange()
    # Set Values using global numering
A.setValues(rows[0],rows[0],comm.rank,addv=False) 
A.assemblyBegin()
A.assemblyEnd()
print(f"[{comm.rank}] A local and global sizes: {A.getSizes()[0]}")
print(f"[{comm.rank}] Rows owned: {A.getOwnershipRange()}")
info = A.getInfo()
print(f"[{comm.rank}] Memory allocation:\n\tnz_allocated/nz_unneeded: "+ \
        f"{info['nz_allocated']}/{info['nz_unneeded']},\n\t"+\
        f"additional (costly) mallocs: {info['mallocs']}")
comm.Barrier()
A.view()

def print_vector(v,has_ghost_values=False):
    """ Print some basic info about a petsc4py.PETSc.Vec """
    (n,N) = L.getSizes()
    rows = L.getOwnershipRange()
    print(f"[{comm.rank}] I own {n} values in [{rows[0]},{rows[1]})")
    print(f"[{comm.rank}] Owned values: {L.getArray()}")
    if has_ghost_values:
        with L.localForm() as L_loc:
            # L_loc = 'local ghosted representation' of L
            n_ghost = L_loc.size-n
            print(f"[{comm.rank}] I have {n_ghost} ghost values: {L_loc[n:-1]}") 

# Create a distributed vector without ghost values
N = 5
L = PETSc.Vec()
L.create(comm)
# L is a vector of length N.
# Each MPI process stores a vector of size n,
# where n is determined by PETSc
L.setSizes((PETSc.DECIDE,N))
L.setFromOptions()
rows=L.getOwnershipRange()
rows_idx = np.arange(rows[0],rows[1],dtype=PETSc.IntType)
L.setValues(rows_idx,rows_idx)
L.assemblyBegin()
L.assemblyEnd()
L.view()
print_vector(L)

# Create a distributed vector with ghost values
N = 6
ghost_idx = [1,2] # global index of ghost values
L = PETSc.Vec()
L.createGhost(ghost_idx,(PETSc.DECIDE,N))
# L is a vector of length N.
# Each MPI process stores a vector of size n, where n is determined by PETSc,
# as well as a copy of the ghost values.
L.setFromOptions()
L.assemblyBegin()
L.assemblyEnd()

# The sequence below illustrates ghost values update and synchronization
print_vector(L)
print(f"[{comm.rank}] I am modifying my copy of the first ghost value.")
with L.localForm() as L_loc:
    (n,N) = L.getSizes()
    L_loc[n] = comm.rank+1 # Updating first ghost value
print_vector(L)
comm.Barrier()
if comm.rank==0:
    print(f"Updating all owned values.")
    # update the values owned by each process using all copies of ghost values
L.ghostUpdate(PETSc.InsertMode.ADD_VALUES, PETSc.ScatterMode.REVERSE)
comm.Barrier()
print_vector(L)
comm.Barrier()
if comm.rank==0:
    print(f"Updating all copies of ghost values.")
    # update all copies of ghost values
L.ghostUpdate(PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.FORWARD)
comm.Barrier()
print_vector(L)