#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solve a tridiagonal linear system using PETSc.
TODO:
    - Get it to work in serial
    - Parallelization
    - Possible to avoid loop? (Cython?)
"""



#%%
from petsc4py import PETSc

# grid size and spacing
m, n  = 32, 32
hx = 1.0/(m-1)
hy = 1.0/(n-1)

# create sparse matrix
A = PETSc.Mat()
A.create(PETSc.COMM_WORLD)
A.setSizes([m*n, m*n])
A.setType('aij') # sparse
A.setPreallocationNNZ(5) # estimate of number of non-null element

# precompute values for setting
# diagonal and non-diagonal entries
diagv = 2.0/hx**2 + 2.0/hy**2
offdx = -1.0/hx**2
offdy = -1.0/hy**2

# loop over owned block of rows on this
# processor and insert entry values
Istart, Iend = A.getOwnershipRange()
for I in range(Istart, Iend) :
    A[I,I] = diagv
    i = I//n    # map row number to
    j = I - i*n # grid coordinates
    if i> 0  : J = I-n; A[I,J] = offdx
    if i< m-1: J = I+n; A[I,J] = offdx
    if j> 0  : J = I-1; A[I,J] = offdy
    if j< n-1: J = I+1; A[I,J] = offdy

# communicate off-processor values
# and setup internal data structures
# for performing parallel operations
A.assemblyBegin()
A.assemblyEnd()


#%% Solve with Krylov subspace method (KSP)
# create linear solver
ksp = PETSc.KSP()
ksp.create(PETSc.COMM_WORLD)
# use conjugate gradients
ksp.setType('cg')
# and incomplete Cholesky
ksp.getPC().setType('icc')
# obtain sol & rhs vectors
x, b = A.createVecs()
x.set(0)
b.set(1)
# and next solve
ksp.setOperators(A)
ksp.setFromOptions()
ksp.solve(b, x)

# TODO: check the solution
b_res = A.createVecs()[0]
A.mult(x, b_res)

    # b_res - b
b_res.axpy(-1,b)
    # maximum of b_res
print(f"||Ax-B||_inf: {b_res.norm(norm_type=PETSc.NormType.NORM_INFINITY)}")
    # L2 norm of b_res
print(f"||Ax-b||_2:  {b_res.norm(norm_type=PETSc.NormType.NORM_2)}")   