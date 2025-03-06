"""
Script that demonstrates solving a quadratic eigenvalue problem using slepc.

Adapted from ex5.py available in upstream repo
"""
from mpi4py import MPI
comm = MPI.COMM_WORLD # dolfinx, petsc, and slepc use mpi communicators

from petsc4py import PETSc
from slepc4py import SLEPc
from scicomp_utils_misc import SLEPc_utils

def construct_operators(m,n):
    """
    Standard symmetric eigenproblem corresponding to the
    Laplacian operator in 2 dimensions.
    """
    # K is the 2-D Laplacian
    K = PETSc.Mat().create()
    K.setSizes([n*m, n*m])
    K.setFromOptions()
    Istart, Iend = K.getOwnershipRange()
    for I in range(Istart,Iend):
        v = -1.0; i = I//n; j = I-i*n;
        if i>0:
            J=I-n; K[I,J] = v
        if i<m-1:
            J=I+n; K[I,J] = v
        if j>0:
            J=I-1; K[I,J] = v
        if j<n-1:
            J=I+1; K[I,J] = v
        v=4.0; K[I,I] = v
    K.assemble()
    # C is the zero matrix
    C = PETSc.Mat().create()
    C.setSizes([n*m, n*m])
    C.setFromOptions()
    C.assemble()
    # M is the identity matrix
    M = PETSc.Mat().createConstantDiagonal([n*m, n*m], 1.0)
    return M, C, K

(M, C, K) = construct_operators(10,10)
PEP = SLEPc_utils.solve_PEP_shiftinvert([K,C,M],comm=comm,
                      nev=5,tol=1e-4,max_it=10,
                      target=10,shift=10)
(eigval,eigvec_r,eigvec_i) = SLEPc_utils.EPS_get_spectrum(PEP)




