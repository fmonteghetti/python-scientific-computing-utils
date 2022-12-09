#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script demonstrates the solution of a generalized eigenvalue problem (GEP)
      (1)      A*u = lambda*B*u, (NxN)
where the matrices A and B have complex coefficients, using a real formulation
      (2)      A_real * [x,y] = mu*B_real * [x,y], (2Nx2N)
where the matrices A_real and B_real have only real coefficients.

Since the spectrum of (2) is twice the length of (1), filtering is necessary.
Two filtering techniques are demonstrated:
        - Based on the residual.
        - Based on an eigenvector criterion (no matrix-vector product)

Computations are done using LAPACK through numpy (dense computation) and
SLEPc.
"""

import numpy as np
import scipy.linalg as lin
import itertools

# Assemble the real counterpart of a complex matrix A
# Ar, Ai (NxN) real and imaginary parts of A
# return [Ar, -Ai] (2Nx2N)
#        [Ai,  Ar]
def assemble_real_GEP(Ar,Ai):
    A = np.vstack((np.hstack((Ar,-Ai)),np.hstack((Ai,Ar))))
    return A

# Print complex spectrum
def print_spectrum(eigval):
    print("eigval_r     eigval_i")
    for i in range(len(eigval)):
        print(f"{np.real(eigval[i]):1.3e}\t{np.imag(eigval[i]):1.3e}")

# Filter spectrum based on the residual r(x) = ||A*x-lambda*B*x||
# The residual is computed using the complex matrix A (NxN)
# Inputs:
#   A, B (NxN) complex matrix
#   eigval (P) eigenvalues of the real GEP
#   eigvec (2NxP) eigenvectors of the real GEP
#   idx_r, idx_i (N) indices of real and imaginary components in x
# Return
#   eigval_f, eigvec_f: filtered eigenvalues and eigenvectors
def filterSpectrum_complexResidual(A,eigval,eigvec,idx_r,idx_i,tol=1e-16,B=None):
    x = eigvec[idx_r,:]
    y = eigvec[idx_i,:]
    eigvec = np.real(x)+1j*np.real(y) # possible complex eigenvectors
    if B is None:
        res = np.linalg.norm(A@eigvec - eigval*eigvec,axis=0)
    else:
        res = np.linalg.norm(A@eigvec - eigval*(B@eigvec),axis=0)
    mask = res<tol
    eigval=eigval[mask]; eigvec=eigvec[:,mask]; res=res[mask]
    print("eigval_r     eigval_i     res")
    for i in range(len(eigval)):
        print(f"{np.real(eigval[i]):1.3e}\t{np.imag(eigval[i]):1.3e}\t{res[i]:1.3e}")
    return (eigval,eigvec)

# Same as 'filterSpectrum_complexResidual', except that the residual is
# computed using real arithmetic.
# Inputs: A,B: real matrices (2Nx2N)
def filterSpectrum_realResidual(A,eigval,eigvec,idx_r,idx_i,tol=1e-16,B=None):
    eigvec = np.real(eigvec)
    x = eigvec[idx_r,:]
    y = eigvec[idx_i,:]
    if B is None:
        res = A@np.vstack((x,y));
        res -= np.real(eigval)*np.vstack((x,y))
        res -= np.imag(eigval)*np.vstack((-y,x))
    else:
        z = B@np.vstack((x,y))
        res = A@np.vstack((x,y))
        res -= np.real(eigval)*z
        res -= np.imag(eigval)*np.vstack((-z[idx_i],z[idx_r]))
    res = np.linalg.norm(res,axis=0)
    mask = res<tol
    eigval=eigval[mask]; eigvec=eigvec[:,mask]; res=res[mask]
    print("eigval_r     eigval_i     res")
    for i in range(len(eigval)):
        print(f"{np.real(eigval[i]):1.3e}\t{np.imag(eigval[i]):1.3e}\t{res[i]:1.3e}")
    return (eigval,eigvec)

# Filter spectrum of real GEP based on an eigenvector criterion.
# Does not require matrix-vector product.
def filterSpectrum_eigenvectors(eigval,eigvec,idx_r,idx_i,tol=1e-16):
    x = eigvec[idx_r,:]; y=eigvec[idx_i,:]
        # Mask for valid complex eigenvectors
        # Based on 'x=i*y' criterion
    mask_cvec=np.max(np.abs(x-1j*y),axis=0)<tol
        # Mask for real eigenvalues
    mask_rval = np.abs(np.imag(eigval))<tol
        # Mask for real eigenvectors
    mask_rvec = np.max(np.abs(np.imag(eigvec)),axis=0)<tol
        # Final mask
        # Limitation: eigenvectors associated with real eigenvalues may be duplicated
    mask = mask_cvec + (mask_rvec*mask_rval)
    eigval_ex = list(itertools.compress(eigval, mask))
    eigvec_ex = np.real(x[:,mask])+1j*np.real(y[:,mask])
    print_spectrum(eigval_ex)
    return (eigval_ex,eigvec_ex)

#%% Generalized eigenvalue problem (LAPACK)
N = 4
A = np.zeros((N,N),dtype=complex)
A[0,0] = 5
A[1:N,1:N] = np.ones((N-1,N-1),dtype=complex)+1j*np.diag(np.ones(N-1))
B = np.zeros((N,N),dtype=complex)
B[0,0] = 10
B[1:N,1:N] = np.random.rand(N-1,N-1)-1j*np.random.rand(N-1,N-1)
    # Assemble real GEP
A_real = assemble_real_GEP(np.real(A), np.imag(A))
B_real = assemble_real_GEP(np.real(B), np.imag(B))
idx_r = np.r_[0:N]; idx_i = np.r_[N:(2*N)]
    # Compute spectra
(eigval,eigvec) = lin.eig(A,B)
(eigval_real,eigvec_real) = lin.eig(A_real,B_real)
print("--- LAPACK eigenvalues (complex formulation)")
print_spectrum(eigval)
print("--- LAPACK eigenvalues (real formulation)")
print_spectrum(eigval_real)
    # Filter using complex arithmetic
print("--- Filtered eigenvalues (residual / complex arithmetic)")
(eigval_ex,eigvec_ex) = filterSpectrum_complexResidual(A, eigval_real, eigvec_real,idx_r,idx_i,tol=1e-10,B=B)
    # Filter using real arithmetic
print("--- Filtered eigenvalues (residual / real arithmetic)")
(eigval_ex,eigvec_ex) = filterSpectrum_realResidual(A_real, eigval_real, eigvec_real,idx_r,idx_i,tol=1e-10,B=B_real)
    # Filter using real arithmetic and no matrix-vector product
print("--- Filtered eigenvalues (eigvec criterion / real arithmetic)")
(eigval_ex,eigvec_ex) = filterSpectrum_eigenvectors(eigval_real,eigvec_real,idx_r,idx_i,tol=1e-14)

#%% Solving the real formulation using SLEPc
from petsc4py import PETSc
from slepc4py import SLEPc
import SLEPc_utils
    # Create sparse eigenvalue problem
A_petsc = PETSc.Mat()
A_petsc.create(PETSc.COMM_WORLD)
A_petsc.setSizes([2*N, 2*N])
A_petsc.setType('aij') # sparse (csr)
A_petsc.setPreallocationNNZ(4*N*N) # estimate of number of non-null element
B_petsc = PETSc.Mat()
B_petsc.create(PETSc.COMM_WORLD)
B_petsc.setSizes([2*N, 2*N])
B_petsc.setType('aij') # sparse (csr)
B_petsc.setPreallocationNNZ(4*N*N) # estimate of number of non-null element
for i in range(2*N):
    for j in range(2*N):
        A_petsc[i,j]=A_real[i,j]
        B_petsc[i,j]=B_real[i,j]
A_petsc.assemblyBegin(); A_petsc.assemblyEnd()
B_petsc.assemblyBegin(); B_petsc.assemblyEnd()
    # Solve
EPS = SLEPc_utils.solve_GEP_shiftinvert(A_petsc,B_petsc,
                          problem_type=SLEPc.EPS.ProblemType.GNHEP,
                          solver=SLEPc.EPS.Type.KRYLOVSCHUR,
                          nev=2,tol=1e-6,max_it=100,
                          target=-1.0,shift=1.0)
    # Filter
(eigval,eigvec)=SLEPc_utils.EPS_get_spectrum_ReImFormulation(EPS,
                                 idx_r.astype(np.int32),idx_i.astype(np.int32))
    # Print
print("--- SLEPc eigenvalues (real formulation)")
print_spectrum(eigval)

