# -*- coding: utf-8 -*-
"""
The purpose of this tutorial script is to showcase the manipulation of matrices
under three different formats:
    - numpy.ndarray. These are dense matrices suited for small problems.
Numpy and scipy provides direct solvers and eigenvalues solvers from LAPACK.

    - scipy.sparse. Several sparse storage formats are supported. The advice is
to use csr/csc for arithmetic and matrix vector products. Building under coo
format is fast and covenient. scipy.sparse.linalg provides iterative solvers
as well as a hook to ARPACK.

    - PETSc.Mat (and PETSc.Vec). Sparse matrix format used by the PETSc library.
By default the format is CSR. This format enables to use a wide library,
notably SLEPc for sparse eigenvalue problems (goes beyond ARPACK).

"""

import numpy as np
from numpy import pi

import scipy.sparse as sp


from petsc4py import PETSc
from slepc4py import SLEPc

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import Poisson_FDM_1D as PoissonFDM
from scicomp_utils_misc import PETSc_utils

import matplotlib.pyplot as plt

# %% Common inputs
N = 50  # number of nodes
L = 1  # domain length
g = lambda x: 2 * x + 1  # boundary condition
# source term and exact solution
n = 20
f = lambda x: np.sin((n * pi / L) * x)
u_ex = lambda x: ((L / (pi * n)) ** 2) * f(x)
# source term and exact solution
f = lambda x: -(x**2 + 3 * x) * np.exp(x)
u_ex = lambda x: (x**2 - x) * np.exp(x)
# exact eigenvalues
lambda_ex = ((pi / L) ** 2) * (np.r_[1:10] ** 2)
# %% Direct problem with Numpy (dense)
(A, b) = PoissonFDM.assemble_numpy(L, N, f, g)
u = np.linalg.solve(A, b)
# %% Direct problem with Scipy (sparse)
(A, b) = PoissonFDM.assemble_scipy(L, N, f, g)
u = PoissonFDM.solve_scipy(A, b, solver="umfpack")
# plt.spy(A) # sparsity pattern
# %% Direct problem with PETSc (sparse)
(A, b) = PoissonFDM.assemble_PETSc(L, N, f, g)
u = PoissonFDM.solve_PETSc(A, b, solver="gmres")
# %% Direct problem: assemble in scipy and solve with PETSc
(A_sp, b_sp) = PoissonFDM.assemble_scipy(L, N, f, g)
# convert
A = PETSc_utils.convert_scipy_to_PETSc(A_sp, PETSc.COMM_WORLD)
b = PETSc.Vec().createWithArray(b_sp)
# solve
u = PoissonFDM.solve_PETSc(A, b, solver="gmres")
# %% Direct problem: assemble with PETSc and solve with scipy
(A_p, b_p) = PoissonFDM.assemble_PETSc(L, N, f, g)
# convert
A = PETSc_utils.convert_PETSc_to_scipy(A_p)
b = b_p.array
# solve
u = PoissonFDM.solve_scipy(A, b, solver="umfpack")
# %% Plot solution to direct problem
# add boundary points
u = PoissonFDM.add_boundary_nodes(u, g, L, N)
(h, x_FDM) = PoissonFDM.get_FDM_nodes(L, N)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
# ax.plot(x_FDM,u_ex(x_FDM),label='Exact')
ax.plot(x_FDM, u, marker="x", linestyle="none", label="FDM")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
# %% Eigenvalue problem: numpy (dense)
(A, b) = PoissonFDM.assemble_numpy(L, N, f, g)
# QR iteration from LAPACK (similar to scipy.linalg.eig)
(eigval, eigvec) = np.linalg.eig(A)
# sort
idx = eigval.argsort()
eigval = eigval[idx]
eigvec = eigvec[:, idx]
# %% Sparse Eigenvalue problem: Scipy
(A, b) = PoissonFDM.assemble_scipy(L, N, f, g)
# ARPACK
(eigval, eigvec) = sp.linalg.eigs(A, k=10, return_eigenvectors=True, which="SI")
# %% Eigenvalue problem: SLEPc
# Asking for too little eigenvectors can hinder convergence
(A, b) = PoissonFDM.assemble_PETSc(L, N, f, g)
E = PoissonFDM.eigensolve_SLEPc(
    A, SLEPc.EPS.ProblemType.NHEP, comm=PETSc.COMM_WORLD, nev=10
)
# E.view() # print all options of EPS object
PoissonFDM.eigensolve_SLEPc_post_process(A, E)
(eigval, eigvec) = PoissonFDM.get_spectrum(A, E)
print(f"Number of converged eigenvalues: {eigval.shape}")
# %% Plot eigenvectors
(h, x_FDM) = PoissonFDM.get_FDM_nodes(L, N)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
for j in range(np.min([eigvec.shape[1], 5])):
    tmp = np.real(eigvec[:, j])
    tmp = tmp / np.linalg.norm(tmp)
    ax.plot(x_FDM[1:-1], tmp, label=f"vec {j}")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
# exact eigenvectors for comparison
tmp = -np.sin(x_FDM * 4 * pi / L)
tmp = tmp / np.linalg.norm(tmp)
ax.plot(x_FDM, tmp, "--")
