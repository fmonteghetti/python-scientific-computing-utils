#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1D finite element for Laplacian with piecewise coefficient.

    -div(kappa*grad(u)) = lambda * u on (0,L),

with boundary conditions

    u(0)=0 and du/dn(L)=0,

and piecewise coefficient:

    kappa(x) = kappa_0 on (0,L_int), kappa_1 on (L_int,L)

"""

# %%
import numpy as np
import ufl
import basix
import dolfinx
import dolfinx.fem.petsc
from scicomp_utils_dolfinx import fenicsx_utils
from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc
from scicomp_utils_misc import SLEPc_utils
import matplotlib as mpl
import matplotlib.pyplot as plt

L = 2
L_int = 0.5
degree = 1
N = 20  # number of cell
kappa_0 = 10
kappa_1 = 1

# Mesh and function space
mesh = dolfinx.mesh.create_interval(MPI.COMM_WORLD, N, [0, L])
CG = basix.ufl.element("CG", mesh.ufl_cell().cellname(), degree=degree)
V = dolfinx.fem.functionspace(mesh, CG)


# Dirichlet boundary conditions at x=0
def Dirichlet_bnd(x):
    return np.isclose(x[0], 0.0)


uD = dolfinx.fem.Function(V)
uD.x.petsc_vec.set(0)

facets = dolfinx.mesh.locate_entities_boundary(mesh, 0, Dirichlet_bnd)
dof = dolfinx.fem.locate_dofs_topological(V, 0, facets)
bc0 = dolfinx.fem.dirichletbc(uD, dof)
bcs = [bc0]
# Discontinuous coefficient using formula
# Preferred way would be to split integration
kappa = dolfinx.fem.Function(V)


def Omega_0(x):
    return x[:, 0] <= L_int


def Omega_1(x):
    return x[:, 0] >= L_int


x = V.tabulate_dof_coordinates()
kappa.x.petsc_vec.setArray(kappa_0 * Omega_0(x) + kappa_1 * Omega_1(x))
kappa.x.petsc_vec.ghostUpdate(
    addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
)
# Measure on whole domain
dx = ufl.dx
# Weak formulation
u, phi = ufl.TrialFunction(V), ufl.TestFunction(V)
a = ufl.inner(kappa * ufl.grad(u), ufl.grad(phi)) * dx
b = ufl.inner(u, phi) * dx
# Assembly
diag_A = 1e2
diag_B = 1e-2
(GEP_A, GEP_B) = fenicsx_utils.assemble_GEP(
    a, b, bcs, diag_A=diag_A, diag_B=diag_B
)
# Solve
OptDB = PETSc.Options()
OptDB["st_ksp_type"] = "preonly"
OptDB["st_pc_type"] = "lu"
OptDB["st_pc_factor_mat_solver_type"] = "mumps"
EPS = SLEPc_utils.solve_GEP_shiftinvert(
    GEP_A,
    GEP_B,
    problem_type=SLEPc.EPS.ProblemType.GHEP,
    solver=SLEPc.EPS.Type.KRYLOVSCHUR,
    nev=50,
    tol=1e-8,
    max_it=1000,
    target=4,
    shift=4,
)
# Plot
i_plot = 2
(eigval, ur, ui) = SLEPc_utils.EPS_get_spectrum(EPS)
fig = plt.figure()
ax = fig.add_subplot(2, 1, 1)
ax.plot(np.real(eigval), np.imag(eigval), linestyle="none", marker="o")
ax.set_xlabel(f"$\Re(\lambda)$")
ax.set_ylabel(f"$\Im(\lambda)$")
ax.set_title(f"Eigenvalues")
ax.set_xlim([0, 1e3])
ax = fig.add_subplot(2, 1, 2)
ax.plot(V.tabulate_dof_coordinates()[:, 0], ur[i_plot].array)
ax.set_xlabel("x")
ax.set_title(f"Mode no. {i_plot}")
