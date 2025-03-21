#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solve the exterior problem for the Helmholtz equation in dolfinx using
the Dirichlet-to-Neumann map on a circle.

The problem is: find u such that

        Δu + k^2 * u = 0                            (on Ω)
        u = 0                                       (on Γ)
        (d/dr - T)u = (d/dr - T)u_i                 (on B_R)

where
    u_i is the known incident field,
    u is the total field,
    T is the Dirichlet-to-Neumann (DtN) operator, defined by a boundary integral
    on a circle of radius R:

        DtN[u](x) = ∫_{B_R} k(x,y)u(y)ds(y) (x∈B_R),

    with kernel defined by a series (Givoli 1992, Numerical Methods for Problem
    in Infinite Domains, (10.5)).
"""

# %%
from mpi4py import MPI

comm = MPI.COMM_WORLD  # dolfinx, petsc, and slepc use mpi communicators
from petsc4py import PETSc
import numpy as np
from scipy.special import h1vp, h2vp
import time
import dolfinx
import dolfinx.fem.petsc
import ufl
from scicomp_utils_mesh import gmsh_utils
from scicomp_utils_dolfinx import fenicsx_utils
import os

DIR_MESH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mesh")


def DtN_Helmholtz_circle(n, m, R, k):
    """Expression of the DtN kernel for Helmholtz's equation on a circle.

    The kernel is written as: (Givoli 1992, Numerical Methods for Problem in
    Infinite Domains, (10.5))

        t(x,y) = Σ_n Σ_m ɑ_{n} * t_{n,m}(x) * t_{n,m}(y).

    Parameters
    ----------
    n: int
        Order (n>=0).
    m: int
        Decomposition index (m∈{0,1}).
    R: float
        Radius of circle.
    k: float
        Wavenumber.

    Returns
    -------
    alpha: float
        Constant scalar factor ɑ_{n,m}.
    t: function x-> float
        Function t_{n,m}.
    """
    if n < 0:
        raise ValueError("Order n must be positive.")
        # ufl.atan_2(x[1],x[0]) not supported with complex PETSc scalar
    theta = lambda x: np.arctan2(x[1], x[0])
    r = lambda x: np.sqrt(x[0] ** 2 + x[1] ** 2)
    # Scalar coefficient
    alpha = (k / np.pi) * (h1vp(n, k * R, n=1)) / (R * h1vp(n, k * R, n=0))
    if n == 0:
        alpha = alpha / 2
        # Kernel
    if m == 0:
        t = lambda x: np.cos(n * theta(x))
    elif m == 1:
        t = lambda x: np.sin(n * theta(x))
    else:
        raise ValueError("Index m must be in {0,1}.")
    return (alpha, t)


degree_mesh = 2
degree_fem = 2
dim = 2
R0, R1 = 0.5, 1
lc = (R1 - R0) / 60
geofile = os.path.join(DIR_MESH, "Annulus-2D.geo")
# Physical names
Gamma_name = ["Gamma-0"]
DtN_name = ["Gamma-1"]
Omega_name = ["Omega"]
DtN_order = 5  # n>=0, Sommerfeld radiation condition if -1
ui_n = 5  # Orthoradial order of incident field (n>=0)
k = 4  # wavenumber (k>0)
# Boundary data and analytical solution
# Incident field
r_expr = lambda x: np.sqrt(x[0] ** 2 + x[1] ** 2)
theta_expr = lambda x: np.arctan2(x[1], x[0])
ui_expr = lambda x: (
    h2vp(ui_n, k * r_expr(x), n=0) / h2vp(ui_n, k * R0, n=0)
) * np.sin(ui_n * theta_expr(x))
# Exact scattered field
us_expr = lambda x: (
    -h1vp(ui_n, k * r_expr(x), n=0) / h1vp(ui_n, k * R0, n=0)
) * np.sin(ui_n * theta_expr(x))
# Exact total field
uEx_expr = lambda x: us_expr(x) + ui_expr(x)
xc, sigma, A = [0.0, 0.0], 1.0, 0.0
f_expr = lambda x: A * np.exp(
    -((x[0] - xc[0]) ** 2 + (x[1] - xc[1]) ** 2) / sigma
)
# == Create and read mesh
gmshfile = geofile.replace(".geo", ".msh")
if comm.rank == 0:  # generate mesh on first thread only
    gmsh_utils.generate_mesh_cli(
        geofile,
        gmshfile,
        dim,
        refinement=0,
        binary=True,
        parameters={"R0": R0, "R1": R1, "lc": lc},
        order=degree_mesh,
    )
comm.Barrier()
# Read .msh file and build a distributed mesh
dmesh = fenicsx_utils.DolfinxMesh.init_from_gmsh(gmshfile, dim, comm=comm)
dim, mesh, name_2_tags = dmesh.dim, dmesh.mesh, dmesh.name_2_tags
Gamma_tags = [t for name in Gamma_name for t in name_2_tags[-2][name]]
Gamma_DtN_tags = [t for name in DtN_name for t in name_2_tags[1][name]]
# == Weak formulation
# Build distributed function space
t0 = time.process_time_ns()
x = ufl.SpatialCoordinate(dmesh.mesh)
nvec = ufl.FacetNormal(dmesh.mesh)
V = dolfinx.fem.functionspace(mesh, ("CG", degree_fem))
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
uD = dolfinx.fem.Function(V)
bcs = fenicsx_utils.create_DirichletBC(
    dmesh.mesh, dmesh.facet_tags, V, [uD], Gamma_tags
)
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dmesh.dx
a += ufl.inner(-(k**2) * u, v) * dmesh.dx
f = dolfinx.fem.Function(V)
f.interpolate(f_expr)
l = ufl.inner(f, v) * dmesh.dx
ui = dolfinx.fem.Function(V)
ui.interpolate(ui_expr)
ui_dn = ufl.dot(nvec, ufl.grad(ui))
for tag in Gamma_DtN_tags:
    l += ufl.inner(ui_dn, v) * dmesh.ds(tag)
if DtN_order == -1:  # Sommerfeld condition
    for tag in Gamma_DtN_tags:
        a += ufl.inner(-1j * k * u, v) * dmesh.ds(tag)
        l += ufl.inner(-1j * k * ui, v) * dmesh.ds(tag)
a = dolfinx.fem.form(a)
A = dolfinx.fem.petsc.assemble_matrix(a, bcs)
A.assemble()
l = dolfinx.fem.form(l)
L = dolfinx.fem.petsc.assemble_vector(l)
L.assemble()
# Apply Dirichlet b.c. to rhs
dolfinx.fem.petsc.apply_lifting(L, [a], [bcs])
L.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
dolfinx.fem.petsc.set_bc(L, bcs)
# Assemble DtN contribution to A and L
if DtN_order >= 0:
    k_fun = dolfinx.fem.Function(V)
    A_tmp = None
    for n in np.arange(0, DtN_order + 1):
        for m in [0, 1]:
            (alpha, k_expr) = DtN_Helmholtz_circle(n, m, R1, k)
            k_fun.interpolate(k_expr)
            # Contribution to A
            A_tmp = fenicsx_utils.assemble_matrix_double_facet_integral(
                k_fun, Gamma_DtN_tags, dmesh, result=A_tmp
            )
            A.axpy(-alpha, A_tmp)
            # Contribution to L
            for tag in Gamma_DtN_tags:
                # DtN term
                form = ufl.inner(k_fun, v) * dmesh.ds(tag)
                Ltmp = dolfinx.fem.petsc.assemble_vector(dolfinx.fem.form(form))
                Ltmp.assemble()
                Ltmp.ghostUpdate(
                    addv=PETSc.InsertMode.ADD_VALUES,
                    mode=PETSc.ScatterMode.REVERSE,
                )
                form = k_fun * ui * dmesh.ds(tag)
                beta = dolfinx.fem.assemble_scalar(dolfinx.fem.form(form))
                beta = comm.allreduce(beta, op=MPI.SUM)
                L.axpy(-alpha * beta, Ltmp)
    # Solve
solver = PETSc.KSP().create(comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)
uh = dolfinx.fem.Function(V)
solver.solve(L, uh.x.petsc_vec)
uh.x.scatter_forward()

comm.Barrier()
if comm.rank == 0:
    print(f"Elapsed time: {1e-9 * (time.process_time_ns() - t0):1.2g}s")

# Error
uEx = dolfinx.fem.Function(V)
uEx.interpolate(uEx_expr)
L2_error = ufl.inner(uh - uEx, uh - uEx) * dmesh.dx
L2_error = dolfinx.fem.form(L2_error)
L2_error = dolfinx.fem.assemble_scalar(L2_error)
L2_error = np.sqrt(comm.allreduce(L2_error, op=MPI.SUM))

if comm.rank == 0:
    print(f"DoF : {uh.x.petsc_vec.size}")
    print(f"Error_L2 : {L2_error:.2e}")
    # Export uh to a single XDMF file (currently supports only one function)
with dolfinx.io.XDMFFile(
    mesh.comm, "shared/demo_helmholtz_dtn.xdmf", "w"
) as file:
    file.write_mesh(mesh)
    file.write_function(uh)
