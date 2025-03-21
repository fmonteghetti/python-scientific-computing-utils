#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solve the exterior problem for the Dirichlet Laplacian in dolfinx using
the Dirichlet-to-Neumann map on a circle.

The problem is
    - Delta(u) = f in Omega
    u = uD on Gamma
    du/dn = DtN[u] on B_R (circle of radius R),

where the DtN is defined by a boundary integral

    DtN[u](x) = ∫_{B_R} k(x,y)u(y)ds(y) (x∈B_R),

with kernel defined by a series (Givoli 1992, Numerical Methods for Problem in
Infinite Domains, (49)).
"""

# %%
from mpi4py import MPI

comm = MPI.COMM_WORLD  # dolfinx, petsc, and slepc use mpi communicators
from petsc4py import PETSc
from slepc4py import SLEPc
import pyvista as pv
import numpy as np
import time
import dolfinx
import dolfinx.fem.petsc
import ufl
from scicomp_utils_mesh import gmsh_utils
from scicomp_utils_dolfinx import fenicsx_utils
from scicomp_utils_misc import SLEPc_utils
from scicomp_utils_misc import PETSc_utils
import os

DIR_MESH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mesh")


def DtN_Laplace_circle(n, m):
    """Expression of the DtN kernel for Laplace's equation on a circle.

    The kernel is written as: (Givoli 1992, Numerical Methods for Problem in
    Infinite Domains, (7.49))

        k(x,y) = Σ_n Σ_m ɑ_{n} * k_{n,m}(x) * k_{n,m}(y),

    where ɑ_{n} = - n/(π*R^2),

        k_{n,0}(x) =  cos(n*(θ(x))), and  k_{n,1}(x) =  sin(n*(θ(x))).

    Parameters
    ----------
    n: int
        Order (n>=1).
    m: int
        Decomposition index (m∈{0,1}).

    Returns
    -------
    alpha: float
        Constant scalar factor ɑ_{n,m}.
    k: function x-> float
        Function k_{n,m}.
    """
    alpha = -n / np.pi
    # ufl.atan_2(x[1],x[0]) not supported with complex PETSc scalar
    theta = lambda x: np.arctan2(x[1], x[0])
    r = lambda x: np.sqrt(x[0] ** 2 + x[1] ** 2)
    if n <= 0:
        raise ValueError("Order n must be nonnegative.")
    if m == 0:
        k = lambda x: np.cos(n * theta(x)) / r(x)
    elif m == 1:
        k = lambda x: np.sin(n * theta(x)) / r(x)
    else:
        raise ValueError("Index m must be in {0,1}.")
    return (alpha, k)


degree_mesh = 2
degree_fem = 2
dim = 2
R0, R1 = 0.5, 1
lc = (R1 - R0) / 30
geofile = os.path.join(DIR_MESH, "Annulus-2D.geo")
# Physical names
Gamma_name = ["Gamma-0"]
DtN_name = ["Gamma-1"]
Omega_name = ["Omega"]
DtN_order = 5  # n>=1 No DtN if 0
r_expr = lambda x: np.sqrt(x[0] ** 2 + x[1] ** 2)
theta_expr = lambda x: np.arctan2(x[1], x[0])
uD_n = 5
uD_expr = lambda x: np.sin(uD_n * theta_expr(x))  # Dirichlet data
uEx_expr = lambda x: ((R0 / r_expr(x)) ** uD_n) * uD_expr(x)
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
# == Weak formulation a(u,v) + a_DtN(u,v) = l(u,v) with v in H^1
# Build distributed function space
t0 = time.process_time_ns()
V = dolfinx.fem.functionspace(mesh, ("CG", degree_fem))
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
x = ufl.SpatialCoordinate(dmesh.mesh)
uD = dolfinx.fem.Function(V)
uD.interpolate(uD_expr)
# Update all ghost values from local values
uD.x.scatter_forward()
bcs = fenicsx_utils.create_DirichletBC(
    dmesh.mesh, dmesh.facet_tags, V, [uD], Gamma_tags
)
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dmesh.dx
f = dolfinx.fem.Function(V)
f.interpolate(f_expr)
l = ufl.inner(f, v) * dmesh.dx
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
# Assemble DtN
if DtN_order > 0:
    k_fun = dolfinx.fem.Function(V)
    A_tmp = None
    for n in np.arange(1, DtN_order + 1):
        for m in [0, 1]:
            (alpha, k_expr) = DtN_Laplace_circle(n, m)
            k_fun.interpolate(k_expr)
            A_tmp = fenicsx_utils.assemble_matrix_double_facet_integral(
                k_fun, Gamma_DtN_tags, dmesh, result=A_tmp
            )
            A.axpy(-alpha, A_tmp)

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
    mesh.comm, "shared/demo_laplace_dtn.xdmf", "w"
) as file:
    file.write_mesh(mesh)
    file.write_function(uh)
