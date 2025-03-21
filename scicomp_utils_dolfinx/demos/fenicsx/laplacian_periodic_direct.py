# This script is adapted from 'demo_periodic.py' provided with dolfinx_mpc.
# It solves Poisson equation and the associated eigenvalue problem.
# List of modifications:
#   - Mesh loaded from gmsh file
#   - Dirichlet boundary condition defined using gmsh tag
#   - MultiPointConstraint defined using gmsh tag
#   - The generalized eigenvalue problem is solved using PETSc
#   - Assenbly function uses diagval parameter to control location of
# spurious eigenvalue. (dolfinx_mpc source code has to be modified
# accordingly.)

import dolfinx
import dolfinx.io
import dolfinx_mpc
from scicomp_utils_dolfinx import dolfinx_mpc_utils

import ufl
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

# Eigenvalue problem
from slepc4py import SLEPc
from scicomp_utils_misc import SLEPc_utils
from scicomp_utils_dolfinx import fenicsx_utils

# Mesh
from scicomp_utils_mesh import gmsh_utils
from scicomp_utils_mesh import meshio_utils

# Plot
import matplotlib.pyplot as plt
import pyvista as pv
from scicomp_utils_dolfinx import pyvista_utils
import os

DIR_MESH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mesh")
# %% Load mesh
geofile = os.path.join(DIR_MESH, "Rectangle.geo")
gmshfile = os.path.join(DIR_MESH, "Rectangle.msh")
x_left = 0.0
x_right = 1.5
gmsh_utils.generate_mesh_cli(
    geofile,
    gmshfile,
    2,
    refinement=0,
    binary=True,
    parameters={"x_l": x_left, "x_r": x_right, "N_x": 1000, "N_y": 100},
)
gmsh_utils.print_summary(gmshfile)
mesh_cells_type = meshio_utils.get_cells_type_str(gmshfile)
dmesh = fenicsx_utils.DolfinxMesh.init_from_gmsh(gmshfile, 2)
phys_tag_1D = gmsh_utils.getPhysicalNames(gmshfile, 1)
phys_tag_2D = gmsh_utils.getPhysicalNames(gmshfile, 2)
# Dirichlet boundary
Dirichlet_bnd = [
    phys_tag_1D["Rectangle-Boundary-Top"][0],
    phys_tag_1D["Rectangle-Boundary-Bot"][0],
]
# Periodic slave boundary
Periodic_slave_bnd = phys_tag_1D["Rectangle-Boundary-Right"]


# Slave boundary -> master boundary map
def periodic_relation(x):
    out_x = np.zeros(x.shape)
    out_x[0] = x_left + (x_right - x[0])
    out_x[1] = x[1]
    out_x[2] = x[2]
    return out_x


# %% Weak formulation
V = dolfinx.fem.functionspace(dmesh.mesh, ("CG", 1))
# Dirichlet boundary condition
x = V.tabulate_dof_coordinates()
uD = dolfinx.fem.Function(V)
uD.x.petsc_vec.setArray(0 * x[:, 0])
uD.x.petsc_vec.ghostUpdate(
    addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
)
bcs = fenicsx_utils.create_DirichletBC(
    dmesh.mesh, dmesh.facet_tags, V, [uD, uD], Dirichlet_bnd
)
# Periodic boundary condition through MultiPointContraint object
# - Slave DoF are removed from the solved system
# - Dirichlet bc cannot be on master DoF
# - Multiple constraints with multiple calls to create_periodic_constraint()
mpc = dolfinx_mpc.MultiPointConstraint(V)
# mpc.create_periodic_constraint(dmesh.facet_tags, Periodic_slave_bnd[0], periodic_relation, bcs)
mpc.create_periodic_constraint_topological(
    V,
    dmesh.facet_tags,
    Periodic_slave_bnd[0],
    periodic_relation,
    bcs,
    float(1.0),
)
mpc.finalize()
# %%
# Variational problem
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
b = ufl.inner(u, v) * ufl.dx
x = ufl.SpatialCoordinate(dmesh.mesh)
dx = x[0] - 0.9
dy = x[1] - 0.5
f = x[0] * ufl.sin(5.0 * ufl.pi * x[1]) + 1.0 * ufl.exp(
    -(dx * dx + dy * dy) / 0.02
)
rhs = ufl.inner(f, v) * ufl.dx
a, b, rhs = dolfinx.fem.form(a), dolfinx.fem.form(b), dolfinx.fem.form(rhs)
# Assemble matrix
import time

start = time.time()
A = dolfinx_mpc.assemble_matrix(a, mpc, bcs=bcs, diagval=1e2)
B = dolfinx_mpc.assemble_matrix(b, mpc, bcs=bcs, diagval=1e-2)
print(f"Python assembly: {time.time() - start}s")
b = dolfinx_mpc.assemble_vector(rhs, mpc)
# Apply boundary conditions
dolfinx.fem.apply_lifting(b, [a], [bcs])
b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
dolfinx.fem.set_bc(b, bcs)
# Solve Linear problem
solver = PETSc.KSP().create(MPI.COMM_WORLD)
opts = PETSc.Options()
opts["ksp_type"] = "cg"
opts["ksp_rtol"] = 1.0e-6
opts["pc_type"] = "hypre"
opts["pc_hypre_type"] = "boomeramg"
opts["pc_hypre_boomeramg_max_iter"] = 1
opts["pc_hypre_boomeramg_cycle_type"] = "v"
# opts["pc_hypre_boomeramg_print_statistics"] = 1
solver.setFromOptions()
solver.setOperators(A)
uh = b.copy()
uh.set(0)
solver.solve(b, uh)
uh.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
# Assign slave DoF (they are not solved for)
mpc.backsubstitution(uh)
# solver.view()
print("Constrained solver iterations {0:d}".format(solver.getIterationNumber()))
# %% Plot solution
grid = fenicsx_utils.create_pyvista_UnstructuredGrid_from_mesh(dmesh.mesh)
# pv.set_plot_theme("document")
colormap = "viridis"
u_h = dolfinx.fem.Function(V)
u_h.x.petsc_vec.setArray(uh.array)
u_h.name = "u_mpc"
grid.clear_data()
grid.point_data["u"] = u_h.x.petsc_vec.array.real
grid.set_active_scalars("u")
plotter = pv.Plotter(shape=(1, 1))
colorbar_args = dict(
    vertical=True,
    height=0.8,
    position_x=0.9,
    position_y=0.1,
    fmt="%.1g",
    title_font_size=1,
    label_font_size=20,
)
plotter.add_mesh(
    grid,
    show_edges=False,
    show_scalar_bar=True,
    scalar_bar_args=colorbar_args,
    cmap=colormap,
)
plotter.view_xy()
plotter.show_bounds(
    xlabel="x", ylabel="y", all_edges=True, minor_ticks=True, font_size=10
)
plotter.show()
# export
# grid.save("result.vtu")
# pv.save_meshio("result.vtk",grid)
# %% Solve GEP
EPS = SLEPc_utils.solve_GEP_shiftinvert(
    A,
    B,
    problem_type=SLEPc.EPS.ProblemType.GHEP,
    solver=SLEPc.EPS.Type.KRYLOVSCHUR,
    nev=5,
    tol=1e-4,
    max_it=10,
    target=1,
    shift=1,
)
(eigval, eigvec_r, eigvec_i) = SLEPc_utils.EPS_get_spectrum(EPS)
for i in range(len(eigval)):
    mpc.backsubstitution(eigvec_r[i])
    mpc.backsubstitution(eigvec_i[i])
# %% Plot spectrum
i_plot = 4
fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
ax.plot(np.real(eigval), np.imag(eigval), marker="o", linestyle="none")
ax.set_xlabel("$Re(\lambda)$")
ax.set_ylabel("$Im(\lambda)$")
ax = fig.add_subplot(1, 2, 2)
grid.clear_data()
grid.point_data["u"] = eigvec_r[i_plot].array / np.max(
    np.abs(eigvec_r[i_plot].array)
)
grid.set_active_scalars("u")
plotter = pv.Plotter(shape=(1, 1), window_size=[1000, 1000])
plotter.add_mesh(grid, show_edges=False, show_scalar_bar=False)
plotter.view_xy()
img = pyvista_utils.get_trimmed_screenshot(plotter)
x_max = np.max(dmesh.mesh.geometry.x, 0)
x_min = np.min(dmesh.mesh.geometry.x, 0)
coll = ax.imshow(
    img, vmin=-1, vmax=1, extent=(x_min[0], x_max[0], x_min[1], x_max[1])
)
fig.colorbar(coll, ax=ax)
ax.set_title(f"$\lambda={eigval[i_plot]:2.2g}$")
ax.set_xlabel("$x$")  # orthonormal axis
ax.set_ylabel("$y$")
ax.grid(False)
ax.set_aspect("equal")  # orthonormal axis
# %% Export all eigenmodes using pyvista
grid.clear_data()
for i in range(len(eigval)):
    name = f"l_{i}_{eigval[i]:4.4g}"
    print(name)
    grid.point_data[name] = eigvec_r[i].array
grid.save("result.vtu")
