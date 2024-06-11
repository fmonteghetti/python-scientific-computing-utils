#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is a re-implementation of fenics_periodicity_using_gmsh.py using fenicsx.

"""
    # Matrix
from petsc4py import PETSc
import numpy as np
    # Eigenvalue problem
from slepc4py import SLEPc
from scicomp_utils_misc import SLEPc_utils
from mpi4py import MPI
    # Dolfinx
import dolfinx
import dolfinx_mpc
import ufl
from scicomp_utils_dolfinx import fenicsx_utils
    # Mesh
from scicomp_utils_mesh import gmsh_utils
from scicomp_utils_mesh import meshio_utils
    # Plot
import matplotlib.pyplot as plt
import pyvista as pv
from scicomp_utils_misc import pyvista_utils
import os
DIR_MESH=os.path.join(os.path.dirname(os.path.abspath(__file__)),"mesh")
#%% Import mesh using gmsh
gmshfile=os.path.join(DIR_MESH,"Circle-Rectangle-Periodic.msh")
geofile=os.path.join(DIR_MESH,"Circle-Rectangle-Periodic.geo")
gmsh_utils.generate_mesh_cli(geofile,gmshfile,2,refinement=0,binary=True)
gmsh_utils.print_summary(gmshfile)
mesh_cells_type = meshio_utils.get_cells_type_str(gmshfile)
dmesh = fenicsx_utils.DolfinxMesh.init_from_gmsh(gmshfile,2)
phys_tag_1D = gmsh_utils.getPhysicalNames(gmshfile,1)
phys_tag_2D = gmsh_utils.getPhysicalNames(gmshfile,2)
    # Geometrical parameters (from .geo file)
R = 1
R_TR=0.5*R
y_offset = -R-np.pi
dx = dmesh.dx
    # Dirichlet boundary  
Dirichlet_bnd = phys_tag_1D['Rectangle-Boundary-Right']
    # Periodic boundary condition
pbc_slave_bnd = list() # slave boundary
pbc_slave_to_master_map = list() # map slave to master boundary
        # 1 - Rectangle bottom -> Rectangle top
pbc_slave_bnd.append(phys_tag_1D['Rectangle-Boundary-Bot'][0])
def map_RectBotBnd_to_RectTopBnd(x):
    y = np.zeros(x.shape)
    y[0] = x[0]; y[1] = x[1]+2*np.pi; y[2] = x[2]
    return y
pbc_slave_to_master_map.append(map_RectBotBnd_to_RectTopBnd)
        # 2- Rectangle left -> Disk boundary
pbc_slave_bnd.append(phys_tag_1D['Rectangle-Boundary-Left'][0])
def map_RectLeftBnd_to_DiskBnd(x):
    y = np.zeros(x.shape)
    y[0] = R_TR*np.cos(x[1]-y_offset); y[1] = R_TR*np.sin(x[1]-y_offset); y[2]=x[2]
    return y
pbc_slave_to_master_map.append(map_RectLeftBnd_to_DiskBnd)
#%% Weak formulation
V = dolfinx.fem.FunctionSpace(dmesh.mesh, ("CG", 1))
    # Dirichlet boundary condition
x = V.tabulate_dof_coordinates()
uD = dolfinx.fem.Function(V)
uD.vector.setArray(0*x[:,0])
uD.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
bcs = fenicsx_utils.create_DirichletBC(dmesh.mesh,dmesh.facet_tags,V,[uD],Dirichlet_bnd)
    # Periodic boundary condition
mpc = dolfinx_mpc.MultiPointConstraint(V)
for i in range(len(pbc_slave_bnd)):
    mpc.create_periodic_constraint_topological(V, dmesh.facet_tags,
                                                pbc_slave_bnd[i],
                                                pbc_slave_to_master_map[i],
                                                bcs,
                                                float(1.0))    
mpc.finalize()
    # Variational problem
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
x = ufl.SpatialCoordinate(dmesh.mesh)
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
weight = ufl.exp(2*x[0])
m = ufl.inner(u, v) * dx(phys_tag_2D['Disk'][0])+\
    weight*ufl.inner(u, v) * dx(phys_tag_2D['Rectangle'][0])
f = PETSc.ScalarType(-6)
l = ufl.inner(f,v)*dx(phys_tag_2D['Disk'][0])+\
    weight*ufl.inner(f,v)*dx(phys_tag_2D['Rectangle'][0])
a,m,l = dolfinx.fem.form(a), dolfinx.fem.form(m), dolfinx.fem.form(l)
    # Assemble matrix
import time
start = time.time()
A = dolfinx_mpc.assemble_matrix(a, mpc, bcs=bcs,diagval=1e2)
M = dolfinx_mpc.assemble_matrix(m, mpc, bcs=bcs,diagval=1e-2)
print(f"Python assembly: {time.time()-start}s")
L = dolfinx_mpc.assemble_vector(l, mpc)
    # Apply boundary conditions
dolfinx.fem.apply_lifting(L, [a], [bcs])
L.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
              mode=PETSc.ScatterMode.REVERSE)
dolfinx.fem.set_bc(L, bcs)
#%% Solve direct problem
solver = PETSc.KSP().create(MPI.COMM_WORLD)
opts = PETSc.Options()
opts["ksp_type"] = "cg"
opts["ksp_rtol"] = 1.0e-6
opts["pc_type"] = "hypre"
opts['pc_hypre_type'] = 'boomeramg'
opts["pc_hypre_boomeramg_max_iter"] = 1
opts["pc_hypre_boomeramg_cycle_type"] = "v"
# opts["pc_hypre_boomeramg_print_statistics"] = 1
solver.setFromOptions()
solver.setOperators(A)
uh = L.copy()
uh.set(0)
solver.solve(L, uh)
uh.ghostUpdate(addv=PETSc.InsertMode.INSERT,
               mode=PETSc.ScatterMode.FORWARD)
    # Assign slave DoF (they are not solved for)
mpc.backsubstitution(uh)
# solver.view()
print("Constrained solver iterations {0:d}".format(solver.getIterationNumber()))
#%% Plot solution
grid = fenicsx_utils.create_pyvista_UnstructuredGrid_from_mesh(dmesh.mesh)
# pv.set_plot_theme("document")
colormap = 'viridis'
u_h = dolfinx.fem.Function(V)
u_h.vector.setArray(uh.array)
u_h.name = "u_mpc"
grid.clear_arrays()
grid.point_data["u"] = u_h.vector.array.real
grid.set_active_scalars("u")
plotter = pv.Plotter(shape=(1, 1))
colorbar_args = dict(vertical=True,height=0.8,position_x=0.9, position_y=0.1,
                     fmt="%.1g",title_font_size=1,label_font_size=20)
plotter.add_mesh(grid, show_edges=False, show_scalar_bar=True,scalar_bar_args=colorbar_args,cmap=colormap)
plotter.view_xy()
plotter.show_bounds(xlabel="x",ylabel="y",all_edges=True,minor_ticks=True,font_size=10)
plotter.show()
    # export
# grid.save("result.vtu")
# pv.save_meshio("result.vtk",grid)
#%% Solve GEP
EPS = SLEPc_utils.solve_GEP_shiftinvert(A,M,
                          problem_type=SLEPc.EPS.ProblemType.GHEP,
                          solver=SLEPc.EPS.Type.KRYLOVSCHUR,
                          nev=20,tol=1e-4,max_it=10,
                          target=1,shift=1)
(eigval,eigvec_r,eigvec_i) = SLEPc_utils.EPS_get_spectrum(EPS)
for i in range(len(eigval)):
    mpc.backsubstitution(eigvec_r[i])
    mpc.backsubstitution(eigvec_i[i])
#%% Plot spectrum
i_plot = 4
fig = plt.figure()
ax=fig.add_subplot(1,2,1)
ax.plot(np.real(eigval),np.imag(eigval),marker='o',linestyle='none')
ax.set_xlabel("$Re(\lambda)$")
ax.set_ylabel("$Im(\lambda)$")
ax = fig.add_subplot(1,2,2)
grid.clear_arrays()
grid.point_data["u"] = eigvec_r[i_plot].array/np.max(np.abs(eigvec_r[i_plot].array))
grid.set_active_scalars("u")
plotter = pv.Plotter(shape=(1, 1),window_size=[1000, 1000])
plotter.add_mesh(grid, show_edges=False, show_scalar_bar=False)
plotter.view_xy()
img = pyvista_utils.get_trimmed_screenshot(plotter)
x_max=np.max(dmesh.mesh.geometry.x,0); x_min=np.min(dmesh.mesh.geometry.x,0)
coll = ax.imshow(img,vmin=-1,vmax=1,extent=(x_min[0],x_max[0],x_min[1],x_max[1]))
fig.colorbar(coll,ax=ax)
ax.set_title(f"$\lambda={eigval[i_plot]:2.2g}$")
ax.set_xlabel('$x$') # orthonormal axis
ax.set_ylabel('$y$')
ax.grid(False)
ax.set_aspect('equal') # orthonormal axis
#%% Export all eigenmodes using pyvista
grid.clear_arrays()
for i in range(len(eigval)):
    name = f"l_{i}_{eigval[i]:4.4g}"
    print(name)
    grid.point_data[name] = eigvec_r[i].array
grid.save("result.vtu")