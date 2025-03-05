#!/usr/bin/env python
# coding: utf-8

"""
Basics of solving a Dirichlet Laplacian in dolfinx.

Adapted from Chapter 1 of the dolfinx tutorial by Jørgen Schartum Dokken.

Content:
- Load high-order (up to third order) meshes with diagnostic messages and plots
- Solution of the Dirichlet Laplacian on an imported gmsh mesh
- domains and boundaries manipulated through physical tags
- Solution of generalized eigenvalue problem through SLEPc
- Export to vtu file
- Post-processing with pyvista and matplotlib
"""
#%%
    # Matrix
from petsc4py import PETSc
import numpy as np
    # Eigenvalue problem
from slepc4py import SLEPc
from scicomp_utils_misc import SLEPc_utils
from mpi4py import MPI
    # Dolfinx
import dolfinx
import dolfinx.fem.petsc
import ufl
from scicomp_utils_dolfinx import fenicsx_utils
    # Mesh
from scicomp_utils_mesh import gmsh_utils
from scicomp_utils_mesh import meshio_utils
    # Plot
import matplotlib.pyplot as plt
import pyvista as pv
from scicomp_utils_dolfinx import pyvista_utils
import os
DIR_MESH=os.path.join(os.path.dirname(os.path.abspath(__file__)),"mesh")

# TODO: make pyvista optional
# TODO: add comparison to exact eigenvalues
# TODO: validate solution given by mixed approach, so as to be able to export
# both fields of the mixed solution
#%% Create and import mesh using gmsh
geofile=os.path.join(DIR_MESH,"Circle-simple.geo")
gmshfile=os.path.join(DIR_MESH,"Circle-simple.msh")
gmsh_utils.generate_mesh_cli(geofile,gmshfile,2,refinement=1,log=1,\
                             parameters={'R':1,'R_TR':0.5},order=2)
gmsh_utils.print_summary(gmshfile)
phys_tag_1D = gmsh_utils.getPhysicalNames(gmshfile,1)
phys_tag_2D = gmsh_utils.getPhysicalNames(gmshfile,2)
mesh_cells_type = meshio_utils.get_cells_type_str(gmshfile)
dmesh = fenicsx_utils.DolfinxMesh.init_from_gmsh(gmshfile,2)
    # Relax Newton-Rhapson iteration tolerance for interpolation
    # Also neeed to write function to XDMF file
    # See issue #1245.
# dmesh.mesh.geometry.cmap.non_affine_atol = 1e-14
# dmesh.mesh.geometry.cmap.non_affine_max_its = 100
#%% Mesh plot
    # Create pvysta mesh
grid = fenicsx_utils.create_pyvista_UnstructuredGrid_from_mesh(dmesh.mesh)
    # Mark cells
grid.cell_data["Marker"] = dmesh.cell_tags.values
grid.set_active_scalars("Marker")
    # Plot
plotter = pv.Plotter(shape=(1, 1))
# plotter.add_mesh(pv.PolyData(dmesh.mesh.geometry.x), color='red',
#        point_size=10, render_points_as_spheres=True)
plotter.add_mesh(grid, show_edges=True,show_scalar_bar=True)
plotter.add_text(f"{gmshfile}\nNodes: {dmesh.mesh.geometry.x.shape[0] }", position="upper_edge", font_size=8, color="black")
plotter.view_xy()
plotter.show()
#%% Weak formulation
V = dolfinx.fem.functionspace(dmesh.mesh, ("CG", 2))
FE_name = V.element.signature()
    # Dirichlet boundary condition
uD = dolfinx.fem.Function(V)
    # Use interpolation
    # Slow and buggy: loosen non_affine_atol and non_affine_max_its
#uD.interpolate(lambda x: 1 + x[0]**2 + 2 * x[1]**2)
    # Don't use interpolation
x = V.tabulate_dof_coordinates()
y = 1+x[:,0]**2+2*x[:,1]**2
uD.vector.setArray(y)
    # Set up Dirichlet boundary condition
uD.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
bcs = fenicsx_utils.create_DirichletBC(dmesh.mesh,dmesh.facet_tags,V,[uD],[phys_tag_1D['Gamma-ext']])
    # Weak formulation
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
f = dolfinx.fem.Constant(dmesh.mesh, PETSc.ScalarType(-6.0))
kappa = dolfinx.fem.Constant(dmesh.mesh, PETSc.ScalarType(1.0))
    # Using measure built from mesh cell tags
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dmesh.dx(phys_tag_2D['Omega-int'][0]) +     ufl.inner(kappa*ufl.grad(u), ufl.grad(v)) * dmesh.dx(phys_tag_2D['Omega-ext'][0])
b = ufl.inner(u, v) * dmesh.dx
L = ufl.inner(f,v) * dmesh.dx(phys_tag_2D['Omega-int'][0])+ufl.inner(f,v) * dmesh.dx(phys_tag_2D['Omega-ext'][0])
#%% Solve
# Solve direct problem
problem = dolfinx.fem.petsc.LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()
# Error
uex = uD # Exact solution when kappa=1
L2_error = ufl.inner(uh - uex, uh - uex) * dmesh.dx
L2_error = dolfinx.fem.form(L2_error)
error_L2 = np.sqrt(dolfinx.fem.assemble_scalar(L2_error))
print(f"DoF : {V.tabulate_dof_coordinates().shape[0]}")
print(f"Error_L2 : {error_L2:.2e}")
#%% Plot solution: native pyvista method
# May induce crash
# pv.set_plot_theme("paraview")
# pv.set_plot_theme("default")
# pv.set_plot_theme("document")
# Colormap
colormap = 'viridis'
# colormap = 'hsv'
# colormap = 'jet'
# colormap = 'turbo'
grid.clear_data()
grid.point_data["u"] = uh.x.array.real
grid.set_active_scalars("u")
plotter = pv.Plotter(shape=(1, 1))
colorbar_args = dict(vertical=True,height=0.8,position_x=0.9, position_y=0.1,
                     fmt="%.1g",title_font_size=1,label_font_size=20)
plotter.add_mesh(grid, show_edges=False, show_scalar_bar=True,scalar_bar_args=colorbar_args,cmap=colormap)
plotter.add_title(f"{gmshfile}\nNodes: {dmesh.mesh.geometry.x.shape[0] }",font_size=10)
plotter.view_xy()
plotter.show_bounds(xlabel="x",ylabel="y",all_edges=True,minor_ticks=True,font_size=10)
plotter.show()
#% Export using pyvista
#grid.save("result.vtu")
#pv.save_meshio("result.vtk",grid)
#%% Plot solution: alternative method
# Create a pyvista barebone pyvista plotter, screenshot, matplotlib
grid.clear_data()
grid.point_data["u"] = uh.x.array.real
grid.set_active_scalars("u")
plotter = pv.Plotter(shape=(1, 1),window_size=[1000, 1000])
plotter.add_mesh(grid, show_edges=False, show_scalar_bar=False,cmap=colormap)
plotter.view_xy()
img = pyvista_utils.get_trimmed_screenshot(plotter)
x_max=np.max(dmesh.mesh.geometry.x,0); x_min=np.min(dmesh.mesh.geometry.x,0)
fig = plt.figure(1)
ax = fig.add_subplot(1,1,1)
coll = ax.imshow(img,vmin=grid.point_data["u"].min(),
                 vmax=grid.point_data["u"].max(),
                 extent=(x_min[0],x_max[0],x_min[1],x_max[1]))
fig.colorbar(coll,ax=ax)
ax.set_title(f"{gmshfile}\nNodes: {dmesh.mesh.geometry.x.shape[0] } {mesh_cells_type} {FE_name}")
ax.set_xlabel('$x$') # orthonormal axis
ax.set_ylabel('$y$')
ax.grid(False)
ax.set_aspect('equal') # orthonormal axis
#%% Generalized eigenvalue problem
    # Assemble
(A,B) = fenicsx_utils.assemble_GEP(a,b,bcs)
    # Solve
EPS = SLEPc_utils.solve_GEP_shiftinvert(A,B,
                          problem_type=SLEPc.EPS.ProblemType.GHEP,
                          solver=SLEPc.EPS.Type.KRYLOVSCHUR,
                          nev=5,tol=1e-4,max_it=10,
                          target=10,shift=10)
(eigval,eigvec_r,eigvec_i) = SLEPc_utils.EPS_get_spectrum(EPS)
#%% Plot spectrum
i_plot = 4
fig = plt.figure()
ax=fig.add_subplot(1,2,1)
ax.plot(np.real(eigval),np.imag(eigval),marker='o',linestyle='none')
ax.set_xlabel("$Re(\lambda)$")
ax.set_ylabel("$Im(\lambda)$")
ax = fig.add_subplot(1,2,2)
grid.clear_data()
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
grid.clear_data()
for i in range(len(eigval)):
    name = f"l_{i}_{eigval[i]:4.4g}"
    print(name)
    grid.point_data[name] = eigvec_r[i].array
grid.save("result.vtu")