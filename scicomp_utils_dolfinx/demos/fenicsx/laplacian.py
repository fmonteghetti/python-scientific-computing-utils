#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basics of solving a Dirichlet Laplacian in dolfinx. Both the direct and
eigenvalue problems are solved. The direct problem is

    -Delt(u)=f in Omega, u=uD on Gamma.

and the eigenvalue problem is

    -Delta(u)=lambda*u in Omega, u=0 on Gamma.

The script can be run on one or multiple threads:
    python ./MPI-test.py (one thread)
    mpirun -n N ./MPI-test.py (N threads).
The latter can be executed from the ipython prompt using:
    ! mpirun -n N ./MPI-test.py.

The following features are demonstrated:
- Generate high-order mesh using gmsh.
- Generate a distributed dolfinx.mesh.Mesh from a .msh file.
- Solve the direct problem using dolfinx.fem.petsc.LinearProblem()
- Manipulate domains and boundaries using physical tags read from .msh file.
- Solve the generalized eigenvalue problem using SLEPc.
- Export solutions to .xdmf file using dolfinx.io.XDMFFile().
- Export solutions to multiple .vtu files using pyvista Unstructured grid.
- Visualize the solution using pyvista.
- Use PETSc.ScalarType for compatibility with both complex and real PETSc.
"""
#%%
from mpi4py import MPI
comm = MPI.COMM_WORLD # dolfinx, petsc, and slepc use mpi communicators
from petsc4py import PETSc
from slepc4py import SLEPc
import pyvista as pv
import numpy as np
import time
import basix
import dolfinx
import dolfinx.fem.petsc
import ufl
from scicomp_utils_mesh import gmsh_utils
from scicomp_utils_dolfinx import gmsh_utils_fenicsx
from scicomp_utils_dolfinx import fenicsx_utils
from scicomp_utils_misc import SLEPc_utils
import os
DIR_MESH=os.path.join(os.path.dirname(os.path.abspath(__file__)),"mesh")
# TODO: encapsulate in a function, call twice with assert for exact values

degree_mesh = 2
degree_fem = 2
geofile=os.path.join(DIR_MESH,"Disc.geo")
Gamma_name = ['Gamma-R'] # physical name of boundary
Omega_name = ['Omega'] # physical name of domain
dim = 2
geofile=os.path.join(DIR_MESH,"sphere.geo")
Gamma_name = ['Gamma'] # physical name of boundary
Omega_name = ['Omega'] # physical name of domain
dim = 3
uD_expr = lambda x: np.cos(x[0]+x[1]) # Dirichlet data
    # == Create and read mesh
gmshfile = geofile.replace(".geo",".msh")
if comm.rank==0: # generate mesh on first thread only
    gmsh_utils.generate_mesh_cli(geofile,gmshfile,dim,refinement=0,binary=True,
                              parameters={'lc':1/10},order=degree_mesh)
comm.Barrier()

    # Read .msh file and build a distributed mesh
dmesh = fenicsx_utils.DolfinxMesh.init_from_gmsh(gmshfile,dim,comm=comm)
dim, mesh, name_2_tags = dmesh.dim, dmesh.mesh, dmesh.name_2_tags
Gamma_tags = [t for name in Gamma_name for t in name_2_tags[-2][name]]

    # == Weak formulation a(u,v) = l(u,v) with v in H^1
    # Build distributed function space
t0=time.process_time_ns()
V = dolfinx.fem.functionspace(mesh, ("CG", degree_fem))
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
x = ufl.SpatialCoordinate(dmesh.mesh)
uD = dolfinx.fem.Function(V)
uD.interpolate(uD_expr)
    # Update all ghost values from local values
uD.x.scatter_forward()
bcs = fenicsx_utils.create_DirichletBC(dmesh.mesh,dmesh.facet_tags,V,[uD],Gamma_tags)
xc, sigma, A = [0.5,0.5], 0.02, 10
f = A * ufl.exp(-((x[0] - xc[0]) ** 2 + (x[1] - xc[1]) ** 2) / sigma)
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dmesh.dx
l = ufl.inner(f, v) * dmesh.dx
    # Assemble matrix and vector
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
    # Solve
solver = PETSc.KSP().create(comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)
uh = dolfinx.fem.Function(V)
solver.solve(L, uh.x.petsc_vec)
uh.x.scatter_forward()
comm.Barrier()
if comm.rank ==0:
    print(f"Elapsed time (direct problem): {1e-9*(time.process_time_ns()-t0):1.2g}s")

# TODO: validate against exact solution

    # Export uh to a single XDMF file (currently supports only one function)
with dolfinx.io.XDMFFile(mesh.comm, "shared/demo_laplace_direct.xdmf", "w") as file:
    file.write_mesh(mesh)
    file.write_function(uh) # TODO: set the name

    # Export to multiple .vtu files using pyvista (supports multiple functions)
    # In paraview, open all the .vtu file and use the "Group Datasets" filter.
    # Alternative: generate a .ptvu file with comm.size pieces using
    # XMLPUnstructuredGridWriter from vtk (difficult).
grid = fenicsx_utils.create_pyvista_UnstructuredGrid_from_FunctionSpace(V)
grid.point_data["u"] = uh.x.array
grid.save(f"shared/demo_laplace_direct_{comm.rank}.vtu")

    # == Mixed formulation a((phi,u),(psi,v)) = l((psi,v)) 
    # with (psi,u) in H(div)xL^2
t0=time.process_time_ns()
# BDM = ufl.FiniteElement("BDM", dmesh.mesh.ufl_cell(), degree=degree_fem)
# DG = ufl.FiniteElement("DG", dmesh.mesh.ufl_cell(), degree=degree_fem-1)
BDM = basix.ufl.element("BDM", dmesh.mesh.ufl_cell().cellname(), degree=degree_fem)
DG = basix.ufl.element("DG", dmesh.mesh.ufl_cell().cellname(), degree=degree_fem-1)
W = dolfinx.fem.functionspace(dmesh.mesh, basix.ufl.mixed_element([BDM,DG]))
V_phi, V_u = W.sub(0).collapse()[0], W.sub(1).collapse()[0]
trial_vec, test_vec = ufl.TrialFunctions(W), ufl.TestFunctions(W)
phi, u = trial_vec[0], trial_vec[1]
psi, v = test_vec[0], test_vec[1]
n = ufl.FacetNormal(dmesh.mesh)
a, l = list(), list()
a.append(ufl.inner(phi, psi)*dmesh.dx + ufl.inner(u, ufl.div(psi))*dmesh.dx)
l.append(ufl.inner(uD,ufl.inner(psi,n))*dmesh.ds)
a.append(-ufl.inner(ufl.div(phi), v)*dmesh.dx)
l.append(ufl.inner(f,v) * dmesh.dx)
a, l = sum(a), sum(l)
# TODO: why no mpi speedup here?
problem = dolfinx.fem.petsc.LinearProblem(a, l,
                          petsc_options={"ksp_type": "gmres", "pc_type": "bjacobi"})
Uh = problem.solve()
comm.Barrier()
if comm.rank ==0:
    print(f"Elapsed time (direct problem, mixed): {1e-9*(time.process_time_ns()-t0):1.2g}s")
    # Export both fields to multiple .vtu files
phih, uh = Uh.sub(0).collapse(), Uh.sub(1).collapse()
grid = fenicsx_utils.create_pyvista_UnstructuredGrid_from_FunctionSpace(V_u)
grid.point_data["u"] = uh.x.array
    # Project phi in [L^2]^dim
V_proj = dolfinx.fem.functionspace(mesh, ("DG", degree_fem-1, (dmesh.mesh.geometry.dim,)))
phih_proj = fenicsx_utils.project(phih, V_proj,
                      petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
grid.point_data["phi"] = np.reshape(phih_proj.x.array.real,(-1,dim))
grid.save(f"shared/demo_laplace_direct_mixed_{comm.rank}.vtu")
    # == Eigenvalue problem a(u,v) = l(v) with v in H^1
t0 = time.process_time_ns()
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dmesh.dx
b = ufl.inner(u, v) * dmesh.dx
uD.x.array[:] = PETSc.ScalarType(0)
bcs = fenicsx_utils.create_DirichletBC(dmesh.mesh,dmesh.facet_tags,V,[uD],Gamma_tags)
(A,B) = fenicsx_utils.assemble_GEP(a,b,bcs)
EPS = SLEPc_utils.solve_GEP_shiftinvert(A,B,comm=comm,
                      problem_type=SLEPc.EPS.ProblemType.GHEP,
                      solver=SLEPc.EPS.Type.KRYLOVSCHUR,
                      nev=5,tol=1e-4,max_it=10,
                      target=10,shift=10)
(eigval,eigvec_r,eigvec_i) = SLEPc_utils.EPS_get_spectrum(EPS)
# TODO: validate against exact eigenvalues
comm.Barrier()
if comm.rank ==0:
    print(f"Elapsed time (eigenvalue problem): {1e-9*(time.process_time_ns()-t0):1.2g}s")

    # Export eigenfunctions to several .vtu file using pyvista
grid = fenicsx_utils.create_pyvista_UnstructuredGrid_from_FunctionSpace(V)
u_out = dolfinx.fem.Function(V)
for i in range(len(eigval)):
    name = f"l_{i}_{eigval[i]:4.4g}"
        # eigvec_r[i] contains only the local dofs, not the ghost dofs
    u_out.x.petsc_vec.setArray(eigvec_r[i])
    u_out.x.scatter_forward()
    grid.point_data[name+"_ur"] = u_out.x.array.copy()
    u_out.x.petsc_vec.setArray(eigvec_i[i])
    u_out.x.scatter_forward()
    grid.point_data[name+"_ui"] = u_out.x.array.copy()
grid.save(f"shared/demo_laplace_eig_{comm.rank}.vtu")

    # == Illustration of the mesh and dof distribution
    # In an MPI communicator, each process has a rank
def mpi_print(s):
    print(f"Rank {comm.rank}: {s}")
    # The mesh is distributed among processes
for d in range(dim+1):
    mpi_print(f"# of local (global) element of dim={d}: {mesh.topology.index_map(d).size_local} ({mesh.topology.index_map(d).size_global})")
    # Each process stores 'local' and 'ghost' dofs of functions in V.
mpi_print(f"# of ghost| local | global dofs: {V.dofmap.index_map.num_ghosts} | {V.dofmap.index_map.size_local} | {V.dofmap.index_map.size_global}")
mpi_print(f"Global indices of local dofs: {V.dofmap.index_map.local_range}")
    # Beware of the difference between uh.x and uh.x.petsc_vec
mpi_print(f"Size of uh.x.petsc_vec (= # of global dofs): {uh.x.petsc_vec.size}")
mpi_print(f"Local size of uh.x.petsc_vec (= # of local dofs): {uh.x.petsc_vec.sizes[0]}")
mpi_print(f"Size of uh.x (= # of local + ghost dofs): {uh.x.array.size}")
# %%
