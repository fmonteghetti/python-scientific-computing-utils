#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Poisson equation under standard form:
    
- Delta(u) = f on Omega, u = u_D on Gamma-D, du/dn = u_N on Gamma-N,

and mixed form:
    
- div(sigma) = f, sigma = grad(u) on Omega,
u = u_D on Gamma-D, (sigma,n) = u_N on Gamma-N.

Demonstrates the use of multiphenics to assemble boundary Lagrange multiplier.

Both 2D and 3D meshes can be used.

"""
import numpy as np
import matplotlib.pyplot as plt
import fenics as fe
from scicomp_utils_mesh import gmsh_utils
from scicomp_utils_dolfin import fenics_utils
from scicomp_utils_mesh import meshio_utils
import multiphenics as mpfe
from scicomp_utils_dolfin import multiphenics_utils
from petsc4py import PETSc
fe.SubSystemsManager.init_petsc()
PETSc.Sys.pushErrorHandler("python")
from scicomp_utils_misc import PETSc_utils
import os
DIR_MESH=os.path.join(os.path.dirname(os.path.abspath(__file__)),"mesh")
#%% Generate mesh using gmsh (2D)
geofile="mesh/mesh/Annulus.geo"; gmshfile="mesh/mesh/Annulus.msh"
geofile=os.path.join(DIR_MESH,"Annulus.geo")
gmshfile=os.path.join(DIR_MESH,"Annulus.msh")
dim = 2
gmsh_utils.generate_mesh_cli(geofile,gmshfile,dim,refinement=1,log=1,\
                             parameters={'R':1,'R_TR':0.5, 'lc': 1/20},order=1,
                             gmshformat=2,binary=True)
gmsh_utils.print_summary(gmshfile)
Dirichlet_bnd_name = 'Gamma-R'
Neumann_bnd_name = 'Gamma-RTR'
#%% Generate mesh using gmsh (3D)
geofile="mesh/mesh/sphere-shell.geo"; gmshfile="mesh/mesh/sphere-shell.msh"
geofile=os.path.join(DIR_MESH,"sphere-shell.geo")
gmshfile=os.path.join(DIR_MESH,"sphere-shell.msh")
dim = 3
gmsh_utils.generate_mesh_cli(geofile,gmshfile,dim,refinement=0,log=1,\
                             parameters={'R_i':0.5,'R_o':1,'lc':1/5},
                             order=1,binary=True)
gmsh_utils.print_summary(gmshfile)
Dirichlet_bnd_name = 'Gamma-Outer'
Neumann_bnd_name = 'Gamma-Inner'
#%% Load mesh: XDMF format
meshio_utils.print_diagnostic_gmsh(gmshfile)
xdmfiles = meshio_utils.convert_gmsh_to_XDMF(gmshfile,prune_z=True)
dmesh = fenics_utils.DolfinMesh.init_from_xdmf(xdmfiles['tetra'*(dim==3)+'triangle'*(dim==2)],
                                               xdmfiles['triangle'*(dim==3)+'line'*(dim==2)])
meshtag = meshio_utils.read_phys_gmsh(xdmfiles['gmsh_physical_entities'])
# fe.plot(dmesh.mesh,title=f"{dmesh.mesh.num_vertices()} vertices")
#%% Problem statement
u_D_expr = 'cos(10*x[0])'
#u_D_expr = 'cos(10*atan(x[1]/x[0]))'
u_D_expr = '1'
u_N_expr = 'cos(10*atan(x[1]/x[0]))'
#u_N_expr = '4'
f_expr = 'sin(3*x[0] + 1)*sin(3*x[1] + 1)'
f_expr = '1'
Dirichlet_bnd = meshtag[dim-2][Dirichlet_bnd_name][0]
Neumann_bnd = meshtag[dim-2][Neumann_bnd_name][0]
#%% 1 -- Standard Formulation - Strong enforcement of Dirichlet b.c.
# (grad(u),grad(v)) = (f,v) + (u_N,v)_Gamma_N,
# with:  - v=0, u=u_D on Gamma_D (enforced at assembly)
V = fe.FunctionSpace(dmesh.mesh, 'P', 1)
u_D = fe.Expression(u_D_expr, degree=2)
u_N = fe.Expression(u_N_expr, degree=2)
f = fe.Expression(f_expr, degree=2)
bc = fe.DirichletBC(V, u_D, dmesh.boundaries,Dirichlet_bnd)
u = fe.TrialFunction(V)
v = fe.TestFunction(V)
a = fe.dot(fe.grad(u), fe.grad(v))*dmesh.dx
L = f*v*dmesh.dx + u_N*v*dmesh.ds(Neumann_bnd)
u = fe.Function(V)
fe.solve(a == L, u, bc)
fenics_utils.export_xdmf("poisson-equation.xdmf",["u"],[V],[0],[u.vector().vec()])
if dim==2:
    c = fe.plot(u)
    ax=c.axes
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(c)
    #fe.plot(dmesh.mesh)
print(f"Total DoF = {u.vector().size()}")
#%% 1 -- Standard Formulation Lagrange multiplier for Dirichlet b.c. using multiphenics
# (grad(u),grad(v)) - (l,v)_Gamma_D = (f,v) + (u_N,v)_Gamma_N,
# (u,m)_Gamma_D = (u_D,m)_Gamma_D,
# where Lagrange multiplier l is defined over Gamma)_D.
    # Mesh resriction on selected boundary only
boundary_restriction =  multiphenics_utils.build_MeshRestriction_from_tags(dmesh.mesh,
                                                dmesh.boundaries,
                                                [Dirichlet_bnd])

V = fe.FunctionSpace(dmesh.mesh, "Lagrange", 1)
W = mpfe.BlockFunctionSpace([V, V], restrict=[None, boundary_restriction])

ul = mpfe.BlockTrialFunction(W)
(u, l) = mpfe.block_split(ul)
vm = mpfe.BlockTestFunction(W)
(v, m) = mpfe.block_split(vm)

u_D = fe.Expression(u_D_expr, element=V.ufl_element())
u_N = fe.Expression(u_N_expr, element=V.ufl_element())
f = fe.Expression(f_expr, element=V.ufl_element())
a = [[fe.inner(fe.grad(u), fe.grad(v))*dmesh.dx, -l*v*dmesh.ds(Dirichlet_bnd)],
     [u*m*dmesh.ds(Dirichlet_bnd)                    , 0     ]]
f =  [v*f*dmesh.dx + u_N*v*dmesh.ds(Neumann_bnd)                , u_D*m*dmesh.ds(Dirichlet_bnd)]

A = mpfe.block_assemble(a)
F = mpfe.block_assemble(f)

U = mpfe.BlockFunction(W)
mpfe.block_solve(A, U.block_vector(), F)

if dim==2:
    plt.figure()
    c = fe.plot(U[0])
    ax=c.axes
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(c)
print(f"Total DoF = {A.size(0)}")
xdmfile = 'poisson-equation.xdmf'
sol_idx=[0]; sol_name = ["u"]
multiphenics_utils.export_xdmf(xdmfile,sol_idx,sol_name,[0],[U.block_vector().vec()],W)
#%% Sparse solve using scipy: OK
import scipy.sparse as sp
import scipy.sparse.linalg as la
(indptr,indices,val) = A.mat().getValuesCSR()
A_scipy = sp.csr_matrix((val, indices, indptr), shape=A.mat().size)
F_np = F.vec().array
#b = np.ones((A.mat().size[0],1))
x = la.spsolve(A_scipy,F_np)
x = la.gmres(A_scipy,F_np); x=x[0]
if dim ==2:
    fun = fe.Function(V,fe.PETScVector(PETSc.Vec().createWithArray(x)))
    fe.plot(fun)
xdmfile = 'poisson-equation.xdmf'
sol_idx=[0]; sol_name = ["u"]
multiphenics_utils.export_xdmf(xdmfile,sol_idx,sol_name,[0],[PETSc.Vec().createWithArray(x)],W)
#%% Solve using PETSc
idx_u = W.block_dofmap().block_owned_dofs__global_numbering(0)
idx_l = W.block_dofmap().block_owned_dofs__global_numbering(1)
A_petsc = A.mat()
F_petsc = F.vec()
PETSc_utils.set_diagonal_entries(A_petsc)
OptDB = PETSc_utils.get_cleared_options_db()
    # GMRES with preconditionner
OptDB["ksp_type"] = "gmres"
OptDB["pc_type"] = "ilu" # none, jacobi, ilu
OptDB["ksp_gmres_restart"] = 100
OptDB["ksp_tol"] = 1e-10
    # Sparse LU: MUMPS
# OptDB["ksp_type"] = "preonly"
# OptDB["pc_type"] = "lu"
# OptDB["pc_factor_mat_solver_type"] = "mumps"
    # Alternative more suited for saddle-point problems
# OptDB["ksp_type"] = "gmres"
# OptDB["pc_type"] = "fieldsplit"
# OptDB["pc_fieldsplit_detect_saddle_point"] = True
# OptDB["pc_fieldsplit_type"] = "schur"
    # Initialize ksp solver.
ksp = PETSc.KSP()
ksp.create(PETSc.COMM_WORLD)
ksp.setFromOptions()
    # obtain sol & rhs vectors
x, b = A_petsc.createVecs()
x.set(0)
b.set(1)
ksp.setOperators(A_petsc)
ksp.setFromOptions()
ksp.solve(F_petsc, x)
    # Print results.
print(f'Converged in {ksp.getIterationNumber()} iterations.')
if dim==2:
    fun = fe.Function(V,fe.PETScVector(x))
    fe.plot(fun)
xdmfile = 'poisson-equation.xdmf'
sol_idx=[0]; sol_name = ["u"]
multiphenics_utils.export_xdmf(xdmfile,sol_idx,sol_name,[0],[x],W)
#%% 2 -- Mixed Formulation, grad-grad
# (sigma,grad(v)) + (grad(u),tau) - (sigma,tau) - (l,v)_Gamma_D = (f,v) + (u_N,v)_Gamma_N,
# (u,m)_Gamma_D = (u_D,m)_Gamma_D,
# where Lagrange multiplier l is defined over Gamma)_D.

V_u = fe.FunctionSpace(dmesh.mesh, 'P', 1)
V_sigma = fe.VectorFunctionSpace(dmesh.mesh, 'P', 1)
V_bnd = fe.FunctionSpace(dmesh.mesh, 'P', 1)
    # Block function space
boundary_restriction =  multiphenics_utils.build_MeshRestriction_from_tags(dmesh.mesh,
                                                dmesh.boundaries,
                                                [Dirichlet_bnd])
W = mpfe.BlockFunctionSpace([V_u, V_sigma, V_bnd], restrict=[None, None, boundary_restriction])
# TRIAL/TEST FUNCTIONS #
ul = mpfe.BlockTrialFunction(W)
(u, sigma, l) = mpfe.block_split(ul)
vm = mpfe.BlockTestFunction(W)
(v, tau, m) = mpfe.block_split(vm)
# ASSEMBLE #
u_D = fe.Expression(u_D_expr, element=V_u.ufl_element())
u_N = fe.Expression(u_N_expr, element=V_u.ufl_element())
f = fe.Expression(f_expr, element=V_u.ufl_element())

a = [[0, fe.inner(sigma, fe.grad(v))*dmesh.dx, -l*v*dmesh.ds(Dirichlet_bnd)],
     [fe.inner(fe.grad(u), tau)*dmesh.dx, -fe.inner(sigma, tau)*dmesh.dx, 0],
     [u*m*dmesh.ds(Dirichlet_bnd), 0, 0]]
f = [v*f*dmesh.dx + u_N*v*dmesh.ds(Neumann_bnd), 0, u_D*m*dmesh.ds(Dirichlet_bnd)]

# SOLVE #
A = mpfe.block_assemble(a)
F = mpfe.block_assemble(f)

U = mpfe.BlockFunction(W)
mpfe.block_solve(A, U.block_vector(), F)

if dim==2:
    plt.figure()
    c = fe.plot(U[0])
    ax=c.axes
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(c)
xdmfile = 'poisson-equation.xdmf'
sol_idx=[0]; sol_name = ["u"]
multiphenics_utils.export_xdmf(xdmfile,sol_idx,sol_name,[0],[U.block_vector().vec()],W)
#%% 2 -- Mixed Formulation, div-div
# (div(sigma),v) + (u,div(tau)) + (sigma,tau) - (l,(tau,n))_Gamma_N = -(f,v) + (u_D,(tau,n))_Gamma_D,
# ((sigma,n),m)_Gamma_N = (u_N,m)_Gamma_N,
# where Lagrange multiplier l is defined over Gamma)_N.
    # First possibility
V_u = fe.FunctionSpace(dmesh.mesh, 'P', 1)
V_sigma = fe.FunctionSpace(dmesh.mesh, 'RT', 2) # Raviart-Thomas
V_bnd = fe.FunctionSpace(dmesh.mesh, 'P', 1)
    # Second possibility (Neumann b.c. not working)
# V_u = fe.FunctionSpace(dmesh.mesh, 'DG', 0)
# V_sigma = fe.FunctionSpace(dmesh.mesh, 'BDM', 1)
# V_bnd = fe.FunctionSpace(dmesh.mesh, 'DG', 0)
    # Block function space
boundary_restriction =  multiphenics_utils.build_MeshRestriction_from_tags(dmesh.mesh,
                                                dmesh.boundaries,
                                                [Neumann_bnd])
W = mpfe.BlockFunctionSpace([V_u, V_sigma, V_bnd], restrict=[None, None, boundary_restriction])
# TRIAL/TEST FUNCTIONS #
ul = mpfe.BlockTrialFunction(W)
(u, sigma, l) = mpfe.block_split(ul)
vm = mpfe.BlockTestFunction(W)
(v, tau, m) = mpfe.block_split(vm)
# ASSEMBLE #
u_D = fe.Expression(u_D_expr, element=V_u.ufl_element())
u_N = fe.Expression(u_N_expr, element=V_u.ufl_element())
f = fe.Expression(f_expr, element=V_u.ufl_element())
n = fe.FacetNormal(dmesh.mesh)
a = [[0, fe.div(sigma)*v*dmesh.dx, 0],
      [u*fe.div(tau)*dmesh.dx, fe.inner(sigma, tau)*dmesh.dx, -l*fe.dot(tau,n)*dmesh.ds(Neumann_bnd)],
      [0, m*fe.dot(sigma,n)*dmesh.ds(Neumann_bnd), 0]]
f = [-v*f*dmesh.dx, u_D*fe.dot(tau,n)*dmesh.ds(Dirichlet_bnd), u_N*m*dmesh.ds(Neumann_bnd)]
# SOLVE #
A = mpfe.block_assemble(a)
F = mpfe.block_assemble(f)

U = mpfe.BlockFunction(W)
mpfe.block_solve(A, U.block_vector(), F)
if dim==2:
    plt.figure()
    c = fe.plot(U[0])
    ax=c.axes
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(c)
xdmfile = 'poisson-equation.xdmf'
sol_idx=[0]; sol_name = ["u"]
multiphenics_utils.export_xdmf(xdmfile,sol_idx,sol_name,[0],[U.block_vector().vec()],W)
#%% get boundary DoF (modified to work with gmsh)
# boundaries = dmesh.boundaries, boundary_id = PhysName_1D_tag['Gamma-R'][0]
def subdomain_dofs(V, boundaries,boundary_id, dof):
    # Define a vector u that corresponds to boundary dofs
    # 0 -> not in the subdomain
    # 1 -> in the subdomain, 0th dimension
    # 2 -> in the subdomain, 1th dimension etc.
    bc0 = fe.DirichletBC(V, range(1,V.num_sub_spaces()+1), boundaries, boundary_id)
    u = fe.Function(V)
    bc0.apply(u.vector())
    u_mpi =  fe.PETScVector()
    u.vector().gather(u_mpi, np.array(range(V.dim()), "intc"))
    # Return the dofs that correspond to input dimension
    return np.where(u_mpi==dof+1)[0].tolist()

V = fe.VectorFunctionSpace(dmesh.mesh, "Lagrange", 1, 2)
#V = fe.FunctionSpace(dmesh.mesh, 'P', 1)
u_x_top_dofs = subdomain_dofs(V, dmesh.boundaries, PhysName_1D_tag['Gamma-R'][0], 0)
u_y_top_dofs = subdomain_dofs(V, dmesh.boundaries, PhysName_1D_tag['Gamma-R'][0], 1)

print("x dofs for top boundary: "+ str(u_x_top_dofs))
print("y dofs for top boundary: "+ str(u_y_top_dofs))