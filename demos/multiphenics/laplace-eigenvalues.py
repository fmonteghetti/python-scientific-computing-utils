#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script demonstrates the computation of eigenvalues of the Laplace 
operator. The eigenproblem is: Find (lambda,u) such that

            -div(grad(u)) = lambda * u  on Omega
            u = 0 or du/dn=z*u on Gamma.
Remarks:
    - Both 2D and 3D geometries can be used
    - Boundary condition on Gamma is implemented using a boundary Lagrange
multiplier.
"""

import numpy as np
import matplotlib.pyplot as plt
import fenics as fe
import gmsh_utils
import fenics_utils
import meshio_utils
import multiphenics as mpfe
import multiphenics_utils
from petsc4py import PETSc
fe.SubSystemsManager.init_petsc()
PETSc.Sys.pushErrorHandler("python")
import PETSc_utils
from slepc4py import SLEPc
import SLEPc_utils
import PDE_exact
import os
DIR_MESH=os.path.join(os.path.dirname(os.path.abspath(__file__)),"mesh")

    # User input
geometry = "rectangle" # ball, cube, rectangle, disc
boundary_condition = "dirichlet" # dirichlet, neumann, robin
robin_coeff = 1 # robin boundary condition
degree = 1 # polynomial degree
boundary_DG = False # use DG space for boundary Lagrange multiplier
solver = "direct" # iterative, direct

    # Geometry definition
exact_eigval = lambda N: np.array([])
if geometry=="ball":
    dim = 3
    geofile=os.path.join(DIR_MESH,"sphere.geo")
    gmshfile=os.path.join(DIR_MESH,"sphere.msh")
    Gamma_bnd_name = ['Gamma']
    R = 1.2
    param = {'R':R,'lc':R/30}
    exact_eigval = lambda N: PDE_exact.Laplace_spectrum.ball(R,boundary_condition,
                      eigval_min=0,eigval_max=100,eigval_N=N,eigval_maxorder=10)
elif geometry=="cube":
    dim = 3
    geofile=os.path.join(DIR_MESH,"cube.geo")
    gmshfile=os.path.join(DIR_MESH,"cube.msh")
    Gamma_bnd_name = ['Gamma']
    Lx, Ly, Lz = 1, 2, 0.7
    param = {'Lx':Lx,'Ly':Ly,'Lz':Lz,'lc':Lx/25}
    exact_eigval = lambda N: PDE_exact.Laplace_spectrum.cube(Lx,Ly,Lz,N,boundary_condition)
elif geometry=="disc":
    dim = 2
    geofile=os.path.join(DIR_MESH,"Disc.geo")
    gmshfile=os.path.join(DIR_MESH,"Disc.msh")
    Gamma_bnd_name = ['Gamma-R']
    R = 1.3
    param = {'R':R, 'lc':R/50}
    exact_eigval = lambda N: PDE_exact.Laplace_spectrum.disc(R,boundary_condition,
                      eigval_min=0,eigval_max=100,eigval_N=N,eigval_maxorder=10)
elif geometry=="rectangle":
    dim = 2
    geofile=os.path.join(DIR_MESH,"Rectangle.geo")
    gmshfile=os.path.join(DIR_MESH,"Rectangle.msh")
    Gamma_bnd_name = ['Rectangle-Boundary-Top', 'Rectangle-Boundary-Bot', 
                      'Rectangle-Boundary-Left', 'Rectangle-Boundary-Right']
    Lx, Ly = 1, 2
    param = {'x_r':Lx,'y_t':Ly,'N_x': 100,'N_y': 100}
    exact_eigval = lambda N: PDE_exact.Laplace_spectrum.rectangle(Lx,Ly,N,boundary_condition)

    # Generate mesh
gmsh_utils.generate_mesh_cli(geofile,gmshfile,dim,refinement=0,log=1,\
                             parameters=param,
                             order=1,binary=True)
gmsh_utils.print_summary(gmshfile)

    # Load mesh: XDMF format
meshio_utils.print_diagnostic_gmsh(gmshfile)
xdmfiles = meshio_utils.convert_gmsh_to_XDMF(gmshfile,prune_z=True)
dmesh = fenics_utils.DolfinMesh.init_from_xdmf(xdmfiles['tetra'*(dim==3)+'triangle'*(dim==2)],
                                               xdmfiles['triangle'*(dim==3)+'line'*(dim==2)])
meshtag = meshio_utils.read_phys_gmsh(xdmfiles['gmsh_physical_entities'])
Gamma_bnd = list()
for name in Gamma_bnd_name:
    for tag in meshtag[dim-2][name]:
        Gamma_bnd.append(tag)

boundary_restriction =  multiphenics_utils.build_MeshRestriction_from_tags(dmesh.mesh,
                                                dmesh.boundaries,
                                                Gamma_bnd)
    # (u,l) in H^1(Omega) x L^2(Gamma)
    # l=du/dn is the scalar-valued boundary Lagrange multiplier.
V = fe.FunctionSpace(dmesh.mesh, "Lagrange", degree)
V_DG = fe.FunctionSpace(dmesh.mesh, "DGT", degree-1)
if boundary_DG:
    W = mpfe.BlockFunctionSpace([V, V_DG], restrict=[None, boundary_restriction])
else:
    W = mpfe.BlockFunctionSpace([V, V], restrict=[None, boundary_restriction])

ul = mpfe.BlockTrialFunction(W)
(u, l) = mpfe.block_split(ul)
vm = mpfe.BlockTestFunction(W)
(v, m) = mpfe.block_split(vm)

a, b = list(), list()
a.append([fe.inner(fe.grad(u), fe.grad(v))*dmesh.dx, -l*v*dmesh.ds])
b.append([fe.inner(u, v)*dmesh.dx, 0])
    
if boundary_condition == "dirichlet": # Dirichlet boundary condition
    a.append([u*m*dmesh.ds,0])
    b.append([0,0])
elif boundary_condition == "neumann": # Neumann boundary condition
    a.append([0,l*m*dmesh.ds])
    b.append([0,0])    
elif boundary_condition == "robin": # Robin boundary condition
    z = fe.Constant(robin_coeff)
    a.append([-z*u*m*dmesh.ds,l*m*dmesh.ds])
    b.append([0,0])

    # The generalized eigenvalue problem is assembled as
    #   a(U,V)=lambda*b(U,V).
A, B = mpfe.block_assemble(a,keep_diagonal=True), mpfe.block_assemble(b,keep_diagonal=True)
A_petsc = A.mat(); B_petsc = B.mat()

import time
start = time.time()
OptDB = PETSc_utils.get_cleared_options_db()
OptDB = PETSc.Options()
if solver=="iterative":
    OptDB["st_ksp_type"] = "fgmres" # fgmres, gmres, gcr, ibcgs
    OptDB["st_pc_type"] = "ilu" # ilu, bjacobi, icc
    OptDB["st_ksp_rtol"] = 1e-4
else:     # Direct solver (MUMPS)
    OptDB["st_ksp_type"] = "preonly"
    OptDB["st_pc_type"] = "lu"
    OptDB["st_pc_factor_mat_solver_type"] = "mumps" # mumps, umfpack
SLEPc_params = {'nev': 20,
              'target': 1, 'shift': 1,
              'problem_type': SLEPc.EPS.ProblemType.GHEP,
              'solver': SLEPc.EPS.Type.KRYLOVSCHUR,
              'tol': 1e-4,
              'max_it': 100}
EPS = SLEPc_utils.solve_GEP_shiftinvert(A_petsc,B_petsc,**SLEPc_params)
(eigval,eigvec_r,eigvec_i) = SLEPc_utils.EPS_get_spectrum(EPS)
end = time.time()
print(f"SLEPc elapsed time: {end - start:1.2e}s")
    # Plot eigenvalues
f = plt.figure()
ax=f.add_subplot(1,1,1)
ax.plot(np.real(eigval),np.imag(eigval),linestyle='none',marker='o',label="FEM")
ax.plot(np.real(exact_eigval(100)),np.imag(exact_eigval(100)),linestyle='none',marker='x',color='red',label="Exact")
ax.set_xlabel(r"$\Re(\lambda)$")
ax.set_ylabel(r"$\Im(\lambda)$")
ax.set_xlim([0,np.max(np.real(eigval))])
# ax.set_xlim([0,30])
ax.set_title(f"{geometry} {boundary_condition}, {solver} (N={eigvec_r[0].size})")
ax.legend()

    # Paraview export of all eigenmodes
xdmfile = "laplace-eigenvalues.xdmf"
output = fe.XDMFFile(xdmfile)
output.parameters["flush_output"] = True
output.parameters["rewrite_function_mesh"] = False
output.parameters["functions_share_mesh"] = True
print(f"Export to {xdmfile}...")
for i in range(len(eigvec_r)):
    idx_u, idx_l = 0, 1
    eigvec_p = eigvec_r[i] / abs(eigvec_r[i]).max()[1]
    u_fun = multiphenics_utils.get_function(eigvec_p, idx_u,f"u_{eigval[i]}",W)
    output.write(u_fun, 0)
    if not boundary_DG:
        u_fun = multiphenics_utils.get_function(eigvec_p, idx_l,f"l_{eigval[i]}",W)
        output.write(u_fun, 0)