#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script demonstrates the computation of eigenvalues of Maxwell's
equations. The eigenvalue problem is: Find (lambda,E,H) such that
    epsilon*lambda*E = curl(H), mu*lambda*H = -curl(E),
with the boundary condition:
    pi_t(E) = z * gm_t(H) on Gamma,
where z>=0 is a positive scalar. z=0 yields a PEC condition. The boundary
condition is weakly enforced using a boundary Lagrange multiplier.

If Omega is a 3D domain, then:
    - Both E and H are 3D vector fields,
    - The tangential trace mappings are
    pi_t(E) = n x gm_t(E), gm_t(H) = H x n
where n is the outward unit normal.
    - The Lagrange multiplier is a tangential vector filed, which is challenging
to compute accurately.

If Omega is a 2D domain:
    - E is a 2D vector field while H is a scalar field (transverse electric),
    - The tangential trace mappings are
    pi_t(E) = (E,t)*t, gm_t(H) = H*t, 
where t = e_z x n is a tangent vector.
    - The Lagrange multiplier is a scalar field, which can be computed
accurately.
"""

import numpy as np
import matplotlib.pyplot as plt
import fenics as fe
from scientific_computing_utils import gmsh_utils
from scientific_computing_utils import fenics_utils
from scientific_computing_utils import meshio_utils
import multiphenics as mpfe
from scientific_computing_utils import multiphenics_utils
from petsc4py import PETSc
fe.SubSystemsManager.init_petsc()
PETSc.Sys.pushErrorHandler("python")
from scientific_computing_utils import PETSc_utils
from slepc4py import SLEPc
from scientific_computing_utils import SLEPc_utils
from scientific_computing_utils import PDE_exact
import os
DIR_MESH=os.path.join(os.path.dirname(os.path.abspath(__file__)),"mesh")

    # User input
geometry = "rectangle" # ball, cube, rectangle, disc
degree = 1 # polynomial space degree (>=1)
boundary_condition = "pec" # pec, ibc
impedance = 1 # impedance value (if ibc)
solver = "direct" # iterative, direct

    # Geometry definition
if geometry=="ball": # Ball, Full 3D system
    dim = 3
    geofile=os.path.join(DIR_MESH,"sphere.geo")
    gmshfile=os.path.join(DIR_MESH,"sphere.msh")
    Gamma_bnd_name = ['Gamma']
    R = 1.4
    param = {'R':R,'lc':1/8}
    exact_eigval = lambda N: PDE_exact.Maxwell_spectrum.ball(R,boundary_condition,
                   eigval_min=0,eigval_max=10,eigval_N=N,eigval_maxorder=4)
elif geometry=="cube": # Cube, Full 3D system
    dim = 3
    geofile=os.path.join(DIR_MESH,"cube.geo")
    gmshfile=os.path.join(DIR_MESH,"cube.msh")
    Gamma_bnd_name = ['Gamma']
    Lx, Ly, Lz = 1, 2, 0.7
    param = {'Lx':Lx,'Ly':Ly,'Lz':Lz,'lc':Lx/10}
    exact_eigval = lambda N: PDE_exact.Maxwell_spectrum.cube(Lx,Ly,Lz,N,boundary_condition)
elif geometry=="rectangle": # Rectangle, TM 2D system
    dim = 2
    geofile=os.path.join(DIR_MESH,"Rectangle.geo")
    gmshfile=os.path.join(DIR_MESH,"Rectangle.msh")
    Gamma_bnd_name = ['Rectangle-Boundary-Top', 'Rectangle-Boundary-Bot', 
                      'Rectangle-Boundary-Left', 'Rectangle-Boundary-Right']
    Lx, Ly = 1, 2
    param = {'x_r':Lx,'y_t':Ly,'N_x': 150,'N_y': 150}
    exact_eigval = lambda N: PDE_exact.Maxwell_spectrum.rectangle_TE(Lx,Ly,N,boundary_condition)
elif geometry=="disc":  # Disc, TM 2D system
    dim = 2
    geofile=os.path.join(DIR_MESH,"Disc.geo")
    gmshfile=os.path.join(DIR_MESH,"Disc.msh")
    Gamma_bnd_name = ['Gamma-R']
    R = 1.3
    param = {'R':R, 'lc':R/50}
    exact_eigval = lambda N: PDE_exact.Maxwell_spectrum.disc_TE(R,boundary_condition,
                      eigval_min=0,eigval_max=100,eigval_N=N,eigval_maxorder=10)

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

    # Physical parameters
epsilon = fe.Expression('1', degree=0) # dielectric permittivity
mu = fe.Expression('1', degree=0) # magnetic permeability
# epsilon = fe.Expression('1+x[0]', degree=1) # dielectric permittivity
# mu = fe.Expression('1+x[0]', degree=1) # magnetic permeability
if boundary_condition=="ibc":
    z = fe.Constant(impedance)
elif boundary_condition=="pec":
    z = fe.Constant(0)

    # Function spaces
V_curl = lambda deg: fe.FunctionSpace(dmesh.mesh, "N1curl", deg)
V_RT = lambda deg: fe.FunctionSpace(dmesh.mesh, "RT", deg)
V_CG_vector = lambda deg: fe.VectorFunctionSpace(dmesh.mesh, "CG", deg)
V_DG_scalar = lambda deg: fe.FunctionSpace(dmesh.mesh, "DG", deg)
V_DG_vector = lambda deg: fe.VectorFunctionSpace(dmesh.mesh, "DG", deg)
V_DG_scalar_bnd = lambda deg: fe.FunctionSpace(dmesh.mesh,'DGT', deg)
V_DG_vector_bnd = lambda deg: fe.VectorFunctionSpace(dmesh.mesh,'DGT', deg)
    # Tangential trace
normal = fe.FacetNormal(dmesh.mesh)
if dim==3:
    gm_t = lambda u: fe.cross(u,normal)
    pi_t = lambda u: fe.cross(normal,gm_t(u))
elif dim==2:
    import ufl
    t = ufl.operators.perp(normal)
        # Define as scalar-valued operator
    gm_t = lambda u: u
    pi_t = lambda u: fe.dot(u,t)
    
    # Weak formulation
V_E = V_curl(degree)
if dim==3:
    V_H = V_DG_vector(degree-1)
    # V_H = V_RT(degree) # identical results
        # theoretically correct, but cannot be solved
    # V_lag = V_DG_vector_bnd(degree-1)
        # can be solved, with some pollution
    V_lag = V_CG_vector(degree)
elif dim==2:
    V_H = V_DG_scalar(degree-1)
    V_lag = V_DG_scalar_bnd(degree-1)

    # (E,H,l) in H(curl) x [L^2]^N x [L^2(Gamma)]^3  if Omega is 3D
    # (E,H,l) in H(curl) x L^2 x L^2(Gamma)          if Omega is 2D
    # l = -gm_t(H) is the boundary Lagrange multiplier.
boundary_restriction =  multiphenics_utils.build_MeshRestriction_from_tags(dmesh.mesh,
                                                dmesh.boundaries,
                                                Gamma_bnd)
W = mpfe.BlockFunctionSpace([V_E, V_H,V_lag], restrict=[None, None,boundary_restriction])
ul = mpfe.BlockTrialFunction(W)
(E, H, l) = mpfe.block_split(ul)
vm = mpfe.BlockTestFunction(W)
(phi_E, phi_H, phi_l) = mpfe.block_split(vm)

a, b = list(), list()
b.append([epsilon*fe.inner(E, phi_E)*dmesh.dx, 0])
b.append([mu*fe.inner(H, phi_H)*dmesh.dx, 0])
b.append([0])
a.append([fe.inner(H, fe.curl(phi_E))*dmesh.dx, fe.inner(l,pi_t(phi_E))*dmesh.ds])
a.append([-fe.inner(fe.curl(E), phi_H)*dmesh.dx, 0])
if dim==3: # enforce nullity of cdot(l,n) (difficult)
    a.append([-z*fe.inner(gm_t(H),phi_l)*dmesh.ds,fe.inner(fe.dot(l,normal),fe.dot(phi_l,normal))*dmesh.ds,fe.inner(pi_t(E),phi_l)*dmesh.ds])
elif dim==2:
    a.append([-z*fe.inner(gm_t(H),phi_l)*dmesh.ds,fe.inner(pi_t(E),phi_l)*dmesh.ds])

    # The generalized eigenvalue problem is assembled as
    #   a(U,V)=lambda*b(U,V).
A, B = mpfe.block_assemble(a,keep_diagonal=True), mpfe.block_assemble(b,keep_diagonal=True)
A_petsc = A.mat(); B_petsc = B.mat()

import time
start = time.time()
OptDB = PETSc_utils.get_cleared_options_db()
OptDB = PETSc.Options()
if solver=="iterative":
    OptDB["st_ksp_type"] = "ibcgs" # fgmres, gmres, gcr, ibcgs
    OptDB["st_pc_type"] = "bjacobi" # ilu, bjacobi, icc
    OptDB["st_ksp_rtol"] = 1e-5
else:     # Direct solver (MUMPS)
    OptDB["st_ksp_type"] = "preonly"
    OptDB["st_pc_type"] = "lu"
    OptDB["st_pc_factor_mat_solver_type"] = "umfpack" # mumps, umfpack
SLEPc_params = {'nev': 100,
              'target': 10, 'shift': 10,
              'problem_type': SLEPc.EPS.ProblemType.GNHEP, # GNHEP, PGNHEP
              'solver': SLEPc.EPS.Type.KRYLOVSCHUR,
              'tol': 1e-8,
              'max_it': 100}
EPS = SLEPc_utils.solve_GEP_shiftinvert(A_petsc,B_petsc,**SLEPc_params)
(eigval,eigvec_r,eigvec_i) = SLEPc_utils.EPS_get_spectrum(EPS)
end = time.time()
print(f"SLEPc elapsed time: {(end - start)/60:1.2e}min")

    # Plot eigenvalues
f = plt.figure()
ax=f.add_subplot(1,1,1)
ax.plot(np.real(eigval),np.imag(eigval),linestyle='none',marker='o',label="FEM")
ax.plot(np.real(exact_eigval(100)),np.imag(exact_eigval(100)),linestyle='none',marker='x',color='red',label="Exact")
ax.set_xlabel(r"$\Re(\lambda)$")
ax.set_ylabel(r"$\Im(\lambda)$")
ax.set_xlim([-10,1])
ax.set_ylim([0,np.max(np.imag(eigval))])
ax.set_ylim([0,10])
ax.set_title(f"{geometry} {boundary_condition} {solver} (N={eigvec_r[0].size})")
ax.legend()

    #%% Post-processing with paraview export
def compute_L2avg_div_jump(u,domain_vol,dmesh):
    """ Compute L^2 average of div(u) and edge normal jump.
        u: dolfin.function.function.Function
        domain_vol: float
        dmesh: fenics_utils.DolfinMesh """
    avg_div = fe.inner(fe.div(u),fe.div(u))*dmesh.dx
    avg_div = np.sqrt(fe.assemble(avg_div)) / np.sqrt(domain_vol)
    n, fa = fe.FacetNormal(dmesh.mesh), fe.FacetArea(dmesh.mesh)
    avg_jump = (1/fa('-'))*fe.inner(fe.jump(u, n),fe.jump(u, n))*dmesh.dS
    avg_jump = np.sqrt(fe.assemble(avg_jump))
    return (avg_div,avg_jump)

volume = fe.assemble(fe.Expression('1', degree=0)*dmesh.dx)
xdmfile = "maxwell-eigenvalues.xdmf"
output = fe.XDMFFile(xdmfile)
output.parameters["flush_output"] = True
output.parameters["rewrite_function_mesh"] = False
output.parameters["functions_share_mesh"] = True
print(f"Export to {xdmfile}...")
for i in range(len(eigvec_r)):
    print(f"Eigenvalue {eigval[i]:1.3e}.")
    idx_E, idx_H, idx_L = 0, 1, 2
    eigvec_p = eigvec_r[i] / abs(eigvec_r[i]).max()[1]
    u_fun = multiphenics_utils.get_function(eigvec_p, idx_L,f"L_{eigval[i]}",W)
    output.write(u_fun, 0)
    u_fun = multiphenics_utils.get_function(eigvec_p, idx_E,f"E_{eigval[i]}",W)
    output.write(u_fun, 0)
    D_fun = fe.project(epsilon*u_fun, V_E)
    (avg_div,avg_jump) = compute_L2avg_div_jump(D_fun,volume,dmesh)
    print(f"\t<div(D)>={avg_div:1.2e} <jump(D,n)>={avg_jump:1.2e}")
    (avg_div,avg_jump) = compute_L2avg_div_jump(fe.project(D_fun,V_RT(degree)),volume,dmesh)
    print(f"\t<div(proj(D))>={avg_div:1.2e} <jump(proj(D),n)>={avg_jump:1.2e}")
    u_fun = multiphenics_utils.get_function(eigvec_p, idx_H,f"H_{eigval[i]}",W)
    output.write(u_fun, 0)
    if dim==3: # H is a scalar otherwise
        B_fun = fe.project(mu*u_fun, V_H)
        (avg_div,avg_jump) = compute_L2avg_div_jump(B_fun,volume,dmesh)
        print(f"\t<div(B)>={avg_div:1.2e} <jump(B,n)>={avg_jump:1.2e}")
        (avg_div,avg_jump) = compute_L2avg_div_jump(fe.project(B_fun,V_RT(degree)),volume,dmesh)
        print(f"\t<div(proj(B))>={avg_div:1.2e} <jump(proj(B),n)>={avg_jump:1.2e}")
