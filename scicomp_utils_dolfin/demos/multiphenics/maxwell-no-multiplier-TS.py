#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script solves Maxwell's equations:
    epsilon*dE/dt = curl(H)-j, mu*dH/dt = -curl(E),
with the boundary condition:
    pi_t(E) = z * gm_t(H) on Gamma,
where z>=0 is a positive scalar. z=0 yields a PEC condition. The boundary
condition is weakly enforced without using a boundary Lagrange multiplier.

If Omega is a 3D domain, then:
    - Both E and H are 3D vector fields
    - The tangential trace mappings are
    pi_t(E) = n x gm_t(E), gm_t(H) = H x n
where n is the outward unit normal.

If Omega is a 2D domain, then
    - E is scalar while H is a 2D vector field (transverse magnetic),
    - The tangential trace mappings are
    pi_t(E) = E*e_z, gm_t(H) = - (H,t)*e_z
where t = e_z x n is a tangent vector.
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
import os
DIR_MESH=os.path.join(os.path.dirname(os.path.abspath(__file__)),"mesh")

    # User input
geometry = "cube" # cube, rectangle
degree = 1 # polynomial space degree (>=1)
boundary_condition = "pec" # pec, ibc
impedance = 1 # impedance value (if ibc)
solver = "direct"
use_x0 = True # use a non-null boundary condition
use_j = False # use a non-null divergence-free current
    # --    
    # jit expressions
jit_dist = lambda xc,dim: '+'.join([f"pow(x[{i}]-{xc[i]},2)" for i in range(0,dim)])
jit_gaussian = lambda xc,sigma,dim: f"exp( -({jit_dist(xc,dim)})/pow({sigma},2) )"
jit_sin = lambda i: f"sin(pi*x[{i}]/{L[i]})"
    # Geometry definition
if geometry=="cube": # Cube, Full 3D system
    dim = 3
    geofile=os.path.join(DIR_MESH,"cube.geo")
    gmshfile=os.path.join(DIR_MESH,"cube.msh")
    Gamma_bnd_name = ['Gamma']
    L = [0.5, 1.4, 1.2]
    xc = np.array(L)/2
    param = {'Lx':L[0],'Ly':L[1],'Lz':L[2],'lc':1/10}
elif geometry=="rectangle": # Rectangle, TM 2D system
    dim = 2
    geofile=os.path.join(DIR_MESH,"Rectangle.geo")
    gmshfile=os.path.join(DIR_MESH,"Rectangle.msh")
    Gamma_bnd_name = ['Rectangle-Boundary-Top', 'Rectangle-Boundary-Bot', 
                      'Rectangle-Boundary-Left', 'Rectangle-Boundary-Right']
    L = [1, 2]
    xc = np.array(L)/2
    param = {'x_r':L[0],'y_t':L[1],'N_x': 100,'N_y': 100}

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

j_expr=['0' for i in range(0,dim)]
sigma = np.min(L)/2
if use_j and dim==3:
    j_expr = [f'exp(-( pow(x[1]-{xc[1]},2)+pow(x[2]-{xc[2]},2) )/pow({sigma},2))','0','0'] # current
if dim==2:
    j_expr = '0'
    if use_j:
        j_expr = jit_gaussian(xc, sigma, dim)

        # Function spaces
V_RT = lambda deg: fe.FunctionSpace(dmesh.mesh, "RT", deg)
V_DG_scalar = lambda deg: fe.FunctionSpace(dmesh.mesh, "DG", deg)
V_DG_vector = lambda deg: fe.VectorFunctionSpace(dmesh.mesh, "DG", deg)
V_CG_vector = lambda deg: fe.VectorFunctionSpace(dmesh.mesh, "CG", deg)
V_CG_scalar = lambda deg: fe.FunctionSpace(dmesh.mesh, "CG", deg)
V_curl = lambda deg: fe.FunctionSpace(dmesh.mesh, "N1curl", deg)

    # Tangential trace
normal = fe.FacetNormal(dmesh.mesh)
if dim==3:
    gm_t = lambda u: fe.cross(u,normal)
elif dim==2:
    import ufl
    t = ufl.operators.perp(normal)
    gm_t = lambda u: fe.dot(u,t)
    # Weak formulation
V_H = V_curl(degree)
if dim==3:
    V_E = V_DG_vector(degree-1)
    # V_E = V_RT(degree) # identical results
    V_j = V_CG_vector(degree)
elif dim==2:
    V_E = V_DG_scalar(degree-1)
    V_j = V_CG_scalar(degree)

    # (E,H) in [L^2]^3 x H(curl) if Omega is 3D
    # (E,H) in L^2 x H(curl)     if Omega is 2D
W = mpfe.BlockFunctionSpace([V_E, V_H], restrict=[None, None])
    # (E,H)
fun_test = mpfe.block_split(mpfe.BlockTestFunction(W))
    # (phi_E,phi_H)
fun_trial = mpfe.block_split(mpfe.BlockTrialFunction(W))

def Maxwell_jacobian(fun_trial,fun_test,dx):
    """ Jacobian matrix for residual. """
    (E, H) = fun_trial; (phi_E, phi_H) = fun_test
    J_yd, J_y = list(), list()
    J_yd.append([epsilon*fe.inner(E, phi_E)*dx])
    J_yd.append([mu*fe.inner(H, phi_H)*dx])    
    J_y.append([-fe.inner(fe.curl(H), phi_E)*dmesh.dx, 0])
    J_y.append([+fe.inner(E, fe.curl(phi_H))*dmesh.dx, +z*fe.inner(gm_t(H),gm_t(phi_H))*dmesh.ds])
    return (J_yd,J_y)

def Maxwell_residual_src(j,t,fun_test,dx):
    """ External forcing. """
    j.t = t
    (phi_E,phi_H) = fun_test
    return [+fe.inner(j,phi_E)*dx]

def Maxwell_compute_norm(t,y,W,dx):
    """ L2 norm. """
    H = np.zeros((len(t),))
    for i in range(len(t)):
        fun_E = multiphenics_utils.get_function(y[i], 0,"E",W)
        fun_H = multiphenics_utils.get_function(y[i], 1,"H",W)
        H[i] = fe.assemble(0.5*epsilon*fe.inner(fun_E,fun_E)*dx+0.5*mu*fe.inner(fun_H,fun_H)*dx)
    return H

    # Build linear PDAE
j = fe.Expression(j_expr, element=V_j.ufl_element(),t=0)
residual_src = lambda t : Maxwell_residual_src(j, t, fun_test, dmesh.dx)
jacobian = lambda : Maxwell_jacobian(fun_trial, fun_test, dmesh.dx)
name = f"Maxwell equation - No multiplier - {geometry}/{boundary_condition}"
dae = multiphenics_utils.PDAE_linear(residual_src,jacobian,W.dim(),name)
    # Time parameters
tf = int(2e0)
dt = 0.01
dt_export = 0.1
    # Initial condition
x0 = dae.get_vector()
    # PEC initial condition (not divergence-free)
sigma = np.min(L)/2
if use_x0:
    if dim==3:
        E0_expr = [jit_gaussian(xc,sigma,dim)+f"*{jit_sin(1)}*{jit_sin(2)}",
                   jit_gaussian(xc,sigma,dim)+f"*{jit_sin(0)}*{jit_sin(2)}",
                   jit_gaussian(xc,sigma,dim)+f"*{jit_sin(0)}*{jit_sin(1)}"]
        H0_expr = [jit_gaussian(xc,sigma,dim)+f"*{jit_sin(0)}",
                   jit_gaussian(xc,sigma,dim)+f"*{jit_sin(1)}",
                   jit_gaussian(xc,sigma,dim)+f"*{jit_sin(2)}"]
    elif dim==2:
        E0_expr = jit_gaussian(xc,sigma,dim)+f"*{jit_sin(0)}*{jit_sin(1)}"
        H0_expr = [jit_gaussian(xc,sigma,dim)+f"*{jit_sin(0)}",
                   jit_gaussian(xc,sigma,dim)+f"*{jit_sin(1)}"]       
    E0_fun = fe.project(fe.Expression(E0_expr, degree=3,t=0),V_E)
    H0_fun = fe.project(fe.Expression(H0_expr, degree=3,t=0),V_H)
    E0_vec, H0_vec = E0_fun.vector().vec(), H0_fun.vector().vec()
    idx_E0 = np.r_[0:E0_vec.size]
    idx_H0 = (1 + idx_E0[-1])  + np.r_[0:H0_vec.size]
    x0.setValues(np.array(idx_E0,dtype="int32"),E0_vec)
    x0.setValues(np.array(idx_H0,dtype="int32"),H0_vec)
    # Solver options
OptDB = PETSc_utils.get_cleared_options_db()
PETSc.Sys.pushErrorHandler("python")
# OptDB["ts_type"] = "beuler"
OptDB["ts_type"] = "cn"
# OptDB["ts_type"] = "bdf"
# OptDB["bdf_order"] = 2
# OptDB["ts_adapt_dt_max"] = dt
OptDB["ksp_type"] = "preonly"
OptDB["pc_type"] = "lu"
OptDB["pc_factor_mat_solver_type"] = "mumps"
# OptDB["pc_type"] = "hypre"
Integration_params = {'cinit': False,'cinit_dt':1e-3,'dt_export':dt_export,'cinit_nstep':int(2)}
PETSc_utils.TS_integration_dae(dae,x0,dt,tf,**Integration_params)
tp = dae.history['t']
yp = dae.history['y']

# Plot Hamiltonian
H = Maxwell_compute_norm(tp,yp,W,dmesh.dx)
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(tp,H/np.max(H),marker="o")
ax.set_title(dae.name)
ax.set_xlabel("t")
ax.set_ylabel("H(t)")
ax.set_ylim([0,1.1])

# Post-processing with paraview export
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

div_D, div_B = np.zeros(len(tp)), np.zeros(len(tp))
xdmfile = "maxwell-no-multipliers-TS.xdmf"
output = fe.XDMFFile(xdmfile)
output.parameters["flush_output"] = True
output.parameters["rewrite_function_mesh"] = False
output.parameters["functions_share_mesh"] = True
print(f"Export to {xdmfile}...")
for i in range(len(tp)):
    print(f"t={tp[i]:1.3e}.")
    idx_E, idx_H = 0, 1
    vec_p = yp[i]
    u_fun = multiphenics_utils.get_function(vec_p, idx_E,"E",W)
    output.write(u_fun, tp[i])
    if dim==3: # E is a scalar otherwise
        D_fun = fe.project(epsilon*u_fun, V_E) # Compute D
        (avg_div,avg_jump) = compute_L2avg_div_jump(fe.project(D_fun,V_RT(degree)),volume,dmesh)
        div_D[i] = avg_div
    u_fun = multiphenics_utils.get_function(vec_p, idx_H,"H",W)
    output.write(u_fun, tp[i])
    B_fun = fe.project(mu*u_fun, V_H) # Compute B
    (avg_div,avg_jump) = compute_L2avg_div_jump(fe.project(B_fun,V_RT(degree)),volume,dmesh)
    div_B[i] = avg_div

# Plot divergence
fig = plt.figure()
ax = fig.add_subplot()
ax.plot(tp,(div_B-div_B[0])/div_B[0],label="proj(B,RT)")
if dim==3:
    ax.plot(tp,(div_D-div_D[0])/div_D[0],label="proj(D,RT)")
ax.set_xlabel("t")
ax.set_ylabel("Relative variation of ||div||_2")
ax.set_ylim([-0.2,0.2])
ax.legend()
ax.set_title(dae.name)