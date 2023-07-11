#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script solves Maxwell's equations:
    epsilon*dE/dt = curl(H)-j, mu*dH/dt = -curl(E),
with the boundary condition:
    pi_t(E) = z * gm_t(H) on Gamma_wall,
    pi_t(E) = z_0 * gm_t(H) on Gamma_out (outlet),
    pi_t(E) = z_0 * gm_t(H) + f(t) on Gamma_in (inlet),
where all impedance values are positive scalar. On Gamma_wall, z=0 yields a PEC
condition. The Silver-Muller boundary condition is obtained for
    z_0 = sqrt(mu/epsilon).
These boundary conditions are weakly enforced using a boundary Lagrange 
multiplier.

This script is restricted to Omega being a 2D rectangular domain:
    - E is a 2D vector field while H is a scalar field (transverse electric),
    - The tangential trace mappings are
    pi_t(E) = (E,t)*t, gm_t(H) = H*t,
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
degree = 1 # polynomial space degree (>=1)
wall_impedance = 0  # wall impedance value
solver = "direct"
use_x0 = False # use a non-null boundary condition
use_j = False # use a non-null divergence-free current
    # --    
    # jit expressions
jit_dist = lambda xc,dim: '+'.join([f"pow(x[{i}]-{xc[i]},2)" for i in range(0,dim)])
jit_gaussian = lambda xc,sigma,dim: f"exp( -({jit_dist(xc,dim)})/pow({sigma},2) )"
jit_sin = lambda i: f"sin(pi*x[{i}]/{L[i]})"
    # Geometry definition
dim = 2
geofile=os.path.join(DIR_MESH,"Rectangle.geo")
gmshfile=os.path.join(DIR_MESH,"Rectangle.msh")
Gamma_wall_name = ['Rectangle-Boundary-Bot', 'Rectangle-Boundary-Top']
Gamma_in_name = ['Rectangle-Boundary-Left']
Gamma_out_name = ['Rectangle-Boundary-Right']
L = [7, 0.1]
xc = np.array(L)/2
param = {'x_r':L[0],'y_t':L[1],'N_x': 200,'N_y': 40}
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
get_tag = lambda name_list: [tag for name in name_list for tag in meshtag[dim-2][name]]
Gamma_wall = get_tag(Gamma_wall_name)
Gamma_in = get_tag(Gamma_in_name)
Gamma_out = get_tag(Gamma_out_name)
Gamma = Gamma_wall+Gamma_in+Gamma_out
    # Physical parameters
epsilon = fe.Expression('1', degree=0) # dielectric permittivity
mu = fe.Expression('1', degree=0) # magnetic permeability
z_sm = fe.Constant(1) # inlet/outlet impedance (Silver-Muller)
z_wall = fe.Constant(wall_impedance) # wall impedance
f_bnd_expr = '0' # boundary inlet source
tc, sigma = 3, 1
f_bnd_expr = f'exp(-pow((t-{tc})/{sigma},2))' # compactly-suported source
j_expr=['0' for i in range(0,dim)]
sigma = np.min(L)/2
if use_j:
    j_expr = [f'exp(-( pow(x[1]-{xc[1]},2))/pow({sigma},2))','0'] # current

        # Function spaces
V_RT = lambda deg: fe.FunctionSpace(dmesh.mesh, "RT", deg)
V_DG_scalar = lambda deg: fe.FunctionSpace(dmesh.mesh, "DG", deg)
V_DG_vector = lambda deg: fe.VectorFunctionSpace(dmesh.mesh, "DG", deg)
V_CG_vector = lambda deg: fe.VectorFunctionSpace(dmesh.mesh, "CG", deg)
V_CG_scalar = lambda deg: fe.FunctionSpace(dmesh.mesh, "CG", deg)
V_curl = lambda deg: fe.FunctionSpace(dmesh.mesh, "N1curl", deg)
V_DG_scalar_bnd = lambda deg: fe.FunctionSpace(dmesh.mesh,'DGT', deg)
    # Tangential trace
normal = fe.FacetNormal(dmesh.mesh)
import ufl
t = ufl.operators.perp(normal)
    # Define as scalar-valued operator
gm_t = lambda u: u
pi_t = lambda u: fe.dot(u,t)
    # Weak formulation
V_E = V_curl(degree)
V_H = V_DG_scalar(degree-1)
V_lag = V_DG_scalar_bnd(degree-1) 
V_j = V_CG_vector(degree)
V_f = V_CG_scalar(degree)

    # (E,H,l) in H(curl) x L^2 x L^2(Gamma)          since Omega is 2D
    # l = -gm_t(H) is the boundary Lagrange multiplier.
boundary_restriction =  multiphenics_utils.build_MeshRestriction_from_tags(dmesh.mesh,
                                                dmesh.boundaries,
                                                Gamma)
W = mpfe.BlockFunctionSpace([V_E, V_H,V_lag], restrict=[None, None,boundary_restriction])
    # (E,H,l)
fun_test = mpfe.block_split(mpfe.BlockTestFunction(W))
    # (phi_E,phi_H,phi_l)
fun_trial = mpfe.block_split(mpfe.BlockTrialFunction(W))

def Maxwell_jacobian(fun_trial,fun_test,dx,ds):
    """ Jacobian matrix for residual. """
    (E, H,l) = fun_trial; (phi_E, phi_H,phi_l) = fun_test
    J_yd, J_y = list(), list()
    J_yd.append([epsilon*fe.inner(E, phi_E)*dx])
    J_yd.append([mu*fe.inner(H, phi_H)*dx])    
    J_y.append([-fe.inner(H, fe.curl(phi_E))*dx, -fe.inner(l,pi_t(phi_E))*ds])
    J_y.append([fe.inner(fe.curl(E), phi_H)*dx, 0])
    J_y.append([])
    for tag in Gamma_wall:
        J_y[-1].append(-z_wall*fe.inner(gm_t(H),phi_l)*ds(tag))
        J_y[-1].append(fe.inner(pi_t(E),phi_l)*ds(tag))
    for tag in Gamma_in:
        J_y[-1].append(-z_sm*fe.inner(gm_t(H),phi_l)*ds(tag))
        J_y[-1].append(fe.inner(pi_t(E),phi_l)*ds(tag))
    for tag in Gamma_out:
        J_y[-1].append(-z_sm*fe.inner(gm_t(H),phi_l)*ds(tag))
        J_y[-1].append(fe.inner(pi_t(E),phi_l)*ds(tag))
    return (J_yd,J_y)

def Maxwell_residual_src(j,f_bnd,t,fun_test,dx,ds_in):
    """ External forcing. """
    j.t =t; f_bnd.t = t 
    (phi_E,phi_H,phi_l) = fun_test
    return [+fe.inner(j,phi_E)*dx,fe.inner(f_bnd,phi_l)*ds_in]

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
f_bnd = fe.Expression(f_bnd_expr, element=V_f.ufl_element(),t=0)
residual_src = lambda t : Maxwell_residual_src(j, f_bnd, t, fun_test, dmesh.dx,dmesh.ds(Gamma_in[0]))
jacobian = lambda : Maxwell_jacobian(fun_trial, fun_test, dmesh.dx,dmesh.ds)
name = f"Maxwell equation - With multiplier - waveguide/z_wall={wall_impedance}"
dae = multiphenics_utils.PDAE_linear(residual_src,jacobian,W.dim(),name)
    # Time parameters
tf = int(14)
dt = 0.1
dt_export = dt
    # Initial condition
x0 = dae.get_vector()
    # PEC initial condition (not divergence-free)
if use_x0:
    sigma = np.min(L)/2
    E0_expr = [jit_gaussian(xc,sigma,dim)+f"*{jit_sin(1)}",
               jit_gaussian(xc,sigma,dim)+f"*{jit_sin(0)}"]
    H0_expr = jit_gaussian(xc,sigma,dim)
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
OptDB["ksp_type"] = "preonly"
OptDB["pc_type"] = "lu"
OptDB["pc_factor_mat_solver_type"] = "mumps"
# OptDB["pc_type"] = "hypre"
Integration_params = {'cinit': False,'cinit_dt':1e-3,'dt_export':dt_export,'cinit_nstep':int(1)}
PETSc_utils.TS_integration_dae(dae,x0,dt,tf,**Integration_params)
tp = dae.history['t']
yp = dae.history['y']

# Plot Hamiltonian
H = Maxwell_compute_norm(tp,yp,W,dmesh.dx)
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(tp,H/np.max(H),marker="o")
# ax.plot(tp,H,marker="o")
ax.set_title(dae.name)
ax.set_xlabel("t")
ax.set_ylabel("H(t)")
ax.set_ylim([0,1]) 

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
xdmfile = "maxwell-with-multipliers-TS.xdmf"
output = fe.XDMFFile(xdmfile)
output.parameters["flush_output"] = True
output.parameters["rewrite_function_mesh"] = False
output.parameters["functions_share_mesh"] = True
print(f"Export to {xdmfile}...")
for i in range(len(tp)):
    print(f"t={tp[i]:1.3e}.")
    idx_E, idx_H = 0, 1
    eigvec_p = yp[i]
    u_fun = multiphenics_utils.get_function(eigvec_p, idx_E,"E",W)
    output.write(u_fun, tp[i])
    D_fun = fe.project(epsilon*u_fun, V_E) # Compute D
    (avg_div,avg_jump) = compute_L2avg_div_jump(fe.project(D_fun,V_RT(degree)),volume,dmesh)
    div_D[i] = avg_div    
    u_fun = multiphenics_utils.get_function(eigvec_p, idx_H,"H",W)
    output.write(u_fun, tp[i])
    if dim==3: # H is a scalar otherwise
        B_fun = fe.project(mu*u_fun, V_H) # Compute B
        (avg_div,avg_jump) = compute_L2avg_div_jump(fe.project(B_fun,V_RT(degree)),volume,dmesh)
        div_B[i] = avg_div

# Plot divergence
fig = plt.figure()
ax = fig.add_subplot()
ax.plot(tp,(div_D-div_D[0]),label="proj(D,RT)")
if dim==3:
    ax.plot(tp,(div_B-div_B[1])/div_B[1],label="proj(B,RT)")
ax.set_xlabel("t")
ax.set_ylabel("Variation of ||div||_2")
ax.legend()
ax.set_title(dae.name)