#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semilinear heat equation with Dirichlet boundary condition:

    du/dt = Delta(u) - h(u) + f on Omega, 
    u=u_D on Gamma,

where h(s)s >= 0 for stability.

Validation is carried out using the method of manufactured solutions. The 
computed solution at t>>1 is compared to an arbitrary function u_ex, which
is the exact stationary solution solution when
    u_D = u_ex, f = -Delta(g) + h(g).

The problem is setup with multiphenics_utils.PDAE_linear (case h=0) or
multiphenics_utils.PDAE_nonlinear (case h!=0).

"""

import numpy as np
import matplotlib.pyplot as plt

import gmsh_utils
import meshio_utils

import fenics as fe
import fenics_utils
import multiphenics as mpfe
import multiphenics_utils

fe.SubSystemsManager.init_petsc()
from petsc4py import PETSc
from mpi4py import MPI
import PETSc_utils
import os
DIR_MESH=os.path.join(os.path.dirname(os.path.abspath(__file__)),"mesh")

#% Generate mesh using gmsh
geofile=os.path.join(DIR_MESH,"Disc.geo")
gmshfile=os.path.join(DIR_MESH,"Disc.msh")
dim = 2
gmsh_utils.generate_mesh_cli(geofile,gmshfile,dim,refinement=1,log=1,\
                             parameters={'R':1,'l_c':1/15},order=1,
                             gmshformat=2,binary=False)
gmsh_utils.print_summary(gmshfile)
#% Load mesh: XDMF format
meshio_utils.print_diagnostic_gmsh(gmshfile)
xdmfiles = meshio_utils.convert_gmsh_to_XDMF(gmshfile,prune_z=True)
dmesh = fenics_utils.DolfinMesh.init_from_xdmf(xdmfiles['triangle'],xdmfiles['line'])
meshtag = meshio_utils.read_phys_gmsh(xdmfiles['gmsh_physical_entities'])
Dirichlet_bnd_name = 'Gamma-R'
#%% Problem definition
    # Manufactured solution: f = -Delta(g) + h(g)
damping_expr = '10'
        # third-order pol
g_expr = 'x[0]*x[0]+4*x[1]*x[1]*x[1]'
dg_expr = '2+24*x[1]'
        # second-order pol
g_expr = 'x[0]*x[0]+4*x[1]*x[1]'
dg_expr = '10'
        # h(u) = damping*u^2
f_expr = '-('+dg_expr+')+'+damping_expr+'*('+g_expr+')*('+g_expr+')'
u_ex_expr = g_expr
u_D_expr = u_ex_expr
    # Boundaries
Dirichlet_bnd = meshtag[dim-2][Dirichlet_bnd_name][0]
Dirichlet_bnd_restriction =  multiphenics_utils.build_MeshRestriction_from_tags(
                                                    dmesh.mesh,
                                                    dmesh.boundaries,
                                                    [Dirichlet_bnd])
ds_D = dmesh.ds(Dirichlet_bnd)
    # Function spaces
order = 2
V_u = fe.FunctionSpace(dmesh.mesh, 'P', order)
V_l = fe.FunctionSpace(dmesh.mesh, 'P', order)
W = mpfe.BlockFunctionSpace([V_u, V_l], 
                            restrict=[None,Dirichlet_bnd_restriction])
    # Test/Trial functions
fun_test = mpfe.block_split(mpfe.BlockTestFunction(W))
fun_trial = mpfe.block_split(mpfe.BlockTrialFunction(W))
    # Lagrange Multipliers
idx_Lagrange_multipliers= W.block_dofmap().block_owned_dofs__global_numbering(1)
    
def Wave_semilinear_jacobian(fun_trial,fun_test,dx,ds_D):
    """ Jacobian matrix for linear part of residual. """
    (u, l) = fun_trial; (v_u, v_l) = fun_test
    J_yd = [[u*v_u*dx]]
    J_y = [
        [fe.inner(fe.grad(u), fe.grad(v_u))*dx, -l*v_u*ds_D],
        [u*v_l*ds_D]]
    return (J_yd,J_y)

def Wave_semilinear_residual_src(u_D_src,f_src,t,fun_test,dx,ds_D):
    """ External forcing. """
    u_D_src.t = t; f_src.t = t
    (v_u,v_l) = fun_test
    return [-f_src*v_u*dx,
            -u_D_src*v_l*ds_D]

def Wave_semilinear_residual_nl(y,W,fun_test,dx,damping):
    """ Non-linear part of residual, h(u)=damping*u^2. """
    u = multiphenics_utils.get_function(y, 0,"u",W)
    v_u = fun_test[0]
    return [damping*u*u*v_u*dx]

def Wave_semilinear_jacobian_nl(y,W,fun_trial,fun_test,dx,damping):
    """ Jacobian matrix associated with non-linear part of residual. """
    u = multiphenics_utils.get_function(y, 0,"u",W)
    z_u = fun_trial[0]
    v_u = fun_test[0]
    J_nl = [[2*damping*u*z_u*v_u*dx]]
    return J_nl

def Wave_semilinear_compute_norm(t,y,W,dx):
    """ L2 norm.
    
    :param t: list(float)
    :param y: list(PETSc.Vec)
    :param W: mpfe BlockFunctionSpace
    :param dx: measure
    """
    H = np.zeros((len(t),))
    for i in range(len(t)):
        u = multiphenics_utils.get_function(y[i], 0,"u",W)
        H[i] = fe.assemble(u*u*dx)
    return H

    # Expressions
u_D = fe.Expression(u_D_expr, element=V_u.ufl_element(),t=0)
f = fe.Expression(f_expr, element=V_u.ufl_element(),t=0)
damping = fe.Expression(damping_expr, element=V_u.ufl_element(),t=0)
u_ex = fe.Expression(u_ex_expr, t=0,degree=2)
    # Build linear PDAE
residual_time_dependent = lambda  t : Wave_semilinear_residual_src(u_D,f,t,fun_test,dmesh.dx,ds_D)
jacobian = lambda : Wave_semilinear_jacobian(fun_trial,fun_test,dmesh.dx,ds_D)
name = f"Heat equation - Linear"
dae = multiphenics_utils.PDAE_linear(residual_time_dependent,jacobian,
            W.dim(),name,idx_alg=idx_Lagrange_multipliers)
    # Build nonlinear PDAE
residual_nl = lambda y : Wave_semilinear_residual_nl(y,W,fun_test,dmesh.dx,damping)
jacobian_nl = lambda y : Wave_semilinear_jacobian_nl(y,W,fun_trial,fun_test,dmesh.dx,damping)
name = f"Heat equation - Nonlinear"
dae = multiphenics_utils.PDAE_nonlinear(residual_time_dependent,jacobian,
                                      residual_nl, jacobian_nl,
                                    W.dim(),name,idx_alg=idx_Lagrange_multipliers,
                                    J_nl_skip=1)
#%% Time integration
    # Time parameters
tf = 2
dt = 0.1
x0 = dae.get_vector()
    # Solver options
OptDB = PETSc_utils.get_cleared_options_db()
PETSc.Sys.pushErrorHandler("python")
OptDB["ts_equation_type"] = PETSc.TS.EquationType.DAE_IMPLICIT_INDEX2
    # Numerical scheme
# OptDB["ts_type"] = "beuler" # beuler, cn
        # BDF
OptDB["ts_type"] = "bdf"
OptDB["bdf_order"] = 4
        # ARKIMEX schemes
# OptDB["ts_type"] = PETSc.TS.Type.ARKIMEX
# OptDB["ts_type_arkimex"] = PETSc.TS.ARKIMEXType.ARKIMEX3
        # ROSW Linear implicit schemes
# OptDB["ts_type"] = PETSc.TS.Type.ROSW
# OptDB["ts_type_rosw"] = "ra34pw2"
    # Linear solver: GMRES
# OptDB["ksp_type"] = "gmres" # preonly, gmres
# OptDB["pc_type"] = "jacobi" # none, jacobi, ilu, icc
     # Linear solver: direct sparse solver
OptDB["ksp_type"] = "preonly"
OptDB["pc_type"] = "lu"
OptDB["pc_factor_mat_solver_type"] = "mumps"
Integration_params = {'cinit': True,'dt_export':0.5*dt}
PETSc_utils.TS_integration_dae(dae,x0,dt,tf,**Integration_params)
tp = dae.history['t']
yp = dae.history['y']
#%% Plot with blockfunction
i_plot = np.min((200,len(yp)-1))
u = multiphenics_utils.get_function(yp[i_plot],0,"e_u",W)
u_ex_fun = fe.interpolate(u_ex,V_u)
# c=fe.plot(u,mode='color', vmin=0, vmax=0.5)
c=fe.plot(u,mode='color')
# c=fe.plot(u_ex_fun,mode='color')
ax=c.axes
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title(f"{dae.name} t={tp[i_plot]:1.2e}")
# ax.set_title(f"Exact solution t=inf")
plt.colorbar(c)
error_L2 = fe.errornorm(u_ex, u, 'L2')
print(f"L2 error at t={tp[i_plot]}: {error_L2}")
#%% L2 norm
H = Wave_semilinear_compute_norm(tp,yp,W,dmesh.dx)
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(tp,H,marker="o")
ax.set_title(dae.name)
ax.set_xlabel("t")
ax.set_ylabel("H")
#%% Export to xdmf
xdmfile = 'heat.xdmf'
sol_idx=[0]; sol_name = ["u"]
multiphenics_utils.export_xdmf(xdmfile,sol_idx,sol_name,tp,yp,W)
#%% Export to pvd
pvdfile = 'heatvtk/solution.pvd'
sol_idx=0; sol_name = "u"
multiphenics_utils.export_pvd(pvdfile,sol_idx,sol_name,tp,yp,W)