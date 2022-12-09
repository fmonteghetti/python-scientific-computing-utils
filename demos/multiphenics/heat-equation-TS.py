#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Heat equation with mixed boundary condition:
    
    du/dt = Delta(u) + f on Omega, u=u_D on Gamma_D, du/dn=u_N on Gamma_N.

This example demonstrate the initialization of PDAE_linear to perform
assembly using multiphenics and time-integration using PETSc TS.

Both 2D and 3D meshes can be used.

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
#%% Generate mesh using gmsh (2D)
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
geofile=os.path.join(DIR_MESH,"sphere-shell.geo")
gmshfile=os.path.join(DIR_MESH,"sphere-shell.msh")

dim = 3
gmsh_utils.generate_mesh_cli(geofile,gmshfile,dim,refinement=0,log=1,\
                             parameters={'R_i':0.5,'R_o':1,'lc':1/12},
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
#%% Problem definition
step = '*tanh(t/0.5)' # tanh(1)=0.76
# u_D_expr = 't*cos(10*x[0])'
u_D_expr = '4+4*cos(10*(atan2(x[1],x[0])-2*pi*t))'
# u_D_expr = '('+u_D_expr+')'+step
# u_N_expr = 'cos(10*(atan2(x[1],x[0])-2*pi*t))'
u_N_expr = '4'
# f_expr = 'cos(10*(atan2(x[1],x[0])-2*pi*t))'
f_expr = '4'
    # Boundaries
Dirichlet_bnd = meshtag[dim-2][Dirichlet_bnd_name][0]
Neumann_bnd = meshtag[dim-2][Neumann_bnd_name][0]
Dirichlet_bnd_res =  multiphenics_utils.build_MeshRestriction_from_tags(
                                                    dmesh.mesh,
                                                    dmesh.boundaries,
                                                    [Dirichlet_bnd])
ds_D = dmesh.ds(Dirichlet_bnd)
ds_N = dmesh.ds(Neumann_bnd)
    # Function spaces
V = fe.FunctionSpace(dmesh.mesh, "Lagrange", 1)
W = mpfe.BlockFunctionSpace([V, V], restrict=[None, Dirichlet_bnd_res])
    # Test/Trial functions
fun_test = mpfe.block_split(mpfe.BlockTestFunction(W))
fun_trial = mpfe.block_split(mpfe.BlockTrialFunction(W))
    # Lagrange Multipliers
idx_Lagrange_multipliers= W.block_dofmap().block_owned_dofs__global_numbering(1)
    
def Wave_jacobian(fun_trial,fun_test,dx,ds_D):
    """ Jacobian matrix for residual. """
    (u, l) = fun_trial; (v_u, v_l) = fun_test
    J_yd = [[u*v_u*dx]]
    J_y = [
        [fe.inner(fe.grad(u), fe.grad(v_u))*dx, -l*v_u*ds_D],
        [u*v_l*ds_D]]
    return (J_yd,J_y)

def Wave_residual_src(u_D,u_N,f,t,fun_test,dx,ds_D,ds_N):
    """ External forcing. """
    u_D.t = t; u_N.t = t; f.t = t
    (v_u,v_l) = fun_test
    return [-f*v_u*dx - u_N*v_u*ds_N,
            -u_D*v_l*ds_D]

def Wave_compute_norm(t,y,W,dx):
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
u_D = fe.Expression(u_D_expr, element=V.ufl_element(),t=0)
u_N = fe.Expression(u_N_expr, element=V.ufl_element(),t=0)
f = fe.Expression(f_expr, element=V.ufl_element(),t=0)
    # Build linear PDAE
residual_src = lambda  t : Wave_residual_src(u_D,u_N,f,t,fun_test,dmesh.dx,ds_D,ds_N)
jacobian = lambda : Wave_jacobian(fun_trial,fun_test,dmesh.dx,ds_D)
name = f"Heat equation - Linear"
dae = multiphenics_utils.PDAE_linear(residual_src,jacobian,
            W.dim(),name,idx_alg=idx_Lagrange_multipliers)
#%% Time integration
    # Time parameters
tf = 2
dt = 0.1/10
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
OptDB["ksp_type"] = "gmres" # preonly, gmres
OptDB["pc_type"] = "jacobi" # none, jacobi, ilu, icc
     # Linear solver: direct sparse solver
# OptDB["ksp_type"] = "preonly"
# OptDB["pc_type"] = "lu"
# OptDB["pc_factor_mat_solver_type"] = "mumps"
Integration_params = {'cinit': True,'dt_export':0.5*dt}
PETSc_utils.TS_integration_dae(dae,x0,dt,tf,**Integration_params)
tp = dae.history['t']
yp = dae.history['y']
#%% Plot with blockfunction (2D only)
i_plot = np.min((1,len(yp)-1))
u = multiphenics_utils.get_function(yp[i_plot],0,"u",W)
# c=fe.plot(u,mode='color', vmin=0, vmax=0.5)
c=fe.plot(u,mode='color')
ax=c.axes
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title(f"{dae.name} t={tp[i_plot]:1.2e}")
# ax.set_title(f"Exact solution t=inf")
plt.colorbar(c)
#%% L2 norm
H = Wave_compute_norm(tp,yp,W,dmesh.dx)
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
#%% Animate with matplolib (2D only)
import matplotlib.pyplot as plt
from matplotlib import animation
tp_dt = np.mean(np.ediff1d(tp))
fps = 0.5 * len(yp)/tp[-1]
colorbar_set = np.array([0])
def animate(i,colorbar_set):
    print(f"frame {i} / {len(yp)} [t={tp[i]:1.3g}]")
    u = multiphenics_utils.get_function(yp[i],0,"u",W)
    c = fe.plot(u,mode='color',vmin=-1,vmax=1)
    ax=c.axes
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"{dae.name} t={tp[i]:1.8e}")
    if colorbar_set[0]==0: # avoid multiple colorbar
        plt.colorbar(c)
        colorbar_set[0] = 1
    return c

fig = plt.gcf()
ani = animation.FuncAnimation(fig, lambda i: animate(i,colorbar_set), 
                              blit=False, cache_frame_data=True,
                              frames=len(yp),interval=20)
writer = animation.FFMpegWriter(
    fps=fps, metadata=dict(title='Fenics video'), bitrate=-1)
ani.save("movie.mp4", writer=writer)