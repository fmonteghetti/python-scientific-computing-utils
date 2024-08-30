#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script demonstrates the use of *multiples* periodicity conditions in 
fenics when working on meshes from gmsh. This is handled using a custom
PeriodicBoundary() class that lightens the syntax.

The PDE is
    -Delta(u) = f, on disk {|x|<R}
    u=0 on {|x|=R}.

The corresponding eigenvalue problem (f -> lambda*u) is also solved.


The disk is truncated at r=R_TR and the outer annulus {R_TR<|x|<R} is
discretized in Euler coordinates (z,theta)=(ln(r),theta), where (r,theta)
are the polar coordinates centered at (x,y)=(0,0). Periodicity conditions
are thus needed to link the Euler domain to the Cartesian domain.

"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import fenics as fe
from scicomp_utils_mesh import gmsh_utils
from scicomp_utils_dolfin import fenics_utils
from scicomp_utils_mesh import meshio_utils
from slepc4py import SLEPc
from scicomp_utils_misc import SLEPc_utils
from scicomp_utils_misc import PDE_exact
import os
DIR_MESH=os.path.join(os.path.dirname(os.path.abspath(__file__)),"mesh")

# Load mesh
geofile=os.path.join(DIR_MESH,"Circle-Rectangle-Periodic.geo")
mshfile=os.path.join(DIR_MESH,"Circle-Rectangle-Periodic.msh")
R = 2
R_TR=0.5*R
y_offset = -R-fe.DOLFIN_PI
gmsh_utils.generate_mesh_cli(geofile,mshfile,2,refinement=0,binary=True,
                             parameters={'lc':R/110,'R':R,'R_TR': R_TR,
                                         'y_offset': y_offset})
meshio_utils.print_diagnostic_gmsh(mshfile)
    # convert to xdmf
xdmfiles = meshio_utils.convert_gmsh_to_XDMF(mshfile,prune_z=True)    
dmesh = fenics_utils.DolfinMesh.init_from_xdmf(xdmfiles['triangle'],xdmfiles['line'])
    # physical names from JSON file (gmsh_utils.getPhysicalNames also possible)
gmsh_phys = meshio_utils.read_phys_gmsh(xdmfiles['gmsh_physical_entities'])
PhysName_1D_tag = gmsh_phys[0]; PhysName_2D_tag = gmsh_phys[1]
dx = dmesh.dx
fe.plot(dmesh.mesh,title=f"{mshfile} \n {dmesh.mesh.num_vertices()} vertices")
#% Multiple periodicity conditions
def isRectTopBnd(x):
    return fe.near(x[1],y_offset+fe.DOLFIN_PI)

def isRectBotBnd(x):
    return fe.near(x[1],y_offset-fe.DOLFIN_PI)

def isRectLeftBnd(x):
    return fe.near(x[0],np.log(R_TR))

def isDiskBnd(x):
    return fe.near(np.sqrt(x[0]**2+x[1]**2),R_TR)

def map_RectBotBnd_to_RectTopBnd(x,y):
    y[0] = x[0]; y[1] = x[1]+2*fe.DOLFIN_PI

def map_RectLeftBnd_to_DiskBnd(x,y):
    y[0] = R_TR*np.cos(x[1]-y_offset); y[1] = R_TR*np.sin(x[1]-y_offset)
    # Initialize PeriodicBoundary object
per = fenics_utils.PeriodicBoundary(); per.init()
    # Rectangle bottom -> Rectangle top
per.append(isRectBotBnd,isRectTopBnd,map_RectBotBnd_to_RectTopBnd)
    # Rectangle left -> Disk boundary
per.append(isRectLeftBnd,isDiskBnd,map_RectLeftBnd_to_DiskBnd)
#% Weak formulation
V = fe.FunctionSpace(dmesh.mesh, 'P', 1,constrained_domain=per)
    # Dirichlet boundary condition
u_D = fe.Constant(0.0)
bc = fe.DirichletBC(V, u_D, dmesh.boundaries,\
                    PhysName_1D_tag['Rectangle-Boundary-Right'][0])
    # Variational forms
u = fe.TrialFunction(V)
v = fe.TestFunction(V)
f = fe.Constant(-6.0)
a = fe.dot(fe.grad(u), fe.grad(v))*dx
weight = fe.Expression('exp(2*x[0])', degree=1)
m = u*v*dx(PhysName_2D_tag['Disk'][0])+\
    weight*u*v*dx(PhysName_2D_tag['Rectangle'][0])
l = f*v*dx(PhysName_2D_tag['Disk'][0])+\
    weight*f*v*dx(PhysName_2D_tag['Rectangle'][0])
#% Solve direct problem
u_fe = fe.Function(V)
fe.solve(a == l,u_fe,bc,solver_parameters={"linear_solver": "lu"})
err = u_D.compute_vertex_values(dmesh.mesh)-u_fe.compute_vertex_values(dmesh.mesh)
err = np.sqrt(np.average(err**2))
print(f"L2 error (High-level): {err}")
c=fe.plot(u_fe,cmap='jet',title="Direct problem")
ax=c.axes
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.colorbar(c)

#% Solve sparse eigenvalue problem
    # Assemble PETSc matrices
A = fe.PETScMatrix(); M = fe.PETScMatrix()
fenics_utils.assemble_GEP(A,a,M,m,[bc],diag_A=1e2,diag_B=1e-2)
EPS = SLEPc_utils.solve_GEP_shiftinvert(A.mat(),M.mat(),
                          problem_type=SLEPc.EPS.ProblemType.GHEP,
                          solver=SLEPc.EPS.Type.KRYLOVSCHUR,
                          nev=20, target=1.0,shift=1.0)
(eigval,eigvec_r,eigvec_i) = SLEPc_utils.EPS_get_spectrum(EPS)
#% Plot eigenfunction (using a dolfin function)
i_plot = 6
eigvec_rp = fe.PETScVector(eigvec_r[i_plot])
u_fe=fe.Function(V,eigvec_rp/eigvec_rp.norm('linf'))
f = plt.figure()
ax=f.add_subplot(1,1,1)
coll=fe.plot(u_fe,title=f"Eigenvalue {eigval[i_plot]}",cmap='jet')
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.colorbar(coll)
#% Plot eigenfunction in (x,y) by interpolating back (using map())
def map_xy_to_ztheta(x,y):
    z=np.log(np.linalg.norm([x,y])); theta=np.arctan2(y,x)+y_offset;
    return (z,theta)

def func(x, y):
    f=0
    if (np.linalg.norm([x,y])<R) and (np.linalg.norm([x,y])>R_TR):
        f=u_fe(map_xy_to_ztheta(x,y))
    elif (np.linalg.norm([x,y])<R_TR):
        f=u_fe([x,y])
    return f

x = np.linspace(-R,R,50); y = x;
Z = np.array([[func(i,j) for i in x] for j in y])
Z = Z/np.max(np.abs(Z))

fig = plt.figure(1)
ax = fig.add_subplot(1,1,1)
coll = ax.pcolormesh(x, y, Z, shading='gouraud', cmap='jet',vmin=-1, vmax=1)
fig.colorbar(coll,ax=ax)
    # clip with a circle path
circ = mpl.patches.Circle((0, 0), R, transform=ax.transData)
coll.set_clip_path(circ)
ax.set_aspect('equal') # orthonormal axis
ax.set_title(f"Eigenvalue {eigval[i_plot]}")
ax.set_xlabel("x")
ax.set_ylabel("y")
# Validate computed eigenvalues
ev = PDE_exact.Laplace_spectrum.disc(R,'dirichlet',eigval_min=0,eigval_max=100,eigval_N=10)
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(np.real(ev),np.imag(ev),marker='o',linestyle='none',label='exact')
ax.plot(np.real(eigval),np.imag(eigval),marker='x',linestyle='none',label=f'FEM N={V.dim()}')
ax.set_title(f"Eigenvalues for R={R}")
ax.set_xlabel("$\Re(\lambda)$")
ax.set_ylabel("$\Im(\lambda)$")
ax.legend()
ax.set_xlim([0,ev[4]])