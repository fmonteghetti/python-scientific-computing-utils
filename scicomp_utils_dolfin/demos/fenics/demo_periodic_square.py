#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script demonstrates the use of *multiples* periodicity conditions in 
fenics when working on meshes from gmsh. This is handled using a custom
PeriodicBoundary() class that lightens the syntax.

The problem solved is: find (u,lambda) such that

    -Delta(u) = lambda * u, on Omega=(0,L1)x(0,L2).

Different set of boundary conditions are available: Dirichlet, Neumann, periodic.
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
L = [1, 1] # square dimensions
bc_hor = "periodic" # boundary conditions on {x=0} and {x=L1} 
                    # periodic, neumann, dirichlet
bc_vert = "periodic" # boundary conditions on {x=0} and {y=L2}
                      # periodic, neumann, dirichlet
# Load mesh
geofile=os.path.join("mesh","Rectangle.geo")
mshfile=os.path.join("mesh","Rectangle.msh")
gmsh_utils.generate_mesh_cli(geofile,mshfile,2,refinement=0,binary=True,
                             parameters={'x_r':L[0],'y_t':L[1],
                                         'N_x': 51, 'N_y': 51})
meshio_utils.print_diagnostic_gmsh(mshfile)
    # convert to xdmf
xdmfiles = meshio_utils.convert_gmsh_to_XDMF(mshfile,prune_z=True)
dmesh = fenics_utils.DolfinMesh.init_from_xdmf(xdmfiles['triangle'],xdmfiles['line'])
    # physical names from JSON file (gmsh_utils.getPhysicalNames also possible)
gmsh_phys = meshio_utils.read_phys_gmsh(xdmfiles['gmsh_physical_entities'])
PhysName_1D_tag = gmsh_phys[0]; PhysName_2D_tag = gmsh_phys[1]
dx = dmesh.dx
#% Multiple periodicity conditions
def isRectTopBnd(x):
    return fe.near(x[1],L[1])

def isRectBotBnd(x):
    return fe.near(x[1],0.0)

def isRectLeftBnd(x):
    return fe.near(x[0],0.0)

def isRectRightBnd(x):
    return fe.near(x[0],L[0])

def map_RectBotBnd_to_RectTopBnd(x,y):
    y[0] = x[0]; y[1] = x[1]+L[1]

def map_RectLeftBnd_to_RectRightBnd(x,y):
    y[0] = x[0]+L[0]; y[1] = x[1]
    # Initialize PeriodicBoundary object
per = fenics_utils.PeriodicBoundary(); per.init()
    # Rectangle bottom -> Rectangle top
if bc_vert=="periodic":
    per.append(isRectBotBnd,isRectTopBnd,map_RectBotBnd_to_RectTopBnd)
    # Rectangle left -> Rectangle right
if bc_hor=="periodic":
    per.append(isRectLeftBnd,isRectRightBnd,map_RectLeftBnd_to_RectRightBnd)
#% Weak formulation
V = fe.FunctionSpace(dmesh.mesh, 'P', 1,constrained_domain=per)
    # Dirichlet boundary condition
bcs = []
u_D = fe.Constant(0.0)
if bc_vert=="dirichlet":
    bcs = [fe.DirichletBC(V, u_D, dmesh.boundaries,\
                        PhysName_1D_tag['Rectangle-Boundary-Top'][0]),
           fe.DirichletBC(V, u_D, dmesh.boundaries,\
                               PhysName_1D_tag['Rectangle-Boundary-Bot'][0])]
elif bc_hor=="dirichlet":
    bcs = [fe.DirichletBC(V, u_D, dmesh.boundaries,\
                        PhysName_1D_tag['Rectangle-Boundary-Right'][0]),
           fe.DirichletBC(V, u_D, dmesh.boundaries,\
                               PhysName_1D_tag['Rectangle-Boundary-Left'][0])]
    # Variational forms
u = fe.TrialFunction(V)
v = fe.TestFunction(V)
a = fe.dot(fe.grad(u), fe.grad(v))*dx
m = u*v*dx
    # Assemble PETSc matrices
A = fe.PETScMatrix(); M = fe.PETScMatrix()
fenics_utils.assemble_GEP(A,a,M,m,bcs,diag_A=1e2,diag_B=1e-2)
EPS = SLEPc_utils.solve_GEP_shiftinvert(A.mat(),M.mat(),
                          problem_type=SLEPc.EPS.ProblemType.GHEP,
                          solver=SLEPc.EPS.Type.KRYLOVSCHUR,
                          nev=20, target=1.0,shift=1.0)
(eigval,eigvec_r,eigvec_i) = SLEPc_utils.EPS_get_spectrum(EPS)
# Validate computed eigenvalues
ex_h = PDE_exact.Laplace_spectrum.interval(L[0], 50, bc_hor)
ex_v = PDE_exact.Laplace_spectrum.interval(L[1], 50, bc_vert)
ev_ex = np.sort([r+q for r in ex_h for q in ex_v])
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(np.real(eigval),np.imag(eigval),marker='o',linestyle='none',label=f'FEM N={V.dim()}')
ax.plot(np.real(ev_ex),np.imag(ev_ex),marker='x',linestyle='none',label='Exact')
title=f"Rectangle: {bc_hor} at x=0,{L[0]} and {bc_vert} at y=0,{L[1]}"
ax.set_title(title)
ax.set_xlabel("$\Re(\lambda)$")
ax.set_ylabel("$\Im(\lambda)$")
ax.legend()
ax.set_xlim([0,ev_ex[4]])
fig.savefig(f"square_{bc_hor}-{bc_vert}_spectrum_fenics.png")
# Plot sparsity pattern
import scipy.sparse as sp
import matplotlib.pyplot as plt
A_petsc = A.mat()
(indptr,indices,val) = A_petsc.getValuesCSR()
A_scipy = sp.csr_matrix((val, indices, indptr), shape=A_petsc.size)
plt.figure()
plt.spy(A_scipy)
plt.title(title)
plt.savefig(f"square_{bc_hor}-{bc_vert}_A_fenics.png")
    # Paraview export of all eigenmodes
xdmfile = f"square_{bc_hor}-{bc_vert}_fenics.xdmf"
output = fe.XDMFFile(xdmfile)
output.parameters["flush_output"] = True
output.parameters["rewrite_function_mesh"] = False
output.parameters["functions_share_mesh"] = True
print(f"Export to {xdmfile}...")
for i in range(len(eigvec_r)):
    idx_u = 0
    name = f"u_{i:02d}_{eigval[i]:4.4g}"
    eigvec_p = fe.PETScVector(eigvec_r[i])
    fun=fe.Function(V,eigvec_p/eigvec_p.norm('linf'))
    fun.rename(name,"")
    output.write(fun, 0)