#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script demonstrates a basic usage of gmsh in fenics:
    - Generate mesh using gmsh (parameter-dependent)
    - Load mesh (in Dolfin-XML or XDMF format)
    - Bilinear forms defined by integration over subboundaries or subdomains
    using mesh physical names
It also shows the usage of SLEPc to solve a generalized eigenvalue pb.

The PDE is a Poisson equation with weight w(x).

"""

import numpy as np
import matplotlib.pyplot as plt
import fenics as fe
import gmsh_utils
import gmsh_utils_fenics
import fenics_utils
import meshio_utils
import os
#%% Generate mesh using gmsh (2D)
# Dolfin only fully supports first-order meshes
geofile=os.path.join("mesh","Circle.geo")
gmshfile=os.path.join("mesh","Circle.msh")
geofile=os.path.join("mesh","Circle-simple.geo")
gmshfile=os.path.join("mesh","Circle-simple.msh")
dim = 2
gmsh_utils.generate_mesh_cli(geofile,gmshfile,dim,refinement=0,log=1,\
                             parameters={'R':1,'R_TR':0.5,'lc':1/40},order=1,
                             gmshformat=2,binary=False)
gmsh_utils.print_summary(gmshfile)
Dirichlet_bnd_name = 'Gamma-ext'
Omega_int_name = 'Omega-int'
Omega_ext_name = 'Omega-ext'
#%% Generate mesh using gmsh (3D)
geofile=os.path.join("mesh","torus.geo")
gmshfile=os.path.join("mesh","torus.msh")
geofile=os.path.join("mesh","torusdouble.geo")
gmshfile=os.path.join("mesh","torusdouble.msh")
dim = 3
gmsh_utils.generate_mesh_cli(geofile,gmshfile,dim,refinement=0,log=1,\
                             parameters={'R_mi':0.5,'R_ma':2,'lc':1/5},
                             order=1,binary=True)
gmsh_utils.print_summary(gmshfile)
Dirichlet_bnd_name = 'Gamma-T'
Omega_int_name = 'Omega-T'
Omega_ext_name = 'Omega-B'
#%% Load mesh: dolfin-XML
# Not compatible with binary gmsh mesh
# Only compatible with msh2 format
# Dolfin-XML format is space-consuming and deprecated
    # physical names
meshtag = [gmsh_utils.getPhysicalNames(gmshfile,1)]
meshtag.append(gmsh_utils.getPhysicalNames(gmshfile,2))
    # Convert to dolfin-xml and load
xmlfile = gmsh_utils_fenics.mesh_convert_gmsh2_to_dolfinXML(gmshfile)
dmesh = fenics_utils.DolfinMesh.init_from_gmsh(xmlfile)
#%% Load mesh: XDMF format
# Handles binary gmsh meshes, both msh2 and msh4 formats
# XDMF is a standard format that can be handled using meshio
    # print diagnostic using only meshio, no gmsh api needed
meshio_utils.print_diagnostic_gmsh(gmshfile)
    # convert to xdmf
xdmfiles = meshio_utils.convert_gmsh_to_XDMF(gmshfile,prune_z=True)
dmesh = fenics_utils.DolfinMesh.init_from_xdmf(xdmfiles['tetra'*(dim==3)+'triangle'*(dim==2)],
                                               xdmfiles['triangle'*(dim==3)+'line'*(dim==2)])
    # physical names from JSON file (gmsh_utils.getPhysicalNames also possible)
meshtag = meshio_utils.read_phys_gmsh(xdmfiles['gmsh_physical_entities'])
#%% Mesh tags
Dirichlet_bnd = meshtag[dim-2][Dirichlet_bnd_name]
Omega_int = meshtag[dim-1][Omega_int_name]
Omega_ext = meshtag[dim-1][Omega_ext_name]
#%% Plot Dolfin mesh (2D only)
fe.plot(dmesh.mesh,title=f"{dmesh.mesh.num_vertices()} vertices")
# coord = dmesh.mesh.coordinates()
# plt.plot(coord[:,0],coord[:,1],marker='*',linestyle='none')
#%% Weak formulation
V = fe.FunctionSpace(dmesh.mesh, 'P', 1)
    # Dirichlet boundary condition
u_D = fe.Constant(0.0)
bc = fe.DirichletBC(V, u_D, dmesh.boundaries,Dirichlet_bnd[0])
    # Variational forms
u = fe.TrialFunction(V)
v = fe.TestFunction(V)
f = fe.Constant(-6.0)
w = fe.Constant(2.0)
a = fe.dot(fe.grad(u), fe.grad(v))*dmesh.dx(Omega_int[0]) + \
    w*fe.dot(fe.grad(u), fe.grad(v))*dmesh.dx(Omega_ext[0])
m = u*v*dmesh.dx
l = f*v*dmesh.dx
#%% Solve
u_fe = fe.Function(V)
fe.solve(a == l,u_fe,bc,solver_parameters={"linear_solver": "lu"})
err = u_D.compute_vertex_values(dmesh.mesh)-u_fe.compute_vertex_values(dmesh.mesh)
err = np.sqrt(np.average(err**2))
print(f"L2 error (High-level): {err}")
fenics_utils.export_xdmf("basics.xdmf",["u"],[V],[0],[u_fe.vector().vec()])
#%% Plot, the high-level way with dolfin functions (2D only)
c=fe.plot(u_fe,cmap='jet',title="Direct problem")
ax=c.axes
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.colorbar(c)
#%% Assemble sparse eigenvalue problem
A = fe.PETScMatrix(); M = fe.PETScMatrix()
fenics_utils.assemble_GEP(A,a,M,m,[bc],diag_A=1e2,diag_B=1e-2)
#%% Solve sparse eigenvalue problem: fenics wrapper on SLEPc
# Pros: easy to use
# Cons: limited options / no monitoring / no detailled summary
EPS = fenics_utils.create_GEP(A,M,opts={
    'spectrum': "smallest magnitude",
    'solver': "arpack",
    'tolerance': 1e-5,
    'maximum_iterations': 100,
    'problem_type': 'gen_hermitian',
#    'spectral_transform' : 'shift-and-invert',
#    'spectral_shift' : 0.0
    })
EPS.solve(5)
print(f"No. of converged eigenvalues: {EPS.get_number_converged()}")
eigval = np.zeros((EPS.get_number_converged(),))
eigval = [EPS.get_eigenvalue(i)[0]+1j*EPS.get_eigenvalue(i)[1]
          for i in range(len(eigval))]
def get_eigenvec(i):
    return EPS.get_eigenpair(i)[2:4]
#%% Solve sparse eigenvalue problem:  using SLEPc directly for more control
from slepc4py import SLEPc
import SLEPc_utils
A_petsc = A.mat(); M_petsc = M.mat()
SLEPc_params = {'nev': 50,
             'target': 10,
             'shift': 10,
             'problem_type': SLEPc.EPS.ProblemType.GHEP,
             'solver': SLEPc.EPS.Type.KRYLOVSCHUR,
              'tol': 1e-6,
             'max_it': 100}
EPS = SLEPc_utils.solve_GEP_shiftinvert(A_petsc,M_petsc,**SLEPc_params)
(eigval,eigvec_r,eigvec_i) = SLEPc_utils.EPS_get_spectrum(EPS)

def get_eigenvec(i):
    return fe.PETScVector(eigvec_r[i]),fe.PETScVector(eigvec_i[i])
#%% Plot (using a dolfin function) (2D only)
i_plot = 10
eigvec_r_p,eigvec_i_p = get_eigenvec(i_plot)

fun=fe.Function(V,eigvec_r_p/eigvec_r_p.norm('linf'))
f = plt.figure()
ax=f.add_subplot(2,1,1)
c=fe.plot(fun,axes=ax,title=f"$\lambda=${eigval[i_plot]:2.2g}",
          mode='color', vmin=-1, vmax=1)
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.colorbar(c)
ax=f.add_subplot(2,1,2)
ax.plot(np.real(eigval),np.imag(eigval),linestyle='none',marker='o')
ax.plot(np.real(eigval[i_plot]),np.imag(eigval[i_plot]),linestyle='none',marker='x')
ax.set_xlabel(r"$\Re(\lambda)$")
ax.set_ylabel(r"$\Im(\lambda)$")
ax.set_xlim([0,100])
#%% Export eigenvectors
fenics_utils.export_xdmf("basics.xdmf",["u"],[V],np.r_[0:len(eigval)],eigvec_r)