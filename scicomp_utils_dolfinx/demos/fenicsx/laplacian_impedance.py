#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solution of the impedance Laplace eigenvalue problem:

    -Delta(u)=lambda^2*u in Omega, du/dn = - z*i*lambda*u on Gamma,

where z>=0 is the impedance. The weak formulation in H^1(Ω) gives an eigenvalue
problem that is quadratic in lambda. This is a non-self-adjoint eigenvalue
problem; when z->0 (resp. z->infty), we recover the Dirichlet (resp.
Neumann) Laplacian.

The script can be run on one or multiple threads:
    python ./MPI-test.py (one thread)
    mpirun -n N ./MPI-test.py (N threads).
The latter can be executed from the ipython prompt using:
    ! mpirun -n N ./MPI-test.py.

"""
#%%
from mpi4py import MPI
import scipy.special
comm = MPI.COMM_WORLD # dolfinx, petsc, and slepc use mpi communicators
from petsc4py import PETSc
from slepc4py import SLEPc
import pyvista as pv
import numpy as np
import time
import basix
import dolfinx
import dolfinx.fem.petsc
import ufl
from scicomp_utils_mesh import gmsh_utils
from scicomp_utils_dolfinx import fenicsx_utils
from scicomp_utils_misc import SLEPc_utils
import matplotlib.pyplot as plt
import os
DIR_MESH=os.path.join(os.path.dirname(os.path.abspath(__file__)),"mesh")

impedance = 1e-16
degree_mesh = 2
degree_fem = 2
geofile=os.path.join(DIR_MESH,"Disc.geo")
Gamma_name = ['Gamma-R'] # physical name of boundary
Omega_name = ['Omega'] # physical name of domain
dim = 2
# Eigenvalues of Dirichlet Laplacian on disc of radius 1
eigval_dirichlet = np.hstack([scipy.special.jn_zeros(n, 5)**2 for n in range(0,4)]) 
# Eigenvalues of Neumann Laplacian on disc of radius 1
eigval_neumann = np.hstack([scipy.special.jnp_zeros(n, 5)**2 for n in range(0,4)]) 

    # == Create and read mesh
gmshfile = geofile.replace(".geo",".msh")
if comm.rank==0: # generate mesh on first thread only
    gmsh_utils.generate_mesh_cli(geofile,gmshfile,dim,refinement=0,binary=True,
                              parameters={'lc':1/10},order=degree_mesh)
comm.Barrier()

    # Read .msh file and build a distributed mesh
dmesh = fenicsx_utils.DolfinxMesh.init_from_gmsh(gmshfile,dim,comm=comm)
dim, mesh, name_2_tags = dmesh.dim, dmesh.mesh, dmesh.name_2_tags
Gamma_tags = [t for name in Gamma_name for t in name_2_tags[-2][name]]

    # == Eigenvalue problem 
    # k(u,v) + lambda * c(u,v) + lambda**2 * m(u,v) = 0
V = dolfinx.fem.functionspace(mesh, ("CG", degree_fem))
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
k = ufl.inner(ufl.grad(u), ufl.grad(v)) * dmesh.dx
m = - ufl.inner(u, v) * dmesh.dx
c = 1j * impedance *  ufl.inner(u, v) * dmesh.ds
K = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(k)); K.assemble()
M = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(m)); M.assemble()
C = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(c)); C.assemble()
PEP = SLEPc_utils.solve_PEP_shiftinvert([K,C,M],comm=comm,
                                nev=20,tol=1e-6,max_it=100, target=4,shift=4)
(eigval,eigvec_r,eigvec_i) = SLEPc_utils.EPS_get_spectrum(PEP)
comm.Barrier()

    # Export eigenfunctions to several .vtu file using pyvista
grid = fenicsx_utils.create_pyvista_UnstructuredGrid_from_FunctionSpace(V)
u_out = dolfinx.fem.Function(V)
for i in range(len(eigval)):
    name = f"l_{i}_{eigval[i]:4.4g}"
        # eigvec_r[i] contains only the local dofs, not the ghost dofs
    u_out.x.petsc_vec.setArray(eigvec_r[i])
    u_out.x.scatter_forward()
    grid.point_data[name+"_ur"] = u_out.x.array.copy()
    u_out.x.petsc_vec.setArray(eigvec_i[i])
    u_out.x.scatter_forward()
    grid.point_data[name+"_ui"] = u_out.x.array.copy()
grid.save(f"shared/demo_laplace_impedance_eig_{comm.rank}.vtu")

# Compare computed spectrum to exact Dirichlet and Neumann  eigenvalues
fig = plt.figure()
ax=fig.add_subplot(1,2,1)
ax.plot(np.real(np.sqrt(eigval_dirichlet)),np.imag(np.sqrt(eigval_dirichlet)),
                    label = "Exact Dirichlet", marker='o',linestyle='none',
                    fillstyle="none")
ax.plot(np.real(np.sqrt(eigval_neumann)),np.imag(np.sqrt(eigval_neumann)),
                      label = "Exact Neumann", marker='s',linestyle='none',
                      fillstyle="none")
ax.plot(np.real(eigval),np.imag(eigval), 
                            label=f"Computed (z={impedance:1.1e})",
                            marker='+', linestyle='none')
ax.set_xlabel(r"$\Re(\lambda)$")
ax.set_ylabel(r"$\Im(\lambda)$")
ax.grid(True)
ax.set_ylim([-10,10])
ax.set_xlim([0,8])
ax.legend()