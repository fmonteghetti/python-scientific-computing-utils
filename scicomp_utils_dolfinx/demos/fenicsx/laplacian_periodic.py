#!/usr/bin/env python3
"""
This script demonstrates the use of *multiples* periodicity conditions in 
fenics when working on meshes from gmsh. This is handled using a custom
PeriodicBoundary() class that lightens the syntax.

The problem solved is: find (u,lambda) such that

    -Delta(u) = lambda * u, on Omega=(0,L1)x(0,L2).

Different set of boundary conditions are available: Dirichlet, Neumann, periodic.
"""
#%%
import dolfinx
import dolfinx.io
import dolfinx_mpc
import dolfinx_mpc.utils

import ufl
import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD
from petsc4py import PETSc
    # Eigenvalue problem
from slepc4py import SLEPc
from scicomp_utils_misc import SLEPc_utils
from scicomp_utils_dolfinx import fenicsx_utils
from scicomp_utils_dolfinx import dolfinx_mpc_utils
    # Mesh
from scicomp_utils_mesh import gmsh_utils
from scicomp_utils_mesh import meshio_utils
    # Plot
import matplotlib.pyplot as plt
import pyvista as pv
from scicomp_utils_dolfinx import pyvista_utils
    # Validation
from scicomp_utils_misc import PDE_exact
import os
DIR_MESH=os.path.join(os.path.dirname(os.path.abspath(__file__)),"mesh")

L = [1, 1] # square dimensions
bc_hor = "periodic" # boundary conditions on {x=0} and {x=L1} 
                     # periodic, neumann, dirichlet
bc_vert = "periodic" # boundary conditions on {x=0} and {y=L2}
                      # periodic, neumann, dirichlet

# Load mesg
geofile=os.path.join(DIR_MESH,"Rectangle.geo")
gmshfile=os.path.join(DIR_MESH,"Rectangle.msh")
if comm.rank==0: # generate mesh on first thread only
    gmsh_utils.generate_mesh_cli(geofile,gmshfile,2,refinement=0,binary=True,
                                 parameters={'x_r':L[0],'y_t':L[1],
                                             'N_x': 100, 'N_y': 100})
comm.Barrier()
dmesh = fenicsx_utils.DolfinxMesh.init_from_gmsh(gmshfile,2,comm=comm)
V = dolfinx.fem.functionspace(dmesh.mesh, ("CG", 2))

    # Horizontal pbc
pbc_slave_tag_hor = dmesh.name_2_tags[-2]['Rectangle-Boundary-Left']
pbc_hor_is_slave = lambda x: x[0] < 1e-14
pbc_hor_is_master = lambda x: x[0] > (1-1e-14)
def pbc_rel_hor(x): # slave to master
    out_x = np.zeros(x.shape)
    out_x[0] = x[0] + L[0]
    out_x[1] = x[1]
    out_x[2] = x[2]
    return out_x
    # Vertical pbc
pbc_slave_tag_vert = dmesh.name_2_tags[-2]['Rectangle-Boundary-Bot']
def pbc_rel_vert(x):
    out_x = np.zeros(x.shape)
    out_x[0] = x[0]
    out_x[1] = x[1] + L[1]
    out_x[2] = x[2]
    return out_x

    # Dirichlet boundary condition
bcs = []
x = V.tabulate_dof_coordinates()
uD = dolfinx.fem.Function(V)
uD.x.array[:] = PETSc.ScalarType(0)
uD.x.scatter_forward()
bcs = []
if bc_vert=="dirichlet":
    Dirichlet_bnd_name = ['Rectangle-Boundary-Top','Rectangle-Boundary-Bot']
    Dirichlet_bnd = [t for name in Dirichlet_bnd_name for t in dmesh.name_2_tags[-2][name]]
    bcs+=fenicsx_utils.create_DirichletBC(dmesh.mesh,dmesh.facet_tags,V,[uD,uD],Dirichlet_bnd)
if bc_hor=="dirichlet":
    Dirichlet_bnd_name = ['Rectangle-Boundary-Left','Rectangle-Boundary-Right']
    Dirichlet_bnd = [t for name in Dirichlet_bnd_name for t in dmesh.name_2_tags[-2][name]]
    bcs+=fenicsx_utils.create_DirichletBC(dmesh.mesh,dmesh.facet_tags,V,[uD,uD],Dirichlet_bnd)
    # Periodic boundary condition
pbc = dolfinx_mpc_utils.PeriodicBoundary()
if bc_hor=="periodic":
    pbc.add_topological_condition(pbc_slave_tag_hor, pbc_rel_hor,
                                  slave_map=pbc_hor_is_slave,
                                  master_map=pbc_hor_is_master)
if bc_vert=="periodic":
    pbc.add_topological_condition(pbc_slave_tag_vert, pbc_rel_vert)
    # Variational problem
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
b = ufl.inner(u, v) * ufl.dx
a,b = dolfinx.fem.form(a), dolfinx.fem.form(b)
pbc.create_finalized_MultiPointConstraint(V, dmesh.facet_tags, bcs)
mpc = pbc.get_MultiPointConstraint()
A = dolfinx_mpc.assemble_matrix(a, mpc, bcs=bcs,diagval=1e2)
B = dolfinx_mpc.assemble_matrix(b, mpc, bcs=bcs,diagval=1e-2)
    # Solve GEP
EPS = SLEPc_utils.solve_GEP_shiftinvert(A,B,
                          problem_type=SLEPc.EPS.ProblemType.GHEP,
                          solver=SLEPc.EPS.Type.KRYLOVSCHUR,
                          nev=5,tol=1e-4,max_it=10,
                          target=1,shift=1,comm=comm)
(eigval,eigvec_r,eigvec_i) = SLEPc_utils.EPS_get_spectrum(EPS)
pbc.set_slave_dofs(eigvec_r)
pbc.set_slave_dofs(eigvec_i)
if comm.rank==0:
    ex_p = PDE_exact.Laplace_spectrum.interval(L[0], 50, bc_hor)
    ex_d = PDE_exact.Laplace_spectrum.interval(L[1], 10, bc_vert)
    ev_ex = np.sort([r+q for r in ex_p for q in ex_d])
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(np.real(eigval),np.imag(eigval),marker='o',linestyle='none',label=f'FEM N={V.dofmap.index_map.size_global}')
    ax.plot(np.real(ev_ex),np.imag(ev_ex),marker='x',linestyle='none',label='Exact')
    title=f"Rectangle: {bc_hor} at x=0,{L[0]} and {bc_vert} at y=0,{L[1]}"
    ax.set_title(title)
    ax.set_xlabel("$\Re(\lambda)$")
    ax.set_ylabel("$\Im(\lambda)$")
    ax.legend()
    ax.set_xlim([0,ev_ex[4]])
    fig.savefig(f"square_{bc_hor}-{bc_vert}_spectrum_fenicsx.png")
    # Export eigenfunctions to several .vtu file using pyvista
grid = fenicsx_utils.create_pyvista_UnstructuredGrid_from_FunctionSpace(V)
u_out = dolfinx.fem.Function(V)
for i in range(len(eigval)):
    name = f"{i:02d}_{eigval[i]:4.4g}"
        # eigvec_r[i] contains only the local dofs, not the ghost dofs
    u_out.x.petsc_vec.setArray(eigvec_r[i])
    u_out.x.scatter_forward()
    grid.point_data[name+"_ur"] = u_out.x.array.copy()
grid.save(f"square_{bc_hor}-{bc_vert}_fenicsx_{comm.rank}.vtu")