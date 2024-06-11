# This script is adapted from 'demo_periodic.py' provided with dolfinx_mpc.
# It solves the eigenvalue problem associated with TWO Poisson equations 
# using a VectorFunctionSpace. Its purpose is to demonstrate:
#   - Setting up Dirichlet b.c. on a component of a vector.
#   - Using dolfinx_mpc to set up periodic b.c. on a vector.
# Note: adding periodic b.c. on a component of a vector or on only part of
# a mixed function space may not be supported by dolfinx_mpc.
#
# The PDE solved is
#   - Delta(u) = lamda * u, -Delta(v) = lamda * v
# with essential boundary conditions
#   u=0 on Gamma_u, v=0 on Gamma_v
# and periodic boundary conditions on the remaining boundaries.
#%%
import dolfinx
import dolfinx.io
import dolfinx_mpc
import dolfinx_mpc.utils

import ufl
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
    # Eigenvalue problem
from slepc4py import SLEPc
from scientific_computing_utils import SLEPc_utils
from scientific_computing_utils import fenicsx_utils
    # Mesh
from scientific_computing_utils import gmsh_utils
from scientific_computing_utils import meshio_utils
    # Plot
import matplotlib.pyplot as plt
import pyvista as pv
from scientific_computing_utils import pyvista_utils
import os
DIR_MESH=os.path.join(os.path.dirname(os.path.abspath(__file__)),"mesh")

enable_pbc = True

#%% Load mesh
geofile=os.path.join(DIR_MESH,"Rectangle.geo")
gmshfile=os.path.join(DIR_MESH,"Rectangle.msh")
x_left = 0.0; x_right = 1.5
mesh_order = 2
gmsh_utils.generate_mesh_cli(geofile,gmshfile,2,refinement=0,binary=True,
                             parameters={'x_l':x_left,'x_r':x_right,
                                         'N_x': 50, 'N_y': 50},order=mesh_order)
gmsh_utils.print_summary(gmshfile)
mesh_cells_type = meshio_utils.get_cells_type_str(gmshfile)

dmesh = fenicsx_utils.DolfinxMesh.init_from_gmsh(gmshfile,2)
phys_tag_1D = gmsh_utils.getPhysicalNames(gmshfile,1)
phys_tag_2D = gmsh_utils.getPhysicalNames(gmshfile,2)
    # Dirichlet boundary for u  
Gamma_u = phys_tag_1D['Rectangle-Boundary-Top']
    # Dirichlet boundary for v
Gamma_v = phys_tag_1D['Rectangle-Boundary-Bot']
    # Periodic slave boundary
Periodic_slave_bnd = phys_tag_1D['Rectangle-Boundary-Right']
    # Slave boundary -> master boundary map
def periodic_relation(x):
    out_x = np.zeros(x.shape)
    out_x[0] = x_left + (x_right - x[0])
    out_x[1] = x[1]
    out_x[2] = x[2]
    return out_x
#%% Weak formulation
V = dolfinx.fem.VectorFunctionSpace(dmesh.mesh, ("CG", mesh_order))
    # Same dirichlet boundary condition on u and v
# uD = dolfinx.Function(V)
# uD.vector.setArray(0)
# uD.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
# bcs = fenicsx_utils.create_DirichletBC(dmesh.mesh,dmesh.facet_tags,V,[uD],Gamma_u)
    # Different Dirichlet boundary condiions on u and v
uD = dolfinx.fem.Function(V.sub(0).collapse()[0])
uD.vector.setArray(0)
uD.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
vD = dolfinx.fem.Function(V.sub(1).collapse()[0])
vD.vector.setArray(0)
vD.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
bcs = fenicsx_utils.create_DirichletBC(dmesh.mesh,dmesh.facet_tags,V,[uD,vD],[Gamma_u,Gamma_v],idx_sub_space=[0,1])
    # Periodic boundary condition through MultiPointContraint object
    # - Slave DoF are removed from the solved system
    # - Dirichlet bc cannot be on master DoF
    # - Multiple constraints with multiple calls to create_periodic_constraint()
if enable_pbc:
    mpc = dolfinx_mpc.MultiPointConstraint(V)
    mpc.create_periodic_constraint_topological(V, dmesh.facet_tags,
                                                Periodic_slave_bnd[0],
                                                periodic_relation,
                                                bcs,
                                                float(1.0))
    mpc.finalize()
    # Variational problem
u = ufl.TrialFunction(V)
phi = ufl.TestFunction(V)
a = ufl.inner(ufl.grad(u[0]), ufl.grad(phi[0])) * ufl.dx + ufl.inner(ufl.grad(u[1]), ufl.grad(phi[1])) * ufl.dx
b = ufl.inner(u, phi) * ufl.dx
a, b = dolfinx.fem.form(a), dolfinx.fem.form(b)
if enable_pbc:
    A = dolfinx_mpc.assemble_matrix(a, mpc, bcs=bcs,diagval=1e2)
    B = dolfinx_mpc.assemble_matrix(b, mpc, bcs=bcs,diagval=1e-2)
else:
    (A,B) = fenicsx_utils.assemble_GEP(a,b,bcs)

    # Solve GEP
EPS = SLEPc_utils.solve_GEP_shiftinvert(A,B,
                          problem_type=SLEPc.EPS.ProblemType.GHEP,
                          solver=SLEPc.EPS.Type.KRYLOVSCHUR,
                          nev=10,tol=1e-4,max_it=10,
                          target=1,shift=1)
(eigval,eigvec_r,eigvec_i) = SLEPc_utils.EPS_get_spectrum(EPS)
if enable_pbc:
    for i in range(len(eigval)):
        mpc.backsubstitution(eigvec_r[i])
        mpc.backsubstitution(eigvec_i[i])

#%% Plot spectrum
i_plot = 4
fig = plt.figure()
ax=fig.add_subplot(1,2,1)
ax.plot(np.real(eigval),np.imag(eigval),marker='o',linestyle='none')
ax.set_xlabel("$Re(\lambda)$")
ax.set_ylabel("$Im(\lambda)$")
#%% Export all eigenmodes using pyvista
# Assume isoparametric finite element
N = V.dofmap.index_map.size_global # Number of DoF per unknown
Nb = V.dofmap.index_map_bs # No. of unknowns
idx = lambda i: slice(i,Nb*N,Nb) # Indices of unknown no. i
grid = fenicsx_utils.create_pyvista_UnstructuredGrid_from_mesh(dmesh.mesh)
grid.clear_arrays()
for i in range(len(eigval)):
    name = f"{i:3d}_k_{eigval[i]:1.3e}_u"
    tmp = eigvec_r[i].array[idx(0)]  
    grid.point_data[name] = tmp/np.max(np.abs(tmp))
    name = f"{i:3d}_k_{eigval[i]:1.3e}_v"
    tmp = eigvec_r[i].array[idx(1)]
    grid.point_data[name] = tmp/np.max(np.abs(tmp))    
if enable_pbc:
    grid.save("result_with_pbc.vtu")
else:
    grid.save("result.vtu")