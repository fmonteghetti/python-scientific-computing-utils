#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basics of solving the Heat equation with homogeneous Neumann boundary condition

    du/dt=Delta(u)+f in Omega, du/dn=0 on Gamma, u(t=0,x)=u0(x).

The script can be run on one or multiple threads:
    python ./MPI-test.py (one thread)
    mpirun -n N ./MPI-test.py (N threads).
The latter can be executed from the ipython prompt using:
    ! mpirun -n N ./MPI-test.py.

The following features are demonstrated:
- Generate high-order mesh using gmsh.
- Generate a distributed dolfinx.mesh.Mesh from a .msh file.
- Time-integration using PETSc.TS
"""
#%%
from mpi4py import MPI
comm = MPI.COMM_WORLD # dolfinx, petsc, and slepc use mpi communicators
from petsc4py import PETSc
from slepc4py import SLEPc
import pyvista as pv
import numpy as np
import time
import dolfinx
import ufl
from scicomp_utils_mesh import gmsh_utils
from scicomp_utils_dolfinx import gmsh_utils_fenicsx
from scicomp_utils_dolfinx import fenicsx_utils
from scicomp_utils_misc import SLEPc_utils
from scicomp_utils_misc import PETSc_utils
import os
DIR_MESH=os.path.join(os.path.dirname(os.path.abspath(__file__)),"mesh")
degree_mesh = 2
degree_fem = 2
dim = 2
geofile=os.path.join(DIR_MESH,"Disc.geo")
Gamma_name = ['Gamma-R'] # physical name of boundary
Omega_name = ['Omega'] # physical name of domain
u0_expr = lambda x: np.cos(x[0]+x[1]) # initial condition
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

    # == Weak formulation m(du/dt,v) + a(u,v) = l(u,v) with v in H^1
    # Build distributed function space
t0=time.process_time_ns()
V = dolfinx.fem.FunctionSpace(mesh, ("CG", degree_fem))
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
x = ufl.SpatialCoordinate(dmesh.mesh)
u0 = dolfinx.fem.Function(V)
u0.interpolate(u0_expr)
u0.x.scatter_forward() # Update all ghost values from local values
# bcs = fenicsx_utils.create_DirichletBC(dmesh.mesh,dmesh.facet_tags,V,[uD],Gamma_tags)
xc, sigma, A = [0.0,0.0], 0.02, 10
f = A * ufl.exp(-((x[0] - xc[0]) ** 2 + (x[1] - xc[1]) ** 2) / sigma)
m = ufl.inner(u, v) * dmesh.dx
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dmesh.dx
l = ufl.inner(f, v) * dmesh.dx
# Assemble as M*dU/dt + K*U = F
M = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(m),[])
M.assemble()
K = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(a),[])
K.assemble()
F = dolfinx.fem.petsc.assemble_vector(dolfinx.fem.form(l))
F.assemble()
# Create DAE
class Heat_equation(PETSc_utils.Fully_implicit_DAE):
    """ Solve F(t,y,yd) = 0, with F(t,y,yd) = J_yd*yd + J_y*y + F_src."""
    def define_equation(self,J_yd,J_y,F_src):
        """ J_yd, J_y: PETSc.Mat, F_src: PETSc.Vec"""
        self.J_yd = J_yd
        self.J_y = J_y
        self.F_src = F_src
        # Required by parent class
        self.F = self.J_yd.createVecRight()
        self.J = self.J_y.duplicate() # c*dF/dyd + dF/dy
        # Allocate temporary vector
        self.F_tmp = self.J_yd.createVecRight()

    def IFunction(self, ts, t, y, yd, F):
        """ Evaluate residual vector F(t,y,yd)."""
        # F = self.J_yd * yd + self.J_y * y + F_src
        self.J_yd.mult(yd,self.F_tmp)
        self.J_y.multAdd(y,self.F_tmp,F)
        F.axpy(1,self.F_src)
        
    def IJacobian(self, ts, t, y, yd, c, Amat, Pmat):
        """ Evaluate jacobian matrix J = c*J_yd + J_y."""
        # Pmat = c*J_yd + J_y
        self.J_y.copy(Pmat)
        Pmat.axpy(c,self.J_yd)
        if Amat != Pmat:
            print("Operator different from preconditioning")
            Amat.assemble()

dae = Heat_equation(F.size,"heat")
dae.define_equation(M,K,-F)
if comm.rank ==0:
    print(f"Elapsed time (direct problem): {1e-9*(time.process_time_ns()-t0):1.2g}s")
# Time integration
    # Time parameters
tf = 5
dt = tf/100
x0 = dae.get_vector()
x0 = u0.vector
    # Solver options
OptDB = PETSc_utils.get_cleared_options_db()
PETSc.Sys.pushErrorHandler("python")
OptDB["ts_equation_type"] = PETSc.TS.EquationType.IMPLICIT
    # Numerical scheme
OptDB["ts_type"] = "beuler" # beuler, cn
        # BDF
# OptDB["ts_type"] = "bdf"
# OptDB["bdf_order"] = 4
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
Integration_params = {'cinit': False,'dt_export':0.5*dt}
PETSc_utils.TS_integration_dae(dae,x0,dt,tf,**Integration_params)
tp = dae.history['t']
yp = dae.history['y']
# Export to paraview
uh = dolfinx.fem.Function(V)
uh.name = "uh"
# uh.interpolate(initial_condition)
with dolfinx.io.XDMFFile(mesh.comm, "demo_heat_equation.xdmf", "w") as file:
    file.write_mesh(dmesh.mesh)
    for (i,y) in enumerate(yp):
        uh.vector.array = y.array
        file.write_function(uh,tp[i])
