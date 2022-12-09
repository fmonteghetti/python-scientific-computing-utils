#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solution of the heat equation under both its standard and mixed form. Lagrange
multipliers are used to enforce essential boundary conditions.

Standard formulation:
    du/dt = Delta(u) + f on Omega, u=u_D on Gamma_D, du/dn=u_N on Gamma_N.
    
Mixed formulation:
    du/dt = Div(sigma) + f, sigma = Grad(u) on Omega.

Block assembly done using multiphenics. Time-integration done using Assimulo.

"""

import numpy as np
import matplotlib.pyplot as plt
import fenics as fe
import gmsh_utils
import gmsh_utils_fenics
import fenics_utils
import meshio_utils
import multiphenics as mpfe
import multiphenics_utils
from petsc4py import PETSc
from mpi4py import MPI
from assimulo.solvers import IDA, Radau5DAE, GLIMDA
from assimulo.problem import Implicit_Problem
import scipy.sparse as sp
import os
DIR_MESH=os.path.join(os.path.dirname(os.path.abspath(__file__)),"mesh")
#%% Generate mesh using gmsh
geofile=os.path.join(DIR_MESH,"Annulus.geo")
gmshfile=os.path.join(DIR_MESH,"Annulus.msh")

gmsh_utils.generate_mesh_cli(geofile,gmshfile,2,refinement=1,log=1,\
                             parameters={'R':1,'R_TR':0.5},order=1,
                             gmshformat=2,binary=False)
gmsh_utils.print_summary(gmshfile)
meshio_utils.print_diagnostic_gmsh(gmshfile)
xdmfiles = meshio_utils.convert_gmsh_to_XDMF(gmshfile,prune_z=True)
dmesh = fenics_utils.DolfinMesh.init_from_xdmf(xdmfiles['triangle'],xdmfiles['line'])
gmsh_phys = meshio_utils.read_phys_gmsh(xdmfiles['gmsh_physical_entities'])
PhysName_1D_tag = gmsh_phys[0]; PhysName_2D_tag = gmsh_phys[1]
fe.plot(dmesh.mesh,title=f"{dmesh.mesh.num_vertices()} vertices")
coord = dmesh.mesh.coordinates()
plt.plot(coord[:,0],coord[:,1],marker='*',linestyle='none')
#%% Problem statement
u_D_expr = 'cos(10*x[0])'
#u_D_expr = 'cos(10*atan(x[1]/x[0]))'
u_D_expr = '1'
#u_N_expr = 'cos(10*atan(x[1]/x[0]))'
u_N_expr = '1'
f_expr = 'sin(3*x[0] + 1)*sin(3*x[1] + 1)'
f_expr = '1'
Dirichlet_bnd = PhysName_1D_tag['Gamma-R'][0]
Neumann_bnd = PhysName_1D_tag['Gamma-RTR'][0]
#%% 1 -- Standard Formulation Lagrange multiplier for Dirichlet b.c. using multiphenics
# Assimulo
# Identifying algebraic variables is not necessary for this case

class DAE_description(Implicit_Problem):
        # default constructor
    def __init__(self,W,u_D,u_N,f,dx,ds_D,ds_N,name):
        self.W = W
        self.u_D = u_D
        self.u_N = u_N
        self.f = f
        self.dx = dx
        self.ds_D = ds_D
        self.ds_N = ds_N
            # Algebraic variable
        idx_l = W.block_dofmap().block_owned_dofs__global_numbering(1)
        idx = np.ones((W.dim(),),dtype="bool")
        idx[idx_l] = False
        self.algvar = idx
        self.name = name
            # Misc        
        self.res_count = 0
        self.jac_count = 0
        
    def set_initial_data(self,y0,yd0):
        self.y0 = y0
        self.yd0 = yd0
        
    #Defines the residual
    # TODO Define assemble_residual functions avoid code duplication
    # TODO Time dependent source terms for smooth ramp up
    # Add class members for speed-up
    def res(self,t,y,yd):
        # y, yd: np.array
        self.res_count += 1
            # numpy vector to BlockPETScVector
        y_bvec=mpfe.la.BlockPETScVector(PETSc.Vec().createWithArray(y))
        yd_bvec=mpfe.la.BlockPETScVector(PETSc.Vec().createWithArray(yd))    
        
        (u, l) = mpfe.block_split(mpfe.BlockFunction(self.W,y_bvec))
        (ud, ld) = mpfe.block_split(mpfe.BlockFunction(self.W,yd_bvec))
    
        (v, m) = mpfe.block_split(mpfe.BlockTestFunction(self.W))
    
        R_bvec = mpfe.block_assemble( [ud*v*self.dx + fe.inner(fe.grad(u), fe.grad(v))*self.dx + \
               -f*v*self.dx - l*v*self.ds_D -u_N*v*self.ds_N,
              u*m*self.ds_D - u_D*m*self.ds_D] )
        return R_bvec.vec().array.flatten()
    
    #Defines the Jacobian*vector product
    # def jacv(self,t,y,yd,res,z,c):
    #     self.jac_count += 1
    #     z_bvec=mpfe.la.BlockPETScVector(PETSc.Vec().createWithArray(z))
    #     (u, l) = mpfe.block_split(mpfe.BlockFunction(self.W,z_bvec))
    #     (v, m) = mpfe.block_split(mpfe.BlockTestFunction(self.W))
    #     ud = c*u
    #     dr_bvec = mpfe.block_assemble( [ud*v*self.dx + fe.inner(fe.grad(u), fe.grad(v))*self.dx + \
    #             - l*v*self.ds_D,
    #           u*m*self.ds_D] )
    #     return dr_bvec.vec().array.flatten()
    
        #Assemble the Jacobian matrix
    def jac(self,c,t,y,yd):
        self.jac_count += 1
        (u, l) = mpfe.block_split(mpfe.BlockTrialFunction(self.W))
        (v, m) = mpfe.block_split(mpfe.BlockTestFunction(self.W))
        ud = c*u
        dr = mpfe.block_assemble([[ud*v*self.dx + fe.inner(fe.grad(u), fe.grad(v))*self.dx, - l*v*self.ds_D],
              [u*m*self.ds_D                    , 0     ]] )        
        
        (indptr,indices,val) = dr.mat().getValuesCSR()
        A_scipy = sp.csr_matrix((val, indices, indptr), shape=dr.mat().size)

        return A_scipy.todense()
 

    # Mesh resriction on selected boundary only
boundary_restriction =  multiphenics_utils.build_MeshRestriction_from_tags(dmesh.mesh,
                                                dmesh.boundaries,
                                                [Dirichlet_bnd])
V = fe.FunctionSpace(dmesh.mesh, "Lagrange", 1)
W = mpfe.BlockFunctionSpace([V, V], restrict=[None, boundary_restriction])

u_D = fe.Expression(u_D_expr, element=V.ufl_element())
u_N = fe.Expression(u_N_expr, element=V.ufl_element())
f = fe.Expression(f_expr, element=V.ufl_element())

imp_mod = DAE_description(W,u_D,u_N,f,dmesh.dx,dmesh.ds(Dirichlet_bnd),dmesh.ds(Neumann_bnd),"Heat Matrix-free (IDA)")
#% Time integration (Assimulo)
y0 = np.zeros((W.dim(),))
yd0 = np.zeros(W.dim())
imp_mod.set_initial_data(y0, yd0)
imp_sim = IDA(imp_mod)
    # Solver
#imp_sim.linear_solver = 'SPGMR'
#imp_sim.atol = 1e-6 #Default 1e-6
#imp_sim.rtol = 1e-6 #Default 1e-6
#imp_sim.inith = 1e-1
    # Consistent initialization
imp_sim.make_consistent("IDA_YA_YDP_INIT")
imp_sim.tout1=1e-3
print(f"Residual call: {imp_sim.problem.res_count}, Jacobian call: {imp_sim.problem.jac_count}")
    # Misc
imp_sim.suppress_alg = False
imp_sim.suppress_sens = True
    # Display
imp_sim.report_continuously = True
imp_sim.display_progress = True
    # Time integration
tf = 1
imp_sim.maxh = tf/1000
imp_sim.maxsteps = int(1e2)
t, y, yd = imp_sim.simulate(tf, 10)
#%% Plot
i_p = np.min([5,len(t)-1])
x_petsc = PETSc.Vec().createWithArray(y[i_p,:])
u_blockVec=mpfe.la.BlockPETScVector(x_petsc)
ul = mpfe.BlockFunction(W,u_blockVec)
(u, l) = mpfe.block_split(ul)

c = fe.plot(u)
ax=c.axes
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.colorbar(c)