#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Poisson equation using fenics.
"""

import numpy as np
from numpy import pi
import fenics as fe
import scipy.sparse as sp

def direct_pb_compute_errors(nsteps_list,V_fun,mesh_deg):
    err = np.zeros((len(nsteps_list),2))
    for i in range(len(nsteps_list)):
        # Build mesh
        nsteps=nsteps_list[i]
        coordinate_degree=mesh_deg # 2 for isoparametric element
        gdim = 2 # 2D
        mesh = fe.UnitDiscMesh.create(fe.MPI.comm_world, nsteps, coordinate_degree, gdim)
        #fe.plot(mesh)
        print(f"Mesh of order {mesh.ufl_coordinate_element().degree()} with {mesh.num_vertices()} vertices and {mesh.coordinates().shape[0]} nodes")
        # Build function space
        V = V_fun(mesh)
        print(f"Finite element space has {V.tabulate_dof_coordinates().shape[0]} DoF")
        # Dirichlet boundary condition
        u_D = fe.Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)
        u_ex = lambda x,y : 1+x**2+2*(y**2)
        def boundary(x, on_boundary):
            return on_boundary
        bc = fe.DirichletBC(V, u_D, boundary)
        # Variational form a(u,v)=l(v)        
        u = fe.TrialFunction(V)
        v = fe.TestFunction(V)
        f = fe.Constant(-6.0)
        a = fe.dot(fe.grad(u), fe.grad(v))*fe.dx
        l = f*v*fe.dx
        # Assemble
        A = fe.PETScMatrix()
        b = fe.PETScVector()
        fe.assemble_system(a,l,bcs=bc,A_tensor=A,b_tensor=b)
        x = fe.PETScVector()
            # Solve
        fe.solve(A,x,b,"lu")
        u_fe = x.vec().array # get PETSc.Vec then numpy array
            # Alternative
#        u_fe = fe.Function(V)
#        u_fe=fe.Function(V,x)        
            # Compute error
        dof = V.tabulate_dof_coordinates();
#        err_L2 = fe.errornorm(u_D, u_fe, 'L2')
        err_L2 = np.sqrt(np.average((u_fe-u_ex(dof[:,0],dof[:,1]))**2))
        print(f"L2 error is {err_L2}")
        err[i,0] = dof.shape[0]
        err[i,1] = err_L2
    
    return err

def convert_PETSc_to_scipy(A):
    """
    Convert from PETSc.Mat to scipy sparse (csr).

    Parameters
    ----------
    A : PETSc.Mat


    Returns
    -------
    A_scipy : scipy.sparse.csr.csr_matrix
    """
    (indptr,indices,val) = A.getValuesCSR()
    A_scipy = sp.csr_matrix((val, indices, indptr), shape=A.size)
    return A_scipy
