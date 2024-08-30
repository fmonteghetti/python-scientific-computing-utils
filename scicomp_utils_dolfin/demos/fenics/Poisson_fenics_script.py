#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script demonstrates the discretization of the Dirichlet Laplacian on a 
disk. Problem 1 is the Poisson equation
        - Delta u = f on |x|<R, u=0 at |x|=R.
We study the convergence of u w.r.t. to the number of DoF. This order is
impacted by the geometrical order of the finite elements.
Problem 2:
        - Delta u = lambda * u, u=0 on |r|=R.
"""
import numpy as np
import fenics as fe
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.sparse as sp
import petsc4py
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import Poisson_fenics as Poisson
#%% 1 -- Create mesh and define function space
nsteps=10
coordinate_degree=2 # 2 for isoparametric element
gdim = 2 # 2D
mesh = fe.UnitDiscMesh.create(fe.MPI.comm_world, nsteps, coordinate_degree, gdim)
#fe.plot(mesh)
print(f"Mesh of order {mesh.ufl_coordinate_element().degree()} with {mesh.num_vertices()} vertices and {mesh.coordinates().shape[0]} nodes")
V = fe.FunctionSpace(mesh, 'P', 2)
print(f"Finite element space has {V.tabulate_dof_coordinates().shape[0]} DoF")
    # Dirichlet boundary condition
u_D = fe.Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)
u_ex = lambda x,y : 1+x**2+2*(y**2)
def boundary(x, on_boundary):
    return on_boundary
bc = fe.DirichletBC(V, u_D, boundary)
    # Variational forms
u = fe.TrialFunction(V)
v = fe.TestFunction(V)
f = fe.Constant(-6.0)
a = fe.dot(fe.grad(u), fe.grad(v))*fe.dx
m = u*v*fe.dx
l = f*v*fe.dx
#%% 1a -- Solve and plot, the high-level way with dolfin functions
u_fe = fe.Function(V)
fe.solve(a == l,u_fe,bc,solver_parameters={"linear_solver": "lu"})
err = u_D.compute_vertex_values(mesh)-u_fe.compute_vertex_values(mesh)
err = np.sqrt(np.average(err**2))
print(f"L2 error (High-level): {err}")
fe.plot(u_fe)
#%% 1b -- Solve and plot, the low-level way with PETSc vectors
A = fe.PETScMatrix(); b = fe.PETScVector() # Dolfin wrapper around PETSc Mat/Vec
fe.assemble_system(a,l,bcs=bc,A_tensor=A,b_tensor=b)
x = fe.PETScVector()
fe.solve(A,x,b,"lu")
u_fe = x.vec().array # get PETSc.Vec then numpy array
    # Compute error
x_dof = V.tabulate_dof_coordinates()
err = np.sqrt(np.average((u_fe-u_ex(x_dof[:,0],x_dof[:,1]))**2))
print(f"L2 error (Low-level): {err}")
    # To plot x, an interpolation to a mesh that can be plotted is needed.
    # The standard way is to build a dolfin function, which
    # offer interpolation and export to standard formats.
        # (a) High-level way
fun = fe.Function(V,x) # build function from vector
fe.plot(fun)
        # (b) Low-level way
x_interp = fun.compute_vertex_values(mesh) # interpolate to mesh vertices
x_mesh = mesh.coordinates(); tri = mesh.cells() # vertices and triangles
Triang = mpl.tri.Triangulation(x_mesh[:,0], x_mesh[:,1],tri)
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
#ax.triplot(Triang,color='black') # mesh
ax.tricontourf(Triang,x_interp) # solution
ax.set_aspect('equal')
        # (c) Saving to external format
#vtkfile = fe.File('solution.pvd')
#vtkfile << fun
#%% 2 -- Convergence of direct problem
nsteps = [10,20,30,40,50,60,70,80,100]
V_fun = lambda mesh: fe.FunctionSpace(mesh, 'P', 2)
err_L2 = Poisson.direct_pb_compute_errors(nsteps,V_fun,2)
err_L2[:,0]=err_L2[:,0]/err_L2[0,0]; err_L2[:,1]=err_L2[:,1]/err_L2[0,1]
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.loglog(err_L2[:,0],err_L2[:,1],marker='x',label="FEM")
ax.set_xlabel('dof')
ax.set_ylabel('L2 error')
ax.set_title('Mesh convergence')
ax.grid('on')
ax.set_aspect('equal')
#%% 3 -- Solve sparse eigenvalue problem
    # Assemble PETSc matrices
A = fe.PETScMatrix(); M = fe.PETScMatrix()
fe.assemble(a,tensor=A); fe.assemble(m,tensor=M)
    # Enforce homogeneous Dirichlet boundary conditions
diagonal_value_A= 1e2; diagonal_value_M = 1e-2
bc.zero_columns(A,fe.Function(V).vector(),diagonal_value_A)
bc.zero_columns(M,fe.Function(V).vector(),diagonal_value_M)
#%% 3a -- Solve using dolfin wrapper around SLEPc
EPS = fe.SLEPcEigenSolver(A,M)
EPS.parameters["spectrum"] = "smallest magnitude"
EPS.parameters["solver"] = "arpack"
EPS.solve(5)
print(f"No. of converged eigenvalues: {EPS.get_number_converged()}")
    #%% Plot (using a dolfin function)
i_plot = 3
eigval_r,eigval_i, eigvec_r,eigvec_i = EPS.get_eigenpair(i_plot)
fun=fe.Function(V,eigvec_r)
fe.plot(fun,title=f"Eigenvalue {eigval_r+1j*eigval_i}")
#%% 3b -- Convert A and M to scipy sparse (csr) and call ARPACK
A_sp=Poisson.convert_PETSc_to_scipy(A.mat()); M_sp=Poisson.convert_PETSc_to_scipy(M.mat())
(eigval,eigvec) = sp.linalg.eigs(A_sp,M=M_sp,k=10,return_eigenvectors=True,which="SI")
    #%% Plot (using a dolfin function)
i_plot=6
tmp2 = petsc4py.PETSc.Vec().createWithArray(np.real(eigvec[:,i_plot]))
eigfun=fe.Function(V,fe.PETScVector(tmp2))
fe.plot(eigfun,title=f"Eigenvalue {eigval[i_plot]}")