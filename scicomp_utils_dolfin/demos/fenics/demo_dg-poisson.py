#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is adapted from demo_dg-poisson.py from fenics repository.
It solves Poisson's equation under both primal and mixed form.

The primal form is

    - div(A * grad u) = f
    u = uD on Gamma_D, dot(A*grad(u),n) = phiN  on Gamma_N,
    
and the mixed form is

    - div(phi) = f, phi = A*grad(u),
    u = uD on Gamma_D, dot(phi,n) = phiN  on Gamma_N.

The purpose of this script is to demonstrate several non-conforming
discretizations. The weak formulations are justified in the companion
document Laplace-equation-discretization.lyx.
"""

# TODO = Normalizing function from mixed space in fenics?
# TODO = Use imported gmsh

import fenics as fe
from fenics import dot, grad, jump, avg
import matplotlib.pyplot as plt
from scicomp_utils_dolfin import fenics_utils
import numpy as np

    # FIXME: Make mesh ghosted
fe.parameters["ghost_mode"] = "shared_facet"
    # Create mesh and define function space
mesh = fe.UnitSquareMesh(20, 20)
def DirichletBoundary_fun(x,on_boundary):
    return on_boundary and fe.near(x[0]*(1 - x[0]), 0)
class DirichletBoundary(fe.SubDomain):
  def inside(self, x, on_boundary):
      return DirichletBoundary_fun(x,on_boundary)
class NeumanBoundary(fe.SubDomain):
  def inside(self, x, on_boundary):
    return on_boundary and fe.near(x[1]*(1 - x[1]), 0)
    # Mark facets of the mesh
boundaries = fe.MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
Gamma_D, Gamma_N = 3, 8
NeumanBoundary().mark(boundaries, Gamma_N)
DirichletBoundary().mark(boundaries, Gamma_D)

dx = fe.dx # domain measure
dS = fe.dS # interior surface measure
    # boundary surface measure aware of Dirichlet and Neumann boundaries
ds = fe.Measure('ds', domain=mesh, subdomain_data=boundaries)

p = 2 # polynomial degree
    # Define normal vector and mesh size
n = fe.FacetNormal(mesh)
h = fe.CellDiameter(mesh)
h_avg = (h('+') + h('-'))/2

# Define the source term f, Dirichlet term uD and Neumann term g
f = fe.Expression('-100.0*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)', degree=2)
uD = fe.Expression('x[0] + 0.25*sin(2*pi*x[1])', degree=2)
phiN = fe.Expression('(x[1] - 0.5)*(x[1] - 0.5)', degree=2)
A = fe.as_matrix([[1, 0], [0, 3]])
Ainv = fe.as_matrix([[1, 0], [0, 1/3]])

# problem = "direct"
problem = "eigenvalue"


form = "Lagrange"
form = "Crouzeix-Raviart"
form = "SIP"
form = "BZ-Primal"
form = "BZ-Mixed"
form = "Local-DG"

# Define variational problem
    # H^1-conformal approximation
if form=="Lagrange":
    V = fe.FunctionSpace(mesh, 'Lagrange', p)  
    bc = [fe.DirichletBC(V, uD, DirichletBoundary_fun)]
    u, v = fe.TrialFunction(V), fe.TestFunction(V)
    a = dot(A*grad(u), grad(v))*fe.dx
    L = f*v*dx + phiN*v*ds(Gamma_N)
    b = u*v*dx
    # H^1 non-conformal approximation
elif form=="Crouzeix-Raviart":
    V = fe.FunctionSpace(mesh, 'CR', 1)
    bc = [fe.DirichletBC(V, uD, DirichletBoundary_fun)]
    u, v = fe.TrialFunction(V), fe.TestFunction(V)
    a = dot(A*grad(u), grad(v))*dx
    L = f*v*dx + phiN*v*ds(Gamma_N)
    b = u*v*dx
    # Symetric Interior Penalty
    # If eta_b=0, then Dirichlet b.c. is never cleam
elif form=="SIP":
    eta, eta_b = 10, 10
    bc = []
    V = fe.FunctionSpace(mesh, 'DG', p)
    u, v = fe.TrialFunction(V), fe.TestFunction(V)
    a_list=[]
    a_list.append(
        dot(grad(u), A*grad(v))*dx \
       - dot(jump(u, n),avg(A*grad(v)))*dS \
       - dot(avg(grad(u)),jump(v, n))*dS
   )
    if eta!=0:
        a_list.append(eta/h_avg*dot(jump(u, n),jump(v, n))*dS)
    a_list.append(   
        - dot(u*n,A*grad(v))*ds(Gamma_D) \
        - dot(A*grad(u),v*n)*ds(Gamma_D) \
        + (eta_b/h)*u*v*ds(Gamma_D)
    )
    a = sum(a_list)
    L = f*v*dx - uD*dot(A*grad(v), n)*ds(Gamma_D) + (eta_b/h)*uD*v*ds(Gamma_D) + phiN*v*ds(Gamma_N)
    b = u*v*dx
    # Local DG
    # Penalization not really needed at all
elif form=="Local-DG":
    eta, eta_b = 0, 0
    beta = fe.as_vector([0,0])
    beta = n('+')/2
    bc = []
    DGv = fe.VectorElement("DG", mesh.ufl_cell(), degree=p,dim=2)
    DG  = fe.FiniteElement("DG", mesh.ufl_cell(), degree=p)
    V = fe.FunctionSpace(mesh, DGv * DG)
    (phi, u) = fe.TrialFunctions(V)
    (psi, v) = fe.TestFunctions(V)
    a_list = []
    a_list.append( \
        dot(Ainv*phi, psi)*dx \
       + dot(phi,grad(v))*dx - dot(grad(u),psi) *dx \
       + dot(jump(u, n),avg(psi))*dS - dot(avg(phi),jump(v, n))*dS \
    )
    if eta!=0:
        a_list.append((eta/h_avg)*dot(jump(u, n),jump(v, n))*dS)
    a_list.append(+ dot(u*n,psi)*ds(Gamma_D) - dot(phi,v*n)*ds(Gamma_D) + (eta_b/h)*u*v*ds(Gamma_D))
    if beta[0]!=0 or beta[1]!=0:
        a_list.append(dot(jump(u,n),beta*jump(psi,n))*dS - dot(beta*jump(phi,n),jump(v,n))*dS)
    a = sum(a_list)
    L = f*v*dx + uD*dot(psi, n)*ds(Gamma_D) + (eta_b/h)*uD*v*ds(Gamma_D) + phiN*v*ds(Gamma_N)    
    b = u*v*dx
    # BZ primal
    # Both supaer-penalizations are necessary
elif form=="BZ-Primal":
    eta, eta_b = 1, 1
    bc = []
    spen = lambda h: h**(-2*p) # super-penalization
    V = fe.FunctionSpace(mesh, 'DG', p)
    u, v = fe.TrialFunction(V), fe.TestFunction(V)
    a_list = []
    a_list.append(dot(grad(u), A*grad(v))*dx)
    if eta!=0:
        a_list.append(spen(h_avg)*eta/h_avg*dot(jump(u, n),jump(v, n))*dS)
    a_list.append(spen(h)*(eta_b/h)*u*v*ds(Gamma_D))
    a = sum(a_list)
    L = f*v*dx + spen(h)*(eta_b/h)*uD*v*ds(Gamma_D) + phiN*v*ds(Gamma_N)
    b = u*v*dx
    # BZ Mixed
    # Both supaer-penalizations are necessary
elif form=="BZ-Mixed":
    eta, eta_b = 0.1, 0.1
    bc = []
    spen = lambda h: h**(-2*p) # super-penalization
    DGv = fe.VectorElement("DG", mesh.ufl_cell(), degree=p,dim=2)
    DG  = fe.FiniteElement("DG", mesh.ufl_cell(), degree=p)
    V = fe.FunctionSpace(mesh, DGv * DG)
    (phi, u) = fe.TrialFunctions(V)
    (psi, v) = fe.TestFunctions(V)
    a_list = [dot(Ainv*phi, psi)*dx]
    a_list.append(dot(phi,grad(v))*dx - dot(grad(u),psi) *dx)
    if eta!=0:
        a_list.append(spen(h_avg)*eta/h_avg*dot(jump(u, n),jump(v, n))*dS)
        a_list.append(spen(h)*(eta_b/h)*u*v*ds(Gamma_D))    
    a = sum(a_list)
    L = f*v*dx + spen(h)*(eta_b/h)*uD*v*ds(Gamma_D) + phiN*v*ds(Gamma_N)
    b = u*v*dx
else:
    raise ValueError(f"Unknown formulation '{form}'")


if problem=="direct":
    U = fe.Function(V)
    fe.solve(a == L, U,bc)
    print("Solution vector norm (0): {!r}".format(U.vector().norm("l2")))
    
    if form=="SIP" or form=="BZ-Primal" or form=="Lagrange" or form=="Crouzeix-Raviart":
        u = U
    elif form=="BZ-Mixed" or form=="Local-DG":
        (phi, u) = U.split()
    
    f = plt.figure()
    ax=f.add_subplot(2,1,1)
    c=fe.plot(u,axes=ax,title=form+f" (N={U.vector().size()})",mode='color')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(c)
elif problem=="eigenvalue":
    A = fe.PETScMatrix(); M = fe.PETScMatrix()
    fenics_utils.assemble_GEP(A,a,M,b,bc,diag_A=1e2,diag_B=1e-2)
    from slepc4py import SLEPc
    from scientific_computing_utils import SLEPc_utils
    A_petsc = A.mat(); M_petsc = M.mat()
    SLEPc_params = {'nev': 20,
                  'target': 10, 'shift': 10,
                  'problem_type': SLEPc.EPS.ProblemType.GHEP,
                  'solver': SLEPc.EPS.Type.KRYLOVSCHUR,
                  'tol': 1e-6,
                  'max_it': 100}
    EPS = SLEPc_utils.solve_GEP_shiftinvert(A_petsc,M_petsc,**SLEPc_params)
    (eigval,eigvec_r,eigvec_i) = SLEPc_utils.EPS_get_spectrum(EPS)
    
    def get_eigenvec(i):
        return fe.PETScVector(eigvec_r[i]),fe.PETScVector(eigvec_i[i])

    f = plt.figure()
    ax=f.add_subplot(1,1,1)
    ax.plot(np.real(eigval),np.imag(eigval),linestyle='none',marker='o')
    ax.set_xlabel(r"$\Re(\lambda)$")
    ax.set_ylabel(r"$\Im(\lambda)$")
    ax.set_xlim([0,100])
    ax.set_title(form+f" (N={eigvec_r[0].size})")

    # Plot (using a dolfin function) (2D only)
    i_plot = 1
    eigvec_r_p,eigvec_i_p = get_eigenvec(i_plot)
    
    if form=="SIP" or form=="BZ-Primal" or form=="Lagrange" or form=="Crouzeix-Raviart":
        fun=fe.Function(V,eigvec_r_p/eigvec_r_p.norm('linf'),name=f"l={eigval[i_plot]}")
        fun_u = fun
    elif form=="BZ-Mixed" or form=="Local-DG":
        fun=fe.Function(V,eigvec_r_p,name=f"l={eigval[i_plot]}")
        (fun_phi, fun_u) = fun.split()
        fun_u.vector()[:] = fun_u.vector()[:]/fun_u.vector().norm('linf')
    
    # f = plt.figure()
    # ax=f.add_subplot(1,1,1)
    # c=fe.plot(fun_u,axes=ax,title=f"$\lambda=${eigval[i_plot]:2.2g}\n"+form+f" (N={eigvec_r[0].size})",
    #           mode='color')
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # plt.colorbar(c)
        # Paraview export
    xdmfile = "solution.xdmf"
    output = fe.XDMFFile(xdmfile)
    output.parameters["flush_output"] = True
    output.parameters["rewrite_function_mesh"] = False
    output.parameters["functions_share_mesh"] = True
    print(f"Export to {xdmfile}...")
    for i in range(len(eigvec_r)):
        fun = fe.Function(V,get_eigenvec(i)[0],name=f"{eigval[i]}")
        fun_u = fun.split()[1]
        output.write(fun_u, 0)