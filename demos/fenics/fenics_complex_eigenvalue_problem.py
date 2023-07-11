#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This scripts demonstrates the solution of a generalized eigenvalue problem
coming from fenics with complex coefficients but real arithmetic.
An extended real-valued formulation is solved by separating real and
and imaginary parts. Since the spectrum of the real formulation is stable
by conjugation, there is a need to filter the spectrum to remove
spurious eigenvalue.

The PDE solved is
    - alpha*Delta(u) = lambda * beta * u (with Dirichlet boundary conditions),
whose exact solution is
    lambda = (alpha/beta)*(km**2),
where km are the zeros of the m-th order Bessel function of the first kind Jm.

"""

import numpy as np
import scipy.special
import fenics as fe
from scientific_computing_utils import fenics_utils
import matplotlib.pyplot as plt
from scientific_computing_utils import SLEPc_utils
from slepc4py import SLEPc
from petsc4py import PETSc
#%% 
    # Complex coefficients
alpha = 1.0 + 1j*1
beta = alpha**2
    # mesh unit circle
dim = 2
geom_order = 1 # (alledgedly)
mesh = fe.UnitDiscMesh.create(fe.MPI.comm_world,40,geom_order,dim)
#fe.plot(mesh)
    # Define function space over mixed element
P1 = fe.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
V = fe.FunctionSpace(mesh,fe.MixedElement((P1,P1)))
Vr = V.sub(0); Vi = V.sub(1)
    # Dirichlet boundary condition
u_D = fe.Constant(0.0)
def boundary(x, on_boundary):
    return on_boundary
bcs = [fe.DirichletBC(V.sub(0), u_D, boundary),
       fe.DirichletBC(V.sub(1), u_D, boundary)]
    # Variational forms
(ur,ui) = fe.TrialFunction(V)
(vr,vi) = fe.TestFunction(V)
alpha_r = fe.Constant(np.real(alpha)); alpha_i = fe.Constant(np.imag(alpha))
beta_r = fe.Constant(np.real(beta)); beta_i = fe.Constant(np.imag(beta))
a_r = lambda u,v : alpha_r*fe.dot(fe.grad(u), fe.grad(v))*fe.dx
a_i = lambda u,v : alpha_i*fe.dot(fe.grad(u), fe.grad(v))*fe.dx
b_r = lambda u,v : beta_r*fe.dot(u, v)*fe.dx
b_i = lambda u,v : beta_i*fe.dot(u, v)*fe.dx

a = a_r(ur,vr) - a_i(ui,vr) + a_i(ur,vi) + a_r(ui,vi)
b = b_r(ur,vr) - b_i(ui,vr) + b_i(ur,vi) + b_r(ui,vi)
    # DoF indices of real and imaginary parts
idx_ur = np.array(Vr.dofmap().dofs())
idx_ui = np.array(Vi.dofmap().dofs())
#%% Assemble and solve sparse eigenvalue problem
A = fe.PETScMatrix(); B = fe.PETScMatrix()
fenics_utils.assemble_GEP(A,a,B,b,bcs,diag_A=1e2,diag_B=1e-2)
EPS = SLEPc_utils.solve_GEP_shiftinvert(A.mat(),B.mat(),
                          problem_type=SLEPc.EPS.ProblemType.GNHEP,
                          solver=SLEPc.EPS.Type.KRYLOVSCHUR,
                          nev=50,tol=1e-6,max_it=100,
                          target=-1.0,shift=1.0)
#%% Filter spectrum to remove spurious eigenvalue: in SLEPc
(eigval,eigvec)=SLEPc_utils.EPS_get_spectrum_ReImFormulation(EPS,
                              idx_ur.astype(np.int32),idx_ui.astype(np.int32))
#%% Filter spectrum to remove spurious eigenvalue: in fenics
# Sorting spurious eigenvalues can also be done in fenics directly,
# by using the criterion imag(ur)=real(ui).
def get_spectrum_ReImFormulation_fenics(EPS,Vr,Vi,mesh,tol=1e-10):
    vr = EPS.getOperators()[0].createVecRight()
    vi = EPS.getOperators()[0].createVecRight()
    eigval = list(); eigvec = list()
    for i in range(EPS.getConverged()):
        val = EPS.getEigenpair(i, vr, vi)        
        # convert petsc4py.PETSc.Vec to dolfin.cpp.la.PETScVector
        eigvec_r = fe.PETScVector(vr); eigvec_i = fe.PETScVector(vi)
        ur_r = fe.Function(Vr,eigvec_r); ur_i = fe.Function(Vr,eigvec_i)
        ui_r = fe.Function(Vi,eigvec_r); ui_i = fe.Function(Vi,eigvec_i)
        delta=ur_i.compute_vertex_values(mesh)-ui_r.compute_vertex_values(mesh)
        print(np.linalg.norm(delta))
        if np.linalg.norm(delta)<tol:
            eigval.append(val)         
            eigvec.append(vr.copy())
    return (eigval, eigvec)

(eigval,eigvec)=get_spectrum_ReImFormulation_fenics(EPS,Vr,Vi,mesh,tol=1e-5)
#%% Plot eigenfunction (using a dolfin function)
i_plot = 4
ur = fe.Function(Vr,fe.PETScVector(eigvec[i_plot]))
ui = fe.Function(Vi,fe.PETScVector(eigvec[i_plot]))
f = plt.figure()
ax=f.add_subplot(2,1,1)
coll=fe.plot(ur,title="$\Re(u)$"+f"\n{eigval[i_plot].real:1.2g}+{eigval[i_plot].imag:1.2g}*j",cmap='jet')
ax.grid(False)
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.colorbar(coll)
ax=f.add_subplot(2,1,2)
coll=fe.plot(ui,title="$\Im(u)$",cmap='jet')
ax.grid(False)
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.colorbar(coll)
#%% Plot spectrum
eigval_ex = (alpha/beta) * np.hstack([scipy.special.jn_zeros(m,4) for m in range(5)])**2
f = plt.figure()
ax=f.add_subplot(1,1,1)
ax.plot(np.real(eigval_ex),np.imag(eigval_ex),linestyle='none',marker='o',label='Exact')
ax.plot(np.real(eigval),np.imag(eigval),linestyle='none',marker='x',label='FEM w/ real formulation')
ax.legend()
ax.set_xlim([0,50])
ax.set_ylim([-40,40])
#ax.set_ylim([0,30])
#ax.set_aspect('equal') # orthonormal axis
ax.set_xlabel("$\Re(\lambda)$")
ax.set_ylabel("$\Im(\lambda)$")
ax.set_title("Spectrum")

