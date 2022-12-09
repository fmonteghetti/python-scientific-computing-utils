# -*- coding: utf-8 -*-
"""
This scipt demonstrates different techniques to enforce Dirichlet
boundary conditions with Fenics. Fenics never eliminates DoF 
associated with essential boundary conditions, but modifies the matrices.
Three different techniques are shown to solve a direct problem
    A*x = b,
and two techniques are shown to deal with eigenvalue problems
    A*x = lambda*x as well as A*x = lambda*B*x.
"""
import numpy as np
import fenics as fe
np.set_printoptions(precision=2)
#%% Setup: 1D problem with u(x)=u_D(x) at x=0 and x=1
# Direct problem A x = b
N = 50 # number of elements
mesh = fe.IntervalMesh(N, 0,1)
V = fe.FunctionSpace(mesh, 'P', 1)
    # Dirichlet boundary condition
u_D = fe.Constant(9.987) # 0.0 for homogeneous b.c.
def boundary(x, on_boundary):
    return on_boundary
bc = fe.DirichletBC(V, u_D, boundary)
print(f"Mesh with {mesh.num_vertices()} vertices:\n{mesh.coordinates().transpose()}")
print(f"Dirichlet boundary conditions:\n{bc.get_boundary_values()}")
    # Variational problem a(u,v)=l(v)
u = fe.TrialFunction(V)
v = fe.TestFunction(V)
f = fe.Constant(-6.0)
a = fe.dot(fe.grad(u), fe.grad(v))*fe.dx
l = f*v*fe.dx
    # Reference case
    # Assemble A and b without essential boundary conditions.
A_n = fe.PETScMatrix()
b_n = fe.PETScVector()
fe.assemble(a, tensor=A_n)
fe.assemble(l, tensor=b_n)
print(f"=== Without essential boundary condition:\nA=\n{A_n.array()}\nb = {b_n.vec().array}\n===")
#%% 1 -- Enforce boundary conditions with 'bc.apply()'
    # Diagonal entries A[i_bc,i_bc] are set to 1.
    # Rows A[i_bc,:] are set to 0.
    # Columns A[:,j_bc] are untouched, to handle inhomogeneous Dirichlet
    # boundary conditions
    # -> This implies that the symmetry of A is broken
A=A_n.copy(); bc.apply(A)
    # b[i_bc] is to the desired boundary value
b=b_n.copy(); bc.apply(b)
print(f"=== 1) Using bc.apply():\nA=\n{A.array()}\nb = {b.vec().array}\n===")
#%% 2 -- Enforce boundary conditions with 'assemble_system()'
    # Diagonal entries A[i_bc,i_bc] are set to 1.
    # Rows A[i_bc,:] and columns A[:,j_bc] are set to 0.
    # b is corrected as follows:
    #  (1) b[i_bc] is set to the desired boundary value
    #  (2) Other entries are corrected
    #       b = b + b_cor,
    # to account for the cancelling of the columns A[:,j_bc].
    # Remark:
    # If the Dirichlet boundary condition is homogenenous, then b_cor=0.
    # Remark:
    # Unfortunately, there is no way of storing b_cor for later reuse:
    # https://bugs.launchpad.net/dolfin/+bug/867680
fe.assemble_system(a,l,bcs=bc,A_tensor=A,b_tensor=b)
print(f"=== 2) Using assemble_system():\nA=\n{A.array()}\nb = {b.vec().array}\n===")
#%% 3 -- Use 'zero_columns()' with 1.0 as diagonal value
# This gives the same result as 'assemble_system()'
A = A_n.copy()
b = b_n.copy()
diagonal_value = 1.0
bc.zero_columns(A,b,diagonal_value)
print(f"=== 3) Using bc.zero_columns():\nA=\n{A.array()}\nb = {b.vec().array}\n===")
#%% 4 -- Using 'zero_columns()' for an eigenvalue problem
# To solve 
#   A*x = lambda*x
# with homogenous Dirichlet boundary conditions on x, use bc.zero_columns().
# This preserve the symmetry of A, which is desirable for eigensolvers.
A = A_n.copy()
diagonal_value = 1.0
bc.zero_columns(A,fe.Function(V).vector(),diagonal_value)
print(f"=== 4) Using bc.zero_columns() for eigenvalue problems:\nA=\n{A.array()}\n===")
#%% 5 -- Using 'zero_columns()' for a generalized eigenvalue problem
# To solve
#       A*x  = lambda*B*x
# with homogeneous Dirichlet boundary conditions on x, use bc.zero_columns()
# on both A and B with a different diagonal value. The two diagonal values
# must be chosen so that the spurious eigenvalue
#       lambda = diagonal_value_A/diagonal_value_B
# is far away from the region of interest.
# The multiplicity of this spurious eigenvalue is the number of Dirichlet DoFs.
# Spurious eigenvectors are localized at the Dirichlet DoFs.
A = A_n.copy()
B = fe.PETScMatrix()
diagonal_value_A = 1e5
diagonal_value_B = 1e-5
bc.zero_columns(A,fe.Function(V).vector(),diagonal_value_A)
fe.assemble(fe.dot(u, v)*fe.dx,tensor=B)
bc.zero_columns(B,fe.Function(V).vector(),diagonal_value_B)
print(f"=== 5) Using bc.zero_columns() for generalized eigenvalue problems:\nA=\n{A.array()}\nB=\n{B.array()}\n===")
    # Solve dense generalized eigenvalue problem and confirm spurious eigenvalue
import scipy.linalg
A_np = A.array()
B_np = B.array()
(w,vec) = scipy.linalg.eig(A_np,B_np,right=True)
print(f'Expected spurious eigenvalue: {diagonal_value_A/diagonal_value_B}\nComputed eigenvalues:\n {w}')
    # Plot eigenvectors
import matplotlib as mpl
import matplotlib.pyplot as plt
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
for i in range(np.min( (len(w),5) )):
    ax.plot(mesh.coordinates(),vec[:,-i],label=f"{i}")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
ax.set_title("Eigenvectors (including spurious)")