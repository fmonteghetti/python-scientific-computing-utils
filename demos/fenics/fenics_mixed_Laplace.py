# 
# .. _demo_mixed_poisson:
# 
# Mixed formulation for Poisson equation
# ======================================

from dolfin import *
import matplotlib.pyplot as plt

# Then, we need to create a :py:class:`Mesh <dolfin.cpp.Mesh>` covering
# the unit square. In this example, we will let the mesh consist of 32 x
# 32 squares with each square divided into two triangles::

# Create mesh
mesh = UnitSquareMesh(32, 32)

# Define finite elements spaces and build mixed space
BDM = FiniteElement("BDM", mesh.ufl_cell(), 1)
DG  = FiniteElement("DG", mesh.ufl_cell(), 0)
W = FunctionSpace(mesh, BDM * DG)


# Define trial and test functions
(sigma, u) = TrialFunctions(W)
(tau, v) = TestFunctions(W)

# Define source function
f = Expression("1", degree=2)

# We are now ready to define the variational forms a and L. Since,
# :math:`u_0 = 0` in this example, the boundary term on the right-hand
# side vanishes. ::

# Define variational form
a = (dot(sigma, tau) + div(tau)*u + div(sigma)*v)*dx
L = - f*v*dx


# Compute solution
w = Function(W)
solve(a == L, w, [])
(sigma, u) = w.split()

# Plot sigma and u
plt.figure()
plot(sigma)

plt.figure()
plot(u)

plt.show()
