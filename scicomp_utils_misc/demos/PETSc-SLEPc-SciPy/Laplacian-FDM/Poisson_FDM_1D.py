# -*- coding: utf-8 -*-
"""
This tutorial module tackles two problems. First, it solves the 1D Poisson equation

    .. math:: -\\Delta u = f\quad(0,L)

with boundary conditions

    .. math:: u = g\quad\{0,L\}

Secondly, it computes the eigenvalues of the Dirichlet Laplacian on (0,L). Three implementations are available:
    - Numpy (dense)
    - Scipy (sparse)
    - PETSc (sparse)

"""

import numpy as np
from numpy import pi

import scipy
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg

from petsc4py import PETSc
from slepc4py import SLEPc


def assemble_numpy(L, N, f, g):
    """
    Assemble the direct problem Ay=b for the interior nodes

    Parameters
    ----------
    L : real (1)
        Domain length
    N : int (1)
        Number of nodes
    f : function
        Source term.
    g : function
        Boundary term.

    Returns
    -------
    A : numpy.ndarray (N-2,N-2)
        FDM matrix for -Delta.
    b : numpy.ndarray (N-2)
        FDM right-hand side.
    """
    (h, x_FDM) = get_FDM_nodes(L, N)
    # dense tridiagonal matrix
    d = -2 * np.ones(N - 2)
    e = 1 * np.ones(N - 3)
    A = np.diag(d) + np.diag(e, k=1) + np.diag(e, k=-1)
    A = -((1 / h) ** 2) * A
    # source term
    F = f(x_FDM[1:-1])
    # boundary term
    G = np.zeros((N - 2,))
    G[0] = g(x_FDM[0])
    G[-1] = g(x_FDM[-1])
    G = ((1 / h) ** 2) * G
    return (A, F + G)


def assemble_scipy(L, N, f, g):
    """
    Identical to `assemble_numpy`, except for the return types.

    Returns
    -------
    A : scipy.sparse.csr.csr_matrix (N-2,N-2)
        FDM matrix for -Delta.
    b : numpy.ndarray (N-2)
        FDM right-hand side.
    """

    (h, x_FDM) = get_FDM_nodes(L, N)
    # sparse tridiagonal matrix
    d = -2 * np.ones(N - 2)
    e = 1 * np.ones(N - 3)
    A_lower = sp.spdiags(e, -1, N - 2, N - 2, format="csr")
    A = (
        sp.spdiags(d, 0, N - 2, N - 2, format="csr")
        + A_lower
        + A_lower.transpose()
    )
    A = -((1 / h) ** 2) * A
    # source term
    F = f(x_FDM[1:-1])
    # boundary term
    G = np.zeros((N - 2,))
    G[0] = g(x_FDM[0])
    G[-1] = g(x_FDM[-1])
    G = ((1 / h) ** 2) * G
    return (A, F + G)


def assemble_PETSc(L, N, f, g):
    """
    Identical to `assemble_numpy`, except for the return types.

    Returns
    -------
    A : PETSc.Mat (N-2,N-2)
        FDM matrix for -Delta.
    b : PETSc.Vec (N-2)
        FDM right-hand side.
    """

    (h, x_FDM) = get_FDM_nodes(L, N)

    # sparse tridiagonal matrix
    A = PETSc.Mat()
    A.create(PETSc.COMM_WORLD)
    A.setSizes([N - 2, N - 2])
    A.setType("aij")  # sparse (csr)
    A.setPreallocationNNZ(N - 2)  # estimate of number of non-null element
    # source vector
    F = PETSc.Vec().create(PETSc.COMM_WORLD)
    F.setSizes(N - 2)
    F.setFromOptions()
    F.set(0)
    # boundary vector
    G = F.duplicate()
    G.set(0)
    # assemble
    fac = (1 / h) ** 2
    # loop over owned block of rows on this
    # processor and insert entry values
    Istart, Iend = A.getOwnershipRange()
    for I in range(Istart, Iend):
        A[I, I] = fac * (2)
        F[I] = f(x_FDM[I + 1])
        if I > 0:
            A[I, I - 1] = fac * (-1)
        else:
            G[I] = fac * g(x_FDM[0])
        if I < (N - 3):
            A[I, I + 1] = fac * (-1)
        else:
            G[I] = fac * g(x_FDM[-1])

        # communicate off-processor values
        # and setup internal data structures
        # for performing parallel operations
    A.assemblyBegin()
    A.assemblyEnd()
    F.assemblyBegin()
    F.assemblyEnd()
    G.assemblyBegin()
    G.assemblyEnd()

    return (A, F + G)


def solve_scipy(A, b, solver="umfpack"):
    """
    Solution of the sparse linear system Ax=b.

    Parameters
    ----------
    A : scipy.sparse matrix

    b : numpy.ndarray vector

    solver : str, optional
        Solver: "umfpack", "gmres", "cg".

    Returns
    -------
    x : numpy.ndarray vector
        Computed solution

    """
    if solver == "umfpack":
        x = splinalg.spsolve(A, b)  # UMFPack
    elif solver == "gmres":
        x = splinalg.gmres(A, b)
        x = x[0]  # GMRES
    else:
        x = splinalg.cg(A, b)
        x = x[0]  # CG

    return x


def solve_PETSc(A, b, solver="gmres", comm=None):
    """
    Solution of the sparse linear system Ax=b.

    Parameters
    ----------
    A : PETSc.Mat

    b : numpy.ndarray vector

    solver : str, optional
        Solver: "gmres", "cg".

    comm : MPI communicator, optional.

    Returns
    -------
    x : numpy.ndarray vector
        Computed solution

    """

    ksp = PETSc.KSP()  # create linear solver
    ksp.create(comm)
    if solver == "gmres":
        ksp.setType("gmres")
    else:
        ksp.setType("cg")
    y, c = A.createVecs()  # obtain sol & rhs vectors
    y.set(0)
    ksp.setOperators(A)  # solve
    ksp.setFromOptions()
    ksp.solve(b, y)
    return y.getArray()  # get numpy array


def eigensolve_SLEPc(
    A, ProblemType=SLEPc.EPS.ProblemType.NHEP, comm=None, nev=1
):
    """
    Compute spectrum of sparse matrix A.

    Parameters
    ----------
    A : PETSc.Mat

    ProblemType : int, optional

    comm : MPI communicator PETSc.Comm, optional

    nev : int, optional
        Number of eigenvalues requested

    Returns
    -------
    E : SLEPc.EPS
        Eigenvalue Problem Solver class from SLEPc. Contains the computational results.

    """

    # Build an Eigenvalue Problem Solver object
    E = SLEPc.EPS()
    E.create(comm=PETSc.COMM_WORLD)
    E.setOperators(A)
    # HEP = Hermitian
    # NHEP = Npn-Hermitian
    E.setProblemType(ProblemType)
    # set the number of eigenvalues requested
    E.setDimensions(nev=nev)
    # desired eigenvalues
    E.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_MAGNITUDE)
    # monitor
    # history = []
    # def monitor(eps, its, nconv, eig, err):
    #    history.append(err[nconv])
    #    #print(its, rnorm)
    # E.setMonitor(monitor)
    E.setFromOptions()
    E.solve()
    return E


def eigensolve_SLEPc_post_process(A, E):
    """
    Print the content of the EPS object E.

    Parameters
    ----------
    A : PETSc.Mat

    E : SLEPc.EPS
        Eigenvalue Problem Solver class from SLEPc.

    Returns
    -------
    None.

    """

    Print = PETSc.Sys.Print

    Print()
    Print("******************************")
    Print("*** SLEPc Solution Results ***")
    Print("******************************")
    Print()

    its = E.getIterationNumber()
    Print("Number of iterations of the method: %d" % its)

    eps_type = E.getType()
    Print("Solution method: %s" % eps_type)

    nev, ncv, mpd = E.getDimensions()
    Print("Number of requested eigenvalues: %d" % nev)

    tol, maxit = E.getTolerances()
    Print("Stopping condition: tol=%.4g, maxit=%d" % (tol, maxit))

    nconv = E.getConverged()
    Print("Number of converged eigenpairs %d" % nconv)

    if nconv > 0:
        # Create the results vectors
        vr, vi = A.createVecs()
        #
        Print()
        Print("        k          ||Ax-kx||/||kx|| ")
        Print("----------------- ------------------")
        for i in range(nconv):
            k = E.getEigenpair(i, vr, vi)
            error = E.computeError(i)
            if k.imag != 0.0:
                Print(" %9f%+9f j %12g" % (k.real, k.imag, error))
            else:
                Print(" %12f       %12g" % (k.real, error))
        Print()


def get_spectrum(A, E):
    """
    Return the eigenvalues and eigenvectors contained in E.

    Parameters
    ----------
    A : PETSc.Mat (N,N)
        Used read-only to create vectors
    E : SLEPc.EPS
        Eigenvalue Problem Solver class from SLEPc.

    Returns
    -------
    eigval : numpy.ndarray (M)
        Eigenvalues (complex)
    eigvec : numpy.ndarray (N,M)
        Eigenvectors

    """
    # Build a list of eigenvectors and eigenvalues
    eigval = np.zeros(E.getConverged(), dtype=complex)
    eigvec_real = np.zeros((A.size[0], E.getConverged()))
    eigvec_imag = np.zeros((A.size[0], E.getConverged()))

    for i in range(E.getConverged()):
        eigval[i] = E.getEigenvalue(i)
        vr, vi = A.createVecs()
        E.getEigenvector(i, vr, vi)
        eigvec_real[:, i] = vr.array[:]
        eigvec_imag[:, i] = vi.array[:]
    return (eigval, eigvec_real + 1j * eigvec_imag)


def get_FDM_nodes(L, N):
    h = L / (N - 1)  # step size
    x_FDM = h * np.r_[0:N]  # nodes
    return (h, x_FDM)


def add_boundary_nodes(y, g, L, N):
    (h, x_FDM) = get_FDM_nodes(L, N)
    y = np.hstack((g(x_FDM[0]), y, g(x_FDM[-1])))
    return y
