#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for slepc4py.

Error codes coming from PETSc are described in  petscerror.h. Here is the list:
PETSC_ERR_MIN_VALUE        54   /* should always be one less then the smallest value */
PETSC_ERR_MEM              55   /* unable to allocate requested memory */
PETSC_ERR_SUP              56   /* no support for requested operation */
PETSC_ERR_SUP_SYS          57   /* no support for requested operation on this computer system */
PETSC_ERR_ORDER            58   /* operation done in wrong order */
PETSC_ERR_SIG              59   /* signal received */
PETSC_ERR_FP               72   /* floating point exception */
PETSC_ERR_COR              74   /* corrupted PETSc object */
PETSC_ERR_LIB              76   /* error in library called by PETSc */
PETSC_ERR_PLIB             77   /* PETSc library generated inconsistent data */
PETSC_ERR_MEMC             78   /* memory corruption */
PETSC_ERR_CONV_FAILED      82   /* iterative method (KSP or SNES) failed */
PETSC_ERR_USER             83   /* user has not provided needed function */
PETSC_ERR_SYS              88   /* error in system call */
PETSC_ERR_POINTER          70   /* pointer does not point to valid address */
PETSC_ERR_MPI_LIB_INCOMP   87   /* MPI library at runtime is not compatible with MPI user compiled with */
PETSC_ERR_ARG_SIZ          60   /* nonconforming object sizes used in operation */
PETSC_ERR_ARG_IDN          61   /* two arguments not allowed to be the same */
PETSC_ERR_ARG_WRONG        62   /* wrong argument (but object probably ok) */
PETSC_ERR_ARG_CORRUPT      64   /* null or corrupted PETSc object as argument */
PETSC_ERR_ARG_OUTOFRANGE   63   /* input argument, out of range */
PETSC_ERR_ARG_BADPTR       68   /* invalid pointer argument */
PETSC_ERR_ARG_NOTSAMETYPE  69   /* two args must be same object type */
PETSC_ERR_ARG_NOTSAMECOMM  80   /* two args must be same communicators */
PETSC_ERR_ARG_WRONGSTATE   73   /* object in argument is in wrong state, e.g. unassembled mat */
PETSC_ERR_ARG_TYPENOTSET   89   /* the type of the object has not yet been set */
PETSC_ERR_ARG_INCOMP       75   /* two arguments are incompatible */
PETSC_ERR_ARG_NULL         85   /* argument is null that should not be */
PETSC_ERR_ARG_UNKNOWN_TYPE 86   /* type name doesn't match any registered type */
PETSC_ERR_FILE_OPEN        65   /* unable to open file */
PETSC_ERR_FILE_READ        66   /* unable to read from file */
PETSC_ERR_FILE_WRITE       67   /* unable to write to file */
PETSC_ERR_FILE_UNEXPECTED  79   /* unexpected data in file */
PETSC_ERR_MAT_LU_ZRPVT     71   /* detected a zero pivot during LU factorization */
PETSC_ERR_MAT_CH_ZRPVT     81   /* detected a zero pivot during Cholesky factorization */
PETSC_ERR_INT_OVERFLOW     84
PETSC_ERR_FLOP_COUNT       90
PETSC_ERR_NOT_CONVERGED    91  /* solver did not converge */
PETSC_ERR_MISSING_FACTOR   92  /* MatGetFactor() failed */
PETSC_ERR_OPT_OVERWRITE    93  /* attempted to over write options which should not be changed */
PETSC_ERR_WRONG_MPI_SIZE   94  /* example/application run with number of MPI ranks it does not support */
PETSC_ERR_USER_INPUT       95  /* missing or incorrect user input */
PETSC_ERR_GPU_RESOURCE     96  /* missing or incorrect user input */
PETSC_ERR_MAX_VALUE        97  /* this is always the one more than the largest error code */
"""
from mpi4py import MPI
from slepc4py import SLEPc
import numpy as np

def monitor_EPS_short(EPS, it, nconv, eig, err,it_skip):
    """
    Concise monitor for EPS.solve().

    Parameters
    ----------
    eps : slepc4py.SLEPc.EPS
        Eigenvalue Problem Solver class.
    it : int
        Current iteration number.
    nconv : int
        Number of converged eigenvalue.
    eig : list
        Eigenvalues
    err : list
        Computed errors.
    it_skip : int
        Iteration skip.

    Returns
    -------
    None.

    """
    if (it==1):
        print('******************************')
        print('***  SLEPc Iterations...   ***')
        print('******************************')
        EPS_print_short(EPS)
        print("Iter. | Conv. | Max. error")
        print(f"{it:5d} | {nconv:5d} | {max(err):1.1e}")
    elif not it%it_skip:
        print(f"{it} | {nconv} | {max(err):1.1e}")


def EPS_print_short(EPS):
    print(f"Problem dimension: {EPS.getOperators()[0].size[0]}")
    print(f"Solution method: '{EPS.getType()}' with '{EPS.getST().getType()}'")
    nev,ncv = EPS.getDimensions()[0:2]
    print( f"Number of requested eigenvalues: {nev}")
    print(f"Number of MPI process: {EPS.comm.Get_size()}")
    if ncv>0:
        print(f'Subspace dimension: {ncv}')
    tol, maxit = EPS.getTolerances()
    print( f"Stopping condition: tol={tol}, maxit={maxit}")
    

def EPS_print_results(EPS):
    print()
    print("******************************")
    print("*** SLEPc Solution Results ***")
    print("******************************")           
    its = EPS.getIterationNumber()
    print(f"Iteration number: {its}")
    nconv = EPS.getConverged()
    print( f"Converged eigenpairs: {nconv}")

    if nconv > 0:
      # Create the results vectors
      vr, vi = EPS.getOperators()[0].createVecs()
      print()
      print("Converged eigval.  Error ")
      print("----------------- -------")
      for i in range(nconv):
        k = EPS.getEigenpair(i, vr, vi)
        error = EPS.computeError(i)
        if k.imag != 0.0:
          print( f" {k.real:2.2e} + {k.imag:2.2e}j {error:1.1e}")
        else:
          print(f" {k.real:2.2e}         {error:1.1e}")
      print()



def solve_GEP_shiftinvert(A,B,
                          problem_type=SLEPc.EPS.ProblemType.GNHEP,
                          solver=SLEPc.EPS.Type.KRYLOVSCHUR,
                          nev=10,tol=1e-6,max_it=10,
                          target=0.0,shift=0.0,defl_space=[],
                          comm=MPI.COMM_WORLD):
    """
    Solve generalized eigenvalue problem A=lambda*B using shift-and-invert
    as spectral transform method.

    Parameters
    ----------
    A,B : PETSc.Mat
    problem_type : SLEPc.EPS.ProblemType, optional
    solver : SLEPc.EPS.Type., optional
    nev : int, optional
        Number of requested eigenvalues.
    tol : float, optional
        Tolerance.
    max_it : int, optional
        Maximum number of iterations.
    target : float, optional
        Target eigenvalue. Also used for sorting.
    shift : float, optional
        Shift 'sigma' used in shift-and-invert.
    defl_space: list of petsc4py.PETSc.Vec, optional
        Deflation space.
    comm: mpi4py.MPI.Intracomm
        MPI communicator. Default is to use all processes.
    Returns
    -------
    eigval : list of complex
        Converged eigenvalues.
    eigvec_r : list of PETSc.Vec
        Converged eigenvector (real_part)
    eigvec_i : TYPE
        Converged eigenvectors (imag_part)
        
    """
    
        # Build an Eigenvalue Problem Solver object
    EPS = SLEPc.EPS(); EPS.create(comm=comm)
    EPS.setOperators(A,B)
        # (G)HEP = (Generalized) Hermitian
        # (G)NHEP = (Generalized) Non-Hermitian
    EPS.setProblemType(problem_type)
        # set the number of eigenvalues requested
    EPS.setDimensions(nev=nev)
        # Set solver
    EPS.setType(solver)
        # set eigenvalues of interest
    EPS.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
    EPS.setTarget(target) # sorting
        # set tolerance and max iterations
    EPS.setTolerances(tol=tol,max_it=max_it)    
        # deflation space
    EPS.setDeflationSpace(defl_space)
        # Set up shift-and-invert
        # Only work if 'whichEigenpairs' is 'TARGET_XX'
    ST=EPS.getST()
    ST.setType(SLEPc.ST.Type.SINVERT)
    ST.setShift(shift)        
    EPS.setST(ST)
        # set monitor
    it_skip=1
    if comm.rank==0:
        EPS.setMonitor(lambda eps, it, nconv, eig, err :
                       monitor_EPS_short(eps, it, nconv, eig, err,it_skip))
        # parse command line options
    EPS.setFromOptions()
        # Display all options (including those of ST object)
    # if comm.rank==0:
    #     EPS.view()          
    EPS.solve()
    if comm.rank==0:
        print('******************************')
        # Print results
    # if comm.rank==0:
    #     SLEPc_utils.EPS_print_results(EPS)   
    return EPS


def EPS_get_spectrum(EPS):
    A = EPS.getOperators()[0]
        # Get results in lists
    eigval = [EPS.getEigenvalue(i) for i in range(EPS.getConverged())]
    eigvec_r = list(); eigvec_i = list()
    vr = A.createVecRight(); vi = A.createVecRight()
    for i in range(EPS.getConverged()):
        EPS.getEigenvector(i,vr,vi)
        eigvec_r.append(vr.copy())
        eigvec_i.append(vi.copy())
    # Sort by increasing real parts
    idx=np.argsort(np.real(np.array(eigval)),axis=0)
    eigval = [eigval[i] for i in idx]
    eigvec_r = [eigvec_r[i] for i in idx]
    eigvec_i = [eigvec_i[i] for i in idx]
    return (eigval,eigvec_r,eigvec_i)

def EPS_get_spectrum_ReImFormulation(EPS,idx_r,idx_i,tol=None):
    """
    Returns the correct spectrum for a generalized eigenvalue problem 
    formulated by splitting real and imaginary parts. Spurious eigenvalues 
    are eliminated by checking the residual.

    Parameters
    ----------
    EPS : slepc4py.SLEPc.EPS
        Solved eigenvalue problem.
    idx_r, idx_i : array
        Indices of DoF associated with the real (resp. imaginary) component.
    tol : float, optional
        Tolerance to filter spurious eigenvalues.

    Returns
    -------
    eigval : np.array of complex
        Eigenvalues.
    eigvec : list of petsc4py.PETSc.Vec 
        Eigenvectors (real-valued)
    """
    (A,B) = EPS.getOperators()
    eigval = list(); eigvec = list()
    vr = A.createVecRight()
    res = A.createVecRight()
    B_vr = A.createVecRight()
    B_vr_swap = A.createVecRight()
    if tol is None:
        tol= EPS.getTolerances()[0]    
    for i in range(EPS.getConverged()):
        val = EPS.getEigenpair(i,vr) # vr = [Re(x);Re(y)]
        # compute residue ||A*x-val*B*x||_2
        B.mult(vr,B_vr) # B*vr
        B_vr_swap[idx_r] = -B_vr[idx_i]
        B_vr_swap[idx_i] =  B_vr[idx_r]
        # res = A*vr - real(val)*B*[Re(x);Re(y)] - imag(val)*B*[-Re(y);Re(x)]
        A.multAdd(vr,-(val.real)*B_vr-(val.imag)*B_vr_swap,res)
        if res.norm()<tol:
            eigval.append(val)
            eigvec.append(vr.copy())
        # Sort by increasing real parts
    idx=np.argsort(np.real(np.array(eigval)),axis=0)
    eigval = [eigval[i] for i in idx]
    eigvec = [eigvec[i] for i in idx]
    return (np.array(eigval),eigvec)
