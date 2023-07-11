"""
 
 Utility functions for dolfinx.
 
"""

from mpi4py import MPI
import dolfinx.plot
import ufl
from scientific_computing_utils import gmsh_utils, gmsh_utils_fenicsx
import pyvista as pv
from petsc4py import PETSc

def create_pyvista_UnstructuredGrid_from_mesh(mesh):
    """ Create pyvista.UnstructuredGrid from dolfinx.cpp.mesh.Mesh """
    topology, cell_types, x = dolfinx.plot.create_vtk_mesh(mesh, mesh.topology.dim)
    return pv.UnstructuredGrid(topology, cell_types, x)

def create_pyvista_UnstructuredGrid_from_FunctionSpace(V):
    """ Create pyvista.UnstructuredGrid from dolfinx.fem.function.FunctionSpace """
    topology, cell_types, x = dolfinx.plot.create_vtk_mesh(V)
    return pv.UnstructuredGrid(topology, cell_types, x)

class DolfinxMesh():
    """  This class is meant to ease the creation and manipulation of a dolfinx
    mesh. """

    def __init__(self,mesh,cell_tags,facet_tags,dim,name_2_tags=None):
        """ mesh: dolfinx.mesh.Mesh
            cell_tags, facet_tags: dolfinx.cpp.mesh.MestTags,
            dim: mesh dimension (int), 
            name_2_tags: link between names and tags (list(dict))."""
        self.mesh = mesh
        self.cell_tags = cell_tags
        self.facet_tags = facet_tags
        self.name_2_tags = name_2_tags
        self.dim = dim
            # domain measure
        self.dx = ufl.Measure("dx", subdomain_data=cell_tags)
            # boundary measure
        self.ds = ufl.Measure("ds", subdomain_data=facet_tags)
            # interior boundary measure
        self.dS = ufl.Measure("dS", subdomain_data=facet_tags)

    @classmethod
    def init_from_gmsh(cls,gmshfile,dim,comm=MPI.COMM_WORLD):
        """
        Initialize from a gmsh file.

        Parameters
        ----------
        gmshfile : str
        dim : int
            Mesh dimension.
        comm : mpi4py.MPI.Intracomm, optional
            MPI communicator. The default is MPI.COMM_WORLD.
        """
        mesh, cell_tags, facet_tags = \
            gmsh_utils_fenicsx.gmsh_mesh_to_mesh(gmshfile, dim,comm=comm)
        if comm.rank == 0: # Read physical names and broacast them
            name_2_tags = gmsh_utils.getAllPhysicalNames(gmshfile, dim)
            comm.bcast(name_2_tags, root=0)
        else:
            name_2_tags = comm.bcast(None, root=0)
        return cls(mesh,cell_tags,facet_tags,dim,name_2_tags)

def create_DirichletBC(mesh,facet_tags,V,uDs,uD_facet_tags,idx_sub_space=[]):
    """
    Create a list of Dirichlet boundary conditions.
    
    Inputs
    ------
    mesh : dolfinx.cpp.mesh.Mesh

    facet_tags : dolfinx.cpp.mesh.MeshTags
    
    V: dolfinx.fem.function.FunctionSpace
    
    uDs : list of dolfinx.fem.functionFunction
        List of functions defining u = uD.

    uD_facet_tags = list of int (facet tags)
        List of facet tags identifying where each Dirichlet boundary
        condition is applied.
        
    idx_sub_space: list of int (optional)
        Index of the subspace to which uDs[i] belong. (Useful e.g. to apply 
        Dirichlet boundary condition to a component of a vector only.)
    
    Returns
    -------
    bcs : list of dolfinx.fem.dirichletbc.DirichletBC
    
    """
    if (len(uDs)!=len(uD_facet_tags)):
        raise ValueError("Lists defining Dirichlet boundary conditions have different sizes.")
    bcs = list()
    for i in range(len(uD_facet_tags)):
        bnd_facets = facet_tags.indices[facet_tags.values == uD_facet_tags[i]]
        if len(idx_sub_space)==0: # uDs[i] belongs to V
            bnd_dof = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim-1, bnd_facets)
            bcs.append(dolfinx.fem.dirichletbc(uDs[i], bnd_dof))
        else: # uDs[i] belongs to V.sub(idx_sub_space[i])
            idx = idx_sub_space[i]
            bnd_dof = dolfinx.fem.locate_dofs_topological((V.sub(idx),V.sub(idx).collapse()[0]), mesh.topology.dim-1, bnd_facets)
            bcs.append(dolfinx.fem.dirichletbc(uDs[i], bnd_dof,V.sub(idx)))
    return bcs

def assemble_GEP(a,b,bcs,diag_A=1e2,diag_B=1e-2):
    """
    Assemble the generalized eigenvalue problem
        a(u,v) = lambda*b(u,v) with Dirichlet boundary conditions 'bc'
    as
        A*U = lambda*B*U.

    Parameters
    ----------
    a,b  : ufl.form.Form
        Input bilinear form.
    bcs : list of dolfinx.fem.dirichletbc.DirichletBC
        Dirichlet boundary conditions
    diag_A, diag_B : float, optional
        Diagonal penalization to enforce Dirichlet boundary condition, see remark.

    Returns
    -------
    A,B : petsc4py.PETSc.Mat
    
    Remark
    ------
    Dirichlet boundary conditions are enforced by modifying A and B as follows:
        A[i,j]=A[j,i]=0, B[i,j]=B[j,i]=0, A_[i,i]=diag_A, B_[i,i]=diag_B,
    for i DoF on Dirichlet boundary. This induces a spurious eigenvalue
    lambda=diag_A/diag_B, whose multiplicity is the number of Dirichlet DoF.
    The associated spurious eigenfunctions are localized at each Dirichlet DoF.

    """
    A = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(a), bcs,diag_A); A.assemble()
    B = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(b), bcs,diag_B); B.assemble()
    return (A,B)

def assemble_GEP_block(a,b,bcs,diag_A=1e2,diag_B=1e-2):
    """ Identitcal to assemble_GEP but for block formulations."""
    A = dolfinx.fem.petsc.assemble_matrix_block(dolfinx.fem.form(a), bcs,diag_A); A.assemble()
    B = dolfinx.fem.petsc.assemble_matrix_block(dolfinx.fem.form(b), bcs,diag_B); B.assemble()
    return (A,B)

def project(u, V, bcs=[],petsc_options=dict()):
    """ Return the projection of u on V. """
    # Weak formulation: a(p,v) = l(u,v) for all v in V
    p, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.inner(p, v) * ufl.dx(V.mesh)
    l = ufl.inner(u, v) * ufl.dx(V.mesh)
    problem = dolfinx.fem.petsc.LinearProblem(a, l,bcs=bcs, 
                          petsc_options=petsc_options)
    return problem.solve()
