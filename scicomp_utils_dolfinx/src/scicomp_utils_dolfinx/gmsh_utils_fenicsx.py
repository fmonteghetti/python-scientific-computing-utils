"""
 
 Utility functions to handle gmsh meshes in dolfinx.
 
"""

from mpi4py import MPI
import dolfinx

def gmsh_mesh_to_mesh(gmshfile, gdim,comm=MPI.COMM_WORLD,
                                   cell_data=True, facet_data=True):
    """
    Read a gmsh .msh file and build a dolfinx dolfinx.mesh.Mesh, distributed 
    over the ranks of the given MPI communicator.
    
    Parameters
    ----------
    gmshfile : str
    gdim : int
        Mesh dimension.
    comm : mpi4py.MPI.intracomm, optional
        MPI communicator. The default is MPI.COMM_WORLD.
    cell_data : boolean, optional
         Read meshtags for cell (gdim-1). The default is True.
    facet_data : boolean, optional
         Read meshtags for facets (gdim-2). The default is True.

    Returns
    -------
    dolfinx.mesh.Mesh
    dolfinx.cpp.mesh.MeshTags
    dolfinx.cpp.mesh.Meshtags

    """
    (mesh,ct,ft) = dolfinx.io.gmshio.read_from_msh(gmshfile,comm=comm,gdim=gdim)

    if cell_data and facet_data:
        return mesh, ct, ft
    elif cell_data and not facet_data:
        return mesh, ct
    elif not cell_data and facet_data:
        return mesh, ft
    else:
        return mesh