"""
 
 Utility functions to handle gmsh meshes in dolfinx.
 
"""


import numpy
import gmsh

from mpi4py import MPI
try:
    from dolfinx.io import extract_gmsh_geometry, extract_gmsh_topology_and_markers, ufl_mesh_from_gmsh
except ImportError: # API change in 0.5.0
    from dolfinx.io.gmshio import extract_geometry, extract_topology_and_markers, ufl_mesh
    extract_gmsh_geometry = extract_geometry
    extract_gmsh_topology_and_markers = extract_topology_and_markers
    ufl_mesh_from_gmsh = ufl_mesh
from dolfinx.cpp.io import perm_gmsh, distribute_entity_data
from dolfinx.cpp.mesh import to_type, cell_entity_type
from dolfinx.graph import create_adjacencylist
from dolfinx.mesh import meshtags_from_entities, create_mesh

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
    # =========================================
    # Copyright (C) 2020 Jørgen S. Dokken
    #
    # SPDX-License-Identifier:    LGPL-3.0-or-later
    #
    # =========================================
    # GMSH model to dolfinx.Mesh converter
    # =========================================
    # Modified to work with dolfinx:v0.4.1, using demo_gmsh.py
    # TODO: Should eventually be replaced by dolfinx.io.gmshio.model_to_mesh()

    if comm.rank == 0: # only rank 0 knows about the gmsh model        
        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 3)
        gmsh.model.add("Mesh from file")
        gmsh.merge(gmshfile)
        model = gmsh.model
        
        # Get mesh geometry
        x = extract_gmsh_geometry(model)

        # Get mesh topology for each element
        topologies = extract_gmsh_topology_and_markers(model)

        # Get information about each cell type from the msh files
        num_cell_types = len(topologies.keys())
        cell_information = {}
        cell_dimensions = numpy.zeros(num_cell_types, dtype=numpy.int32)
        for i, element in enumerate(topologies.keys()):
            properties = model.mesh.getElementProperties(element)
            name, dim, order, num_nodes, local_coords, _ = properties
            cell_information[i] = {"id": element, "dim": dim,
                                   "num_nodes": num_nodes}
            cell_dimensions[i] = dim
            # print(f"Cell: {cell_information[i]['id']} order {order}")

        # We are done with gmsh
        gmsh.finalize()

        # Sort elements by ascending dimension
        perm_sort = numpy.argsort(cell_dimensions)

        # Broadcast cell type data and geometric dimension
        cell_id = cell_information[perm_sort[-1]]["id"]
        tdim = cell_information[perm_sort[-1]]["dim"]
        num_nodes = cell_information[perm_sort[-1]]["num_nodes"]
        cell_id, num_nodes = comm.bcast([cell_id, num_nodes], root=0)

        # Check for facet data and broadcast if found
        if facet_data:
            if tdim - 1 in cell_dimensions:
                num_facet_nodes = comm.bcast(
                    cell_information[perm_sort[-2]]["num_nodes"], root=0)
                gmsh_facet_id = cell_information[perm_sort[-2]]["id"]
                marked_facets = numpy.asarray(topologies[gmsh_facet_id]["topology"], dtype=numpy.int64)
                facet_values = numpy.asarray(topologies[gmsh_facet_id]["cell_data"], dtype=numpy.int32)
            else:
                raise ValueError("No facet data found in file.")

        cells = numpy.asarray(topologies[cell_id]["topology"], dtype=numpy.int64)
        cell_values = numpy.asarray(topologies[cell_id]["cell_data"], dtype=numpy.int32)

    else:
        cell_id, num_nodes = comm.bcast([None, None], root=0)
        cells, x = numpy.empty([0, num_nodes], dtype=numpy.int32), numpy.empty([0, gdim])
        cell_values = numpy.empty((0,), dtype=numpy.int32)
        if facet_data:
            num_facet_nodes = comm.bcast(None, root=0)
            marked_facets = numpy.empty((0, num_facet_nodes), dtype=numpy.int32)
            facet_values = numpy.empty((0,), dtype=numpy.int32)

    # Create distributed mesh
    ufl_domain = ufl_mesh_from_gmsh(cell_id, gdim)
    gmsh_cell_perm = perm_gmsh(to_type(str(ufl_domain.ufl_cell())), num_nodes)
    cells = cells[:, gmsh_cell_perm]
    mesh = create_mesh(comm, cells, x[:, :gdim], ufl_domain)
    # Create MeshTags for cells
    if cell_data:
        local_entities, local_values = distribute_entity_data(
            mesh, mesh.topology.dim, cells, cell_values)
        mesh.topology.create_connectivity(mesh.topology.dim, 0)
        adj = create_adjacencylist(local_entities)
        ct = meshtags_from_entities(mesh, mesh.topology.dim,
                                    adj, numpy.int32(local_values))
        ct.name = "Cell tags"

    # Create MeshTags for facets
    if facet_data:
        # Permute facets from MSH to Dolfin-X ordering
        facet_type = cell_entity_type(to_type(str(ufl_domain.ufl_cell())),
                                      mesh.topology.dim - 1,0)
        gmsh_facet_perm = perm_gmsh(facet_type, num_facet_nodes)
        marked_facets = marked_facets[:, gmsh_facet_perm]

        local_entities, local_values = distribute_entity_data(
            mesh, mesh.topology.dim - 1, marked_facets, facet_values)
        mesh.topology.create_connectivity(
            mesh.topology.dim - 1, mesh.topology.dim)
        adj = create_adjacencylist(local_entities)
        ft = meshtags_from_entities(mesh, mesh.topology.dim - 1,
                             adj, numpy.int32(local_values))
        ft.name = "Facet tags"

    if cell_data and facet_data:
        return mesh, ct, ft
    elif cell_data and not facet_data:
        return mesh, ct
    elif not cell_data and facet_data:
        return mesh, ft
    else:
        return mesh