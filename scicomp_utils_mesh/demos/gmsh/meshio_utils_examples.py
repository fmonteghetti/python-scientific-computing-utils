#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Examples using functions of meshio_utils.

"""

from scicomp_utils_mesh import meshio_utils

# %% Convert gmsh format to XDMF (2D mesh)
# High-order meshes are supported
# ASCII and binary formats
# msh2 and msh4
gmsh_file = "mesh/mesh/Circle_msh4-order-1.msh"
gmsh_file = "mesh/mesh/Circle_msh4-order-1-binary.msh"
gmsh_file = "mesh/mesh/Circle_msh4-order-2.msh"

meshio_utils.print_diagnostic_gmsh(gmsh_file)
# Export: physical entities are exported in a separate JSON file
xdmfiles = meshio_utils.convert_gmsh_to_XDMF(gmsh_file, prune_z=True)
# Read physical entities from JSON file (gmsh_utils.getPhysicalNames also possible)
gmsh_phys = meshio_utils.read_phys_gmsh(xdmfiles["gmsh_physical_entities"])
PhysName_1D_tag = gmsh_phys[0]
PhysName_2D_tag = gmsh_phys[1]

# %% Convert gmsh format to XDMF (3D mesh)
gmsh_file = "mesh/mesh/torus.msh"
meshio_utils.print_diagnostic_gmsh(gmsh_file)
xdmfiles = meshio_utils.convert_gmsh_to_XDMF(gmsh_file, prune_z=False)
gmsh_phys = meshio_utils.read_phys_gmsh(xdmfiles["gmsh_physical_entities"])
PhysName_1D_tag = gmsh_phys[0]
PhysName_2D_tag = gmsh_phys[1]
PhysName_3D_tag = gmsh_phys[2]
