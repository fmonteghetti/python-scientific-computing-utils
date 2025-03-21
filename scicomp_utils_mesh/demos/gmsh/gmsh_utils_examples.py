#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script demonstrates the use of the gmsh_utils functions.

"""

from scicomp_utils_mesh import gmsh_utils

import os

DIR_MESH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mesh")

# %% 1 -- Mesh without physical groups
geofile = os.path.join(DIR_MESH, "Circle.geo")
mshfile = os.path.join(DIR_MESH, "Circle.msh")
# Generate mesh using CLI
# Note the use of 'parameters' to parametrize the mesh
gmsh_utils.generate_mesh_cli(
    geofile,
    mshfile,
    2,
    refinement=1,
    log=1,
    parameters={"R": 1, "R_TR": 0.5},
    order=2,
)
# gmsh_utils.generate_mesh_api(geofile,mshfile,2,refinement=1,log=0)
gmsh_utils.print_summary(mshfile)
# %% 2 -- Mesh with physical groups and periodicity
geofile = os.path.join(DIR_MESH, "Circle-Rectangle-Periodic.geo")
mshfile = os.path.join(DIR_MESH, "Circle-Rectangle-Periodic.msh")
# gmsh_utils.generate_mesh_cli(geofile,mshfile,2,refinement=0)
gmsh_utils.generate_mesh_api(geofile, mshfile, 2, refinement=3)
gmsh_utils.print_summary(mshfile)
# %% 2 -- Check periodicity
curve1 = {"type": "disk", "name": "Disk-Boundary"}
curve2 = {"type": "vline", "name": "Rectangle-Boundary-Left"}
delta = gmsh_utils.checkPeriodicCurve(mshfile, [curve1, curve2])
curve1 = {"type": "vline", "name": "Rectangle-Boundary-Right"}
curve2 = {"type": "vline", "name": "Rectangle-Boundary-Left"}
delta = gmsh_utils.checkPeriodicCurve(mshfile, [curve1, curve2])
curve1 = {"type": "hline", "name": "Rectangle-Boundary-Top"}
curve2 = {"type": "hline", "name": "Rectangle-Boundary-Bot"}
delta = gmsh_utils.checkPeriodicCurve(mshfile, [curve1, curve2])
# %% 3 -- 3D mesh with physical groups
geofile = os.path.join(DIR_MESH, "torus.geo")
mshfile = os.path.join(DIR_MESH, "torus.msh")
gmsh_utils.generate_mesh_cli(
    geofile,
    mshfile,
    3,
    refinement=0,
    log=1,
    parameters={"R_mi": 0.5, "R_ma": 1, "lc": 1 / 2},
    order=1,
    binary=True,
)
gmsh_utils.print_summary(mshfile)
