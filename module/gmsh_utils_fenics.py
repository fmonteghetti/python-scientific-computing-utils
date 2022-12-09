#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility function for using gmsh meshes in fenics.

@author: bino
"""
import os
import errno

def mesh_convert_gmsh2_to_dolfinXML(mshfile):
    """
    Convert MSH file (ASCII, format 2) to DOLFIN-XML.

    Parameters
    ----------
    mshfile : str
        Input mesh file (MSH ASCII format v2, .msh).

    Returns
    -------
    xmlfile : str
        Output mesh file (DOLFIN-XML format, .xml)
        
    Note
    -----
    Input mesh must only contain first-order elements. An alternative is to
    use meshio-convert (--prune-z-0), which is the most modern option.
    """
    __check_read_access(mshfile)
    xmlfile=os.path.splitext(mshfile)[0]+".xml"
    __check_write_access(xmlfile)
    print(f"Converting '{mshfile}' to '{xmlfile}'...")
    os.system(f"dolfin-convert {mshfile} {xmlfile}")
    print(f"Mesh file '{xmlfile}' generated.")
    return xmlfile


def __check_read_access(file):
    file = open(file, "r"); file.close()
    
def __check_write_access(file):
    file = open(file, "w"); file.close()