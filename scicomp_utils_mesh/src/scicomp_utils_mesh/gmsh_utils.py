#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gather convenient functions for gmsh.

"""

import os
import random
import string
import gmsh
import numpy as np


##############################################################################
# MESH GENERATION
##############################################################################

def generate_mesh_cli(geofile,gmshfile,dim,order=1,
                           gmshformat="auto",meshing=-1,recombination=-1,
                           refinement=0,log=0,verbosity=3,flexible_transfinite=False,
                           binary=False,parameters=dict(),save_and_exit=False,
                           use_system_options=False):
    """
    Generate mesh (MSH format, ASCII) from gmsh geometry file.

    Parameters
    ----------
    geofile : str
        gmsh geometry file (.geo)
    gmshfile : str
        gmsh mesh file (.msh)
    dim : int
        Number of spatial dimensions.
    order : int, optional
        Geometrical order of mesh. The default is 1. (Mesh.ElementOrder)
    meshing: int, optional
        Meshing algorithm. (Mesh.Algorithm)
    recombination: int, optional
        Recombination algorithm. (Mesh.RecombinationAlgorithm)
    refinement : int, optional
        Number of refinment steps. The default is 1.
    gmshformat : str, optional
        Msh format. The default is "auto".
    log : int, optional
        Write log to a .log file. The default is 0.
    verbosity : int, optional
        Verbosity level (General.Verbosity). The default is 3.
    flexible_transfinite: boolean, optional
        Enable option Mesh.FlexibleTransfinite.
    binary : boolean, optional
        Produce binary mesh (not supported with all format).
    parameters: dict, optional
        Parameters values for the .geo file. Format: {'param':val}.
    save_and_exit: bool, optional
        Save mesh and exit. Useful only when geofile contains meshing
        directives.
    use_system_options: bool, optional
        Use the default 'OptionsFileName' and 'SessionFileName'. This is disabled
        by default for reproducibility.

    Returns
    -------
    None.
    
    Remark
    ------
    The ``parameters'' argument is useful to parametrize the geometry described
    by the .geo file. In the .geo file, define parameters using
        If (!Exists(param)) param=val_default; EndIf,
    so that they can be overriden from Python.

    """
    
    __check_read_access(geofile)
    __check_write_access(gmshfile)
        
    print(f"Generating mesh '{gmshfile}' from '{geofile}'...")
    arg=list() # command line argument
    arg.append(f"-order {order}")
    arg.append(f"-format {gmshformat}")
    arg.append(f"-v {verbosity}")
    command=list() # command string
    if binary:
        arg.append("-bin")        
    if save_and_exit:
        arg.append("-save")
    else:
        arg.append(f"-{dim}")
    if meshing!=-1:
        command.append(f"Mesh.Algorithm={meshing};")
    if recombination!=-1:
        command.append(f"Mesh.RecombinationAlgorithm={recombination};")
    if flexible_transfinite:
        command.append("Mesh.FlexibleTransfinite=1;")
    if log:
        fname=os.path.splitext(gmshfile)[0]
        arg.append(f"-log {fname}.log")
    for param_name in parameters:
        arg.append(f"-setnumber {param_name} {parameters[param_name]}")
    arg=' '.join(arg)
    command=' '.join(command)
    if use_system_options==False:
        # By setting GMSH_HOME to a non-existent folder, the system
        # option and session files are not used (General.OptionsFileName, 
        # General.SessionFileName).
        def get_random_string(N):
            return ''.join(random.choice(string.ascii_uppercase + string.digits)
                            for _ in range(N))
        old_env = dict(os.environ)
        os.environ['GMSH_HOME'] = get_random_string(5) 
    os.system(f"gmsh {geofile} -o {gmshfile} {arg} -string \"{command}\"")   
    for i in range(refinement):
        os.system(f"gmsh {gmshfile} -refine {arg}")
    print(f"Mesh '{gmshfile}' generated from '{geofile}'.")
    if use_system_options==False:
        os.environ.clear()
        os.environ.update(old_env) 

def generate_mesh_api(geofile,gmshfile,dim,order=1,refinement=0,
                       gmshformat=2,log=0,verbosity=3,binary=False):
    """
    Identical to ``generate_mesh_cli``, but relies on the API. Does not handle
    parameters.
    
    """
    
    __check_read_access(geofile)
    __check_write_access(gmshfile)
   
    gmsh.initialize()
    if log:
        gmsh.logger.start()
    print(f"Opening geometry file '{geofile}'...")
    gmsh.open(geofile)
    gmsh.model.mesh.setOrder(order) # mesh geometrical order
    gmsh.option.setNumber("Mesh.MshFileVersion", gmshformat)
    gmsh.option.setNumber("General.Verbosity", verbosity)
    gmsh.option.setNumber("Mesh.Binary", binary)
    gmsh.model.mesh.generate(dim)
    for i in range(refinement):
        gmsh.model.mesh.refine()
    gmsh.write(gmshfile)
    print(f"Mesh '{gmshfile}' generated from '{geofile}'.")
    if log:
        fname=os.path.splitext(gmshfile)[0]
        logfile = open(f"{fname}.log", "w")
        logfile.write('\n'.join(gmsh.logger.get()))
        logfile.close()
        gmsh.logger.stop()
    gmsh.finalize()


##############################################################################
# MESH MANIPULATION THROUGH GMSH PYTHON API
##############################################################################


def print_summary(mshfile):
    """
    Print statistics about the mesh file ``mshfile``.

    Parameters
    ----------
    mshfile : str
        gmsh mesh file (.msh)

    Returns
    -------
    None.

    """
    
    __check_read_access(mshfile)

    gmsh.initialize()
    print(f"Opening '{mshfile}'...")
    gmsh.open(mshfile)
    n_nodes=len(gmsh.model.mesh.getNodes(returnParametricCoord=False)[0])
    print(f"Total # of nodes: {n_nodes}")
    n_elemType=len(gmsh.model.mesh.getElements()[0])
    print(f"Total # of element types: {n_elemType}")
    n_physGroup = len(gmsh.model.getPhysicalGroups())
    print(f"Total # of physical groups: {n_physGroup}")
    for dim in [0,1,2,3]:
        print(f"Element type(s) of dim {dim}:")
        (elemTypes,elemTags) = gmsh.model.mesh.getElements(dim)[0:2]
        for i in range(len(elemTypes)):
            name=gmsh.model.mesh.getElementProperties(elemTypes[i])[0]
            n_elemType=len(elemTags[i])
            print(f"\t<{name}> (#{elemTypes[i]})  {n_elemType}")        
        if len(elemTypes)==0:
            print("\tNone")
    for dim in [0,1,2,3]:
        print(f"Physical group(s) of dim {dim}:")
        phys_group = gmsh.model.getPhysicalGroups(dim)
        for (dim,tag) in phys_group:
            name = gmsh.model.getPhysicalName(dim,tag)
            (nodeTags,coord)=gmsh.model.mesh.getNodesForPhysicalGroup(dim,tag)
            n_node=len(nodeTags)
            print(f"\t'{name}': Tag {tag}, {n_node} nodes")        
        if len(phys_group)==0:
            print("\tNone")
    print(f"Closing '{mshfile}'...")
    gmsh.finalize()


def getAllPhysicalNames(mshfile,dim_mesh,verbosity=3):
    """
    Get all the physical names stored in ``mshfile``.
    
    Parameters
    ----------
    mshfile : str
        gmsh mesh file (.msh)

    dim_mesh : int
        Spatial dimension of the mesh.
        
    verbosity : int, optional
        Verbosity level (General.Verbosity). The default is 3.

    Returns
    -------
    physNames: list(dict)
        physNames[d] is a dictionary wit format dict['name']=[tag1,tag2,...].
    """

    __check_read_access(mshfile)    
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", verbosity)
    gmsh.open(mshfile)
    phys_names = list()
    for dim in range(dim_mesh+1):
        phys_group = gmsh.model.getPhysicalGroups(dim)
        dico = dict()
        for (dim,tag) in phys_group:
            name = gmsh.model.getPhysicalName(dim,tag)
            if name in dico:
                dico[name].append(tag)
            else:
                dico[name]=[tag]
        phys_names.append(dico)
    gmsh.finalize()
    return phys_names

def getPhysicalNames(mshfile,dim,verbosity=3):
    """
    Get all physical names associated with entities of dimension ``dim`` stored
    in ``mshfile``.
    
    Parameters
    ----------
    mshfile : str
        gmsh mesh file (.msh)

    dim : int
        Spatial dimension (0,1,2,3).
        
    verbosity : int, optional
        Verbosity level (General.Verbosity). The default is 3.

    Returns
    -------
    physNames: dict
        Dictionary with format dict['name']=[tag1,tag2,...].

    """
    
    __check_read_access(mshfile)
    
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", verbosity)
    gmsh.open(mshfile)
    phys_group = gmsh.model.getPhysicalGroups(dim)
    dico = dict()
    for (dim,tag) in phys_group:
        name = gmsh.model.getPhysicalName(dim,tag)
        if name in dico:
            dico[name].append(tag)
        else:
            dico[name]=[tag]
    gmsh.finalize()
    return dico

def checkPeriodicCurve(mshfile,curves):
    """
    Check if the curves are periodic in mesh ``meshfile``.

    Parameters
    ----------
    mshfile : str
        Mesh file.
        
    curves : list(dict)
        List containing two dictionaries with format
            {'type': 'disk|hline|vline', 'name'='PhysicalName'}
            
    Returns
    -------
    None.

    """
    
    __check_read_access(mshfile)
    
    if len(curves)!=2:
        raise ValueError("'curves' should be a list with two elements.")
    
        # Get curvilinear abscissa for each curve
    s = __checkPeriodicCurve_getS(mshfile,curves)
    s1=s[0]; s2=s[1]
    names=[curves[0]['name'],curves[1]['name']]

        # If the lengths of s1 and s2 differ by one, this is usually caused
        # by one of the boundary element missing (i.e. the first or last node 
        # of the curve)
        # This code is a rough attempt to correct s1 and s2 if needed.
    l1=len(s1); l2=len(s2)
    if (abs(l1-l2)==1): 
        l=min(l1,l2)
        print("Lengths differing by one element, but the curves may still be periodic.")
        if ( (s1[0]-s2[0])/(1+s2[0]) < 1e-10 ): # node 0 match
            # shorten the longest vector by removing its last node
            s1 = s1[0:l]; s2=s2[0:l]
        elif ( (s1[-1]-s2[-1])/(1+s2[-1]) < 1e-10 ):
            # shorten the longest vector by removing its first node
            s1 = s1[(l1-l):-1]; s2 = s2[(l2-l):-1]
        else:
            s1 = s1[(l1-l):l]; s2 = s2[(l2-l):l]

    if len(s1)!=len(s2):
        print(f"'{names[0]}' and '{names[1]}' are not periodic (different length).")
        delta = -1
    else:
        s2 = s2 + np.average(s1-s2) # guess offset
        delta = s2-s1 # compute error
        print(f"Periodicity error btw '{names[0]}' and '{names[1]}':"
              f"\n\tavg={np.average(delta)}, max={delta.max()}")
    return delta

def __checkPeriodicCurve_getS(mshfile,curves):
    # Get curvilinear abscissa for each curve in curves  
        # get physical names
    physGroup_1D = getPhysicalNames(mshfile,1)
    gmsh.initialize()
    gmsh.open(mshfile)
    s_lst=[] # coord
    for i in range(len(curves)):
        if type(curves[i])!=dict:
            raise ValueError(f"Element #{i} of 'curves' should be a dictionary.")
        if not('name' in curves[i]):
            raise ValueError("'curves[{i}]' should have key 'name'")
        if curves[i]['name'] in physGroup_1D:
            tag = physGroup_1D[curves[i]['name']]
        else:
            raise ValueError(f"'{curves[i]['name']}' is not a valid physical name.")
        if len(tag)>1:
            raise ValueError(f"'{curves[i]['name']}' should have just one tag in the mesh.")
        tag=tag[0]
            # Get coordinates
                # First possibility -> potential size mismatch
        (ntag,x)=gmsh.model.mesh.getNodesForPhysicalGroup(1,tag)
                # Second possibility -> correct size but can outright fail
        # (ntag,x)=gmsh.model.mesh.getNodes(1,tag,includeBoundary=True)[0:2]  
        x=x.reshape(-1,3)
            # Get curvilinear absissa from curve
        if curves[i]['type']=='disk':
            s=np.arctan2(x[:,0],x[:,1])
        if curves[i]['type']=='vline':
            s=x[:,1]
        if curves[i]['type']=='hline':
            s=x[:,0]
        s.sort()
            # Collect
        s_lst.append(s)
    gmsh.finalize()
    return s_lst

def __check_read_access(file):
    file = open(file, "r"); file.close()
    
def __check_write_access(file):
    file = open(file, "w"); file.close()
