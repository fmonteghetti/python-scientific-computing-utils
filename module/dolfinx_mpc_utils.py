#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 
Utility functions for dolfinx_mpc.
 
"""


import dolfinx.fem
import dolfinx_mpc
import  numpy as np
from petsc4py import PETSc

class PeriodicBoundary():
    """
    Class that describes a list of periodic boundary conditions. It is
    designed as a wrapper around dolfinx_mpc.MultiPointConstraint.

    Example
    --------
        # Create and initialize class
    pbc = PeriodicBoundary()
        # Add periodic boundary conditions
        # Create a finalized dolfinx_mpc.MultiPointConstraint  
    pbc.create_finalized_MultiPointConstraint(V, facet_tags,bcs)
    mpc = pbc.get_MultiPointConstraint()
        # Compute sol
        # Set slave dofs
    pbc.set_slave_dofs(sol)

    """
    
    def __init__(self):
        self.__pbc = list() # list of periodic boundary conditions
        self.__mpc = None   # MultiPointConstraint object

    
    def add_topological_condition(self,slave_tag,slave_to_master_map,
                                  slave_map=lambda x: [], 
                                  master_map=lambda x: []):
        """
        Add a topological periodic boundary condition.
        
        Parameters
        ----------
        slave_tag : list(int)
            tag value(s) of the slave boundary in mesh.
            
        slave_to_master_map : function x -> y
            geometrical map of slave boundary to master boundary
            
        slave_map : function x -> boolean (Optional)
            indicator function for slave boundary. Only needed if the boundary
            condition may be overlapped by one of the next conditions.

        master_map : function x -> boolean (Optional)
            indicator function for master boundary. Only needed if the boundary
            condition may be overlapped by one of the next conditions.

        """
        if not isinstance(slave_tag, list):
            raise TypeError("slave_tag should be a list of integers.")
            
        pbc_dict = {'slave_tag': slave_tag,
                    'slave_to_master_map': slave_to_master_map,
                    'master_map': master_map,
                    'slave_map': slave_map}
        self.__pbc.append(pbc_dict)
           
    def __patch_slave_to_master_map(self):
        """
        Patch __pbc to ensure that no dofs are constrained twice, to avoid
        dolfin_mpc failing silently. Require overlapped pbc to provide
        'is_slave' and 'is_master' maps.
        """ 
        
        for i, p in enumerate(self.__pbc):
            if i==0: # first pbc does not overlap any other
                self.__pbc[0]['slave_to_master_map_patched'] = \
                                        self.__pbc[0]['slave_to_master_map']
            else: # Patch __pbc[i]['slave_to_master_map'] to exclude dofs that 
                  # are master or slaves of one of the i-1 previously-defined
                  # periodic boundary conditions.
                def slave_to_master_map(x,p,pbc,i):
                    out_x = p['slave_to_master_map'](x)
                    for q in pbc[0:i]:
                        # index of overlapping slave nodes (empty if slave_map and 
                        # master_map are empty)
                        idx_s = q['slave_map'](x) + q['master_map'](x)
                        # index of overlapping master nodes
                        idx_m = q['slave_map'](out_x) + q['master_map'](out_x)
                        out_x[0][idx_s] = np.nan
                        out_x[0][idx_m] = np.nan
                    return out_x
                p['slave_to_master_map_patched'] = \
                    lambda x, p=p, pbc=self.__pbc, i=i: slave_to_master_map(x,p,pbc,i)
        
    def create_finalized_MultiPointConstraint(self,V,facet_tags,bcs):
        """
        
        Create a finalized MultiPointConstraint.

        Parameters
        ----------
        V : dolfinx.fem.function.FunctionSpace
        facet_tags : Out[63]: dolfinx.cpp.mesh.MeshTags
        bcs : list(dolfinx.fem.DirichletBCMetaClass)

        """
                
        self.__patch_slave_to_master_map()        
        self.__mpc = dolfinx_mpc.MultiPointConstraint(V)
        for p in self.__pbc:
            for slave_tag in p['slave_tag']:
                self.__mpc.create_periodic_constraint_topological(V, facet_tags,
                                                        slave_tag,
                                                        p['slave_to_master_map_patched'],
                                                        bcs)
        self.__mpc.finalize()

    def set_slave_dofs(self,vec):
        """
        Set slave dofs from master dofs.

        Parameters
        ----------
        vec : list(petsc4py.PETSc.Vec) or petsc4py.PETSc.Vec

        """
        if not isinstance(vec, list):
            v_list=[vec]
        else:
            v_list=vec
            # Backsubstituting using a Function ensures proper results
            # in multi-threaded execution
        f = dolfinx.fem.Function(self.__mpc.function_space)
        for i in range(len(v_list)):
            f.vector.setArray(v_list[i])
            f.x.scatter_forward()
            self.__mpc.backsubstitution(f.vector)
            v_list[i] = f.vector.copy()
        
    def get_MultiPointConstraint(self):
        """
        Return an initialized dolfinx_mpc.MultiPointConstraint object or None.

        Returns
        -------
        dolfinx_mpc.MultiPointConstraint

        """
        return self.__mpc