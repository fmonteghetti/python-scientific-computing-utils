#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script solves a high-dimensional index-1 DAE using IDA through Assimulo. 
Its objective is to investigate the various options.

"""


import numpy as np
import pylab as P
from assimulo.solvers import IDA
from assimulo.problem import Implicit_Problem

# Create a class that inherists form Implicit Problem

#%% Problem from TutorialIDAE.py
class DAE_description(Implicit_Problem):
        
        # default constructor
    def __init__(self,y0,yd0,name):
        self.y0 = y0
        self.yd0 = yd0
        self.name = name
        
    #Defines the residual
    def res(self,t,y,yd):
        res_0 = yd[0] - y[1]
        res_1 = yd[1] + 9.82
        print(f'res t={t}')
        return np.array([res_0,res_1])
    
    #Defines the Jacobian*vector product
    def jacv(self,t,y,yd,res,v,c):
        j_0 = c*v[0] - v[1]
        j_1 = c*v[1]
        print(f'jac t={t}')
        return np.array([j_0,j_1])

#Initial conditions
y0 = np.array([1.0,0.0]) 
yd0 = 0*np.array([0.0, -9.82])

#Defines an Assimulo implicit problem
imp_mod = DAE_description(y0,yd0,name = 'Example using the Jacobian Vector product')
#imp_mod.algvar = idx


#%% High-dimensional index-1 DAE
class DAE_description(Implicit_Problem):
        # default constructor
    def __init__(self,src_diff,src_alg,name):
        self.N_diff = src_diff.size
        self.N_alg = src_alg.size
        self.src_diff = src_diff
        self.src_alg = src_alg
        self.algvar = np.ones((self.N_diff+self.N_alg,),dtype="bool")
        self.algvar[self.N_diff+np.array(range(self.N_alg))] = False
        self.y0 = np.zeros((self.N_diff+self.N_alg))
        self.yd0 = np.zeros((self.N_diff+self.N_alg))
        self.name = name
        self.res_count = 0
        self.jac_count = 0
        

    #Defines the residual
    def res(self,t,y,yd):
        self.res_count += 1
        res_diff = yd[self.algvar] + y[self.algvar] - self.src_diff
        res_alg = y[~self.algvar] - self.src_alg        
        return np.concatenate((res_diff,res_alg))
    
    #Defines the Jacobian*vector product
    def jacv(self,t,y,yd,res,v,c):
        self.jac_count += 1
        res_diff = c*v[self.algvar] + v[self.algvar]
        res_alg = v[~self.algvar]     
        return np.concatenate((res_diff,res_alg))

    # Create problem
N_diff = int(1e4)
N_alg = int(1e2)
src_diff = np.ones((N_diff,)) 
src_alg = 1*(np.random.rand(N_alg)) 
imp_mod = DAE_description(src_diff,src_alg,name = 'High-dim index-1 DAE')
    # Create IDA solver instance
imp_sim = IDA(imp_mod)
    # Tolerances
imp_sim.atol = 1e-5 #Default 1e-6
imp_sim.rtol = 1e-5 #Default 1e-6
    # Solver
imp_sim.linear_solver = 'SPGMR'
    # Consistent initialization
imp_sim.make_consistent("IDA_YA_YDP_INIT")
imp_sim.tout1=0.001
print(f"Residual call: {imp_sim.problem.res_count}, Jacobian call: {imp_sim.problem.jac_count}")
    # Misc
imp_sim.suppress_alg = True
imp_sim.suppress_sens = True
    # Display
imp_sim.report_continuously = True
imp_sim.display_progress = True
    # Time integration
imp_sim.re_init(0.0, imp_mod.y0+0.5, imp_mod.yd0)
tf = 5
imp_sim.maxh = tf/10
imp_sim.maxsteps = int(1e3)
t, y, yd = imp_sim.simulate(tf, 20)
    # Plot
P.plot(t,y[:,N_diff-1])
P.xlabel('Time')
P.ylabel('State')
P.title(imp_mod.name)
P.show()