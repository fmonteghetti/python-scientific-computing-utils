#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script demonstrates root finding.
"""

import numpy as np
import scipy.special
from scientific_computing_utils import scipy_utils
import matplotlib.pyplot as plt
        

    # Spherical Bessel function
n = 1
f = lambda x: scipy.special.spherical_jn(n,x)

xmin,xmax = 0, 10
zeros = scipy_utils.find_zeros(f,xmin,xmax,0.1)
x = np.linspace(xmin,xmax,num=100)
plt.plot(x,f(x))
plt.plot(zeros,0*zeros,marker='o',linestyle='none')
plt.title("Zeros of f(x)")
plt.xlabel("x")
plt.ylabel("f(x)")



