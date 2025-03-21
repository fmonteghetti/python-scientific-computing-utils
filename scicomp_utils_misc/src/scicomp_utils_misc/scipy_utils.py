#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions built around scipy library.

"""

import numpy as np


def find_zeros(f, xmin, xmax, dx):
    """
    Dumb rootfinding function based on Brent’s method.

    The interval [xmin,xmax] is divided into sub-interval of length dx, and
    Brent's method is applied to each of these sub-interval. At most one zero
    can be found in each sub-interval.

    Parameters
    ----------
    f : scalar function
    xmin, xmax : float
        Interval in which to look for roots.
    dx : float

    Returns
    -------
    zeros : np.array([])
        A subset of the zeros of f in [xmin,xmax].
    """
    from scipy.optimize import brentq

    N = np.ceil((xmax - xmin) / dx)
    # divide [xmin,xmax] in N subintervals
    x = np.linspace(xmin, xmax, num=int(N + 1))

    zeros = np.array([])
    for i in range(len(x) - 1):  # for each subinterval [x[i],x[i+1]]
        try:
            z = brentq(f, a=x[i], b=x[i + 1], full_output=False, disp=False)
        except ValueError:  # root finding failed in [a,b]
            z = []
        zeros = np.append(zeros, z)
    return zeros
