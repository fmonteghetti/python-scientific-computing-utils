#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Utility functions relying (mostly) on the python standard library.

"""

import numpy as np


def build_global_indices_list(l):
    """
    Build global indices to access elements of list l.

        Example: The input list of list
        l = [ l0, [], l1 ] with len(l0)=2 and len(l1)=3
    produces the output list
        [[0,1], [], [2,3,4]].

    :param l: list of list
    :return: list of np.array
    """
    l_range = [np.arange(len(el)) for el in l]
    l_len = [len(el) for el in l]
    idx = [l_range[i] + sum(l_len[0:i]) for i in range(len(l))]
    return idx


def sum_list(l):
    """
    Sum element of a list. Useful for non-numerical lists.
    """

    if len(l) == 1:
        return l
    else:
        return sum(l[1:], l[0])
