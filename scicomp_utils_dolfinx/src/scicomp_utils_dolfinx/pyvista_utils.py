#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for pyvista.

"""

import pyvista as pv
from PIL import Image
from scientific_computing_utils import PIL_utils

def get_trimmed_screenshot(plotter,file=None):
    """
    Get trimmed screenshot from pyvista Plotter.

    Parameters
    ----------
    plotter : pyvista Plotter
        
    file : string (optional)
        Save trimmed screenshot to file.

    Returns
    -------
    img PIL.Image.Image
        Trimmed screenshot.

    """
    img = plotter.screenshot(transparent_background=True)
    img = Image.fromarray(img)
    img = PIL_utils.trim(img)
    if file!=None:
        img.save("result.png")
    return img
