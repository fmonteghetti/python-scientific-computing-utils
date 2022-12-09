#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for Python Imaging Library.

"""
from PIL import Image, ImageChops
def trim(im):
    """ Trim image, from stackoverflow.com/questions/10615901/trim-whitespace-using-pil"""
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
        # discard alpha channel
    diff = ImageChops.difference(im, bg).convert('RGB')
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)
