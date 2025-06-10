#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on August 8 2022

@author: dzhigd
"""
import numpy as np

def cosd(input):
    """
    Calculate cosine for degrees value
    """
    return np.cos(np.radians(input))

def sind(input):
    """
    Calculate sine for degrees value
    """
    return np.sin(np.radians(input))

def tand(input):
    """
    Calculate tangent for degrees value
    """
    return np.tan(np.radians(input))