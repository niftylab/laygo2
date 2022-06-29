#!/usr/bin/python
########################################################################################################################
#
# Copyright (c) 2014, Regents of the University of California
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
#   disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
#    following disclaimer in the documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################################################################

"""
Utility functions for coordinate tranformations.
"""

__author__ = "Jaeduk Han"
__maintainer__ = "Jaeduk Han"
__status__ = "Prototype"

import numpy as np

def combine(transform1, transform2):
    """
    Returns the resulting transform parameter of 
    two consecutive transforms
    """
    if transform1 == 'R0':
        if transform2 == 'R0': return 'R0'
        elif transform2 == 'MX': return 'MX'
        elif transform2 == 'MY': return 'MY'
        elif transform2 == 'MXY': return 'MXY'
    elif transform1 == 'MX':
        if transform2 == 'R0': return 'MX'
        elif transform2 == 'MX': return 'R0'
        elif transform2 == 'MY': return 'MXY'
        elif transform2 == 'MXY': return 'MY'
    elif transform1 == 'MY':
        if transform2 == 'R0': return 'MY'
        elif transform2 == 'MX': return 'MXY'
        elif transform2 == 'MY': return 'R0'
        elif transform2 == 'MXY': return 'MX'
    raise ValueError("Transformation mapping is not matched.")


def Mt(transform):
    """
    Returns the transform matrix.

    Parameters
    ----------
    transform : str
        The transform parameter. Possible values are 'R0', 'MX', 'MY', 'MXY', and 'R180'.

    Returns
    -------
    numpy.ndarray(dtype=int)
        The transform matrix corresponding to the transform parameter.

    """
    transform_map = {
        'R0': np.array([[1, 0], [0, 1]]),
        'R90': np.array([[0, -1], [1, 0]]),
        'R180': np.array([[-1, 0], [0, -1]]),
        'R270': np.array([[0, 1], [-1, 0]]),
        'MX': np.array([[1, 0], [0, -1]]),
        'MY': np.array([[-1, 0], [0, 1]]),
        'MXY': np.array([[0, 1], [1, 0]]),  # mirror to the y=x line.
    }
    return transform_map[transform]


def Mtinv(transform):
    """
    Returns the inverse of the transform matrix.

    Parameters
    ----------
    transform : str
        The transform parameter. possible values are 'R0', 'MX', 'MY', 'MXY', and 'R180'.

    Returns
    -------
    numpy.ndarray(dtype=int)
        The inverse of the transform matrix.
    """
    transform_map = {
        'R0': np.array([[1, 0], [0, 1]]),    'MX': np.array([[1, 0], [0, -1]]),
        'MY': np.array([[-1, 0], [0, 1]]),
        'MXY': np.array([[0, 1], [1, 0]]),  # mirror to the y=x line.
        'R180': np.array([[-1, 0], [0, -1]]),
    }
    return transform_map[transform]


def Md(direction):
    """
    Returns the direction(projection) matrix. The direction matrix is used when placing an object based on relative
    information to other instance(s). For example, if an instance's center is located at xyc0=[xc0, yc0],
    the xy-coordinate of the center of the new instance xyc1 can be computed from the following equation:

    (1) xyc1 = xyc0 + 0.5 * Md * (xys0 + xys1)

    where xys0, xys1 are the size of the reference and the new instance, respectively, and Md is the direction matrix
    corresponding to the direction of the placement.

    Parameters
    ----------
    direction : str
        The direction parameter. Possible values are 'left', 'right', 'top', 'bottom', 'omni', 'x', 'y'.

    Returns
    -------
    np.array([[int, int], [int, int]])
        The direction matrix.

    Notes
    -----
    The following equation will be used instead of (1) in the future versions, to avoid the 0.5 scaling that increases
    the precision requirement.

    (2) xy1 = xy0 + 0.5 * [(Md + Mt0) * xys0 + (Md - Mt1) * xys1]
    """
    direction_map = {
        'left': np.array([[-1, 0], [0, 0]]),
        'right': np.array([[1, 0], [0, 0]]),
        'top': np.array([[0, 0], [0, 1]]),
        'bottom': np.array([[0, 0], [0, -1]]),
        'omni': np.array([[1, 0], [0, 1]]),  # omnidirectional
        'x': np.array([[1, 0], [0, 0]]),
        'y': np.array([[0, 0], [0, 1]]),
    }
    return direction_map[direction]


