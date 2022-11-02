#!/usr/bin/python
########################################################################################################################
#
# Copyright (c) 2020, Nifty Chips Laboratory, Hanyang University
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
This module implements the interface with matplotlib.

"""
import logging
from math import log10
from decimal import *

import numpy as np
import laygo2.object
import laygo2.util.transform as tf

import matplotlib
import matplotlib.pyplot as plt

__author__ = ""
__maintainer__ = ""
__status__ = "Prototype"


def _translate_obj(objname, obj, colormap, scale=1, master=None, offset=np.array([0, 0])):
    """
    Convert an object to corresponding matplotlib object.
    offset : np.array([int, int])
        Offsets to obj.xy
    """
    if master is None:
        mxy = np.array([0, 0])
        mtf = "R0"
    else:  # if the translated object has a master (e.g. VirtualInstance)
        mxy = master.xy
        mtf = master.transform
    if obj.__class__ == laygo2.object.Rect:
        # Draw a rectangle
        _xy = np.sort(obj.xy, axis=0)  # make sure obj.xy is sorted
        _xy = mxy + np.dot(
            _xy + np.array([[-obj.hextension, -obj.vextension], [obj.hextension, obj.vextension]]),
            tf.Mt(mtf).T,
        )
        size = [_xy[1, 0] - _xy[0, 0], _xy[1, 1] - _xy[0, 1]]
        if obj.layer[0] in colormap:
            rect = matplotlib.patches.Rectangle(
                (_xy[0, 0], _xy[0, 1]),
                size[0],
                size[1],
                facecolor=colormap[obj.layer[0]][1],
                edgecolor=colormap[obj.layer[0]][0],
                alpha=colormap[obj.layer[0]][2],
                lw=2,
            )
            # ax.add_patch(rect)
            return [[rect, obj.layer[0]]]
        return []
    elif obj.__class__ == laygo2.object.Path:
        # TODO: implement path export function.
        pass
    elif obj.__class__ == laygo2.object.Pin:
        if obj.elements is None:
            _objelem = [obj]
        else:
            _objelem = obj.elements
        for idx, _obj in np.ndenumerate(_objelem):
            # Invoke _laygo2_generate_pin(cv, name, layer, bbox) in {header_filename}
            _xy = mxy + np.dot(_obj.xy, tf.Mt(mtf).T)
            size = [_xy[1, 0] - _xy[0, 0], _xy[1, 1] - _xy[0, 1]]
            if obj.layer[0] in colormap:
                rect = matplotlib.patches.Rectangle(
                    (_xy[0, 0], _xy[0, 1]),
                    size[0],
                    size[1],
                    facecolor=colormap[obj.layer[0]][1],
                    edgecolor=colormap[obj.layer[0]][0],
                    alpha=colormap[obj.layer[0]][2],
                    lw=2,
                )
                return [[rect, obj.layer[0]]]
                # ax.add_patch(rect)
            return []
    elif obj.__class__ == laygo2.object.Text:
        # TODO: implement text export function.
        pass
    elif obj.__class__ == laygo2.object.Instance:
        # Invoke _laygo2_generate_instance( cv name libname cellname viewname loc orient num_rows num_cols
        # sp_rows sp_cols params params_order )
        _xy = mxy + np.dot(obj.xy, tf.Mt(mtf).T)
        if master is None:
            transform = obj.transform
        else:  # if the translated object has a master (e.g. VirtualInstance)
            transform = tf.combine(obj.transform, master.transform)
        if obj.shape is None:
            num_rows = 1
            num_cols = 1
            sp_rows = 0
            sp_cols = 0
        else:
            num_rows = obj.shape[1]
            num_cols = obj.shape[0]
            sp_rows = obj.pitch[1]
            sp_cols = obj.pitch[0]

        _xy0 = obj.xy0
        _xy1 = np.dot(_obj.size, tf.Mt(mtf).T) * np.array([num_rows, num_cols])
        rect = matplotlib.patches.Rectangle(
            (_xy0[0], _xy0[1]), _xy1[0], _xy1[1], color="yellow", edgecolor="black", lw=2
        )
        return [[rect, None]]
        # ax.add_patch(rect)
        # return True
    elif obj.__class__ == laygo2.object.VirtualInstance:
        pypobjs = []
        if obj.shape is None:
            for elem_name, elem in obj.native_elements.items():
                if not elem.__class__ == laygo2.object.Pin:
                    if obj.name == None:
                        obj.name = "NoName"
                    else:
                        pass
                    _pypobj = _translate_obj(obj.name + "_" + elem_name, elem, colormap, scale=scale, master=obj)
                    pypobjs += _pypobj
        else:  # arrayed VirtualInstance
            for i, j in np.ndindex(tuple(obj.shape.tolist())):  # iterate over obj.shape
                for elem_name, elem in obj.native_elements.items():
                    if not elem.__class__ == laygo2.object.Pin:
                        _pypobj = _translate_obj(
                            obj.name + "_" + elem_name + str(i) + "_" + str(j),
                            elem,
                            colormap,
                            scale=scale,
                            master=obj[i, j],
                        )
                        pypobjs += _pypobj
        return pypobjs
    else:
        return [obj.translate_to_matplotlib()]  #

    return []


def export(
    db,
    cellname=None,
    scale=1,
    colormap=None,
    order=None,
    xlim=[-100, 400],
    ylim=[-100, 300],
):
    """
    Export a laygo2.object.database.Library object to a matplotlib plot.

    Parameters
    ----------
    db: laygo2.database.Library
        The library database to exported.
    cellname: str or List[str]
        The name(s) of cell(s) to be exported.
    scale: float
        The scaling factor between laygo2's integer coordinates and plot coordinates.

    Returns
    -------
    matplotlib.pyplot.figure or list: The generated figure object(s).

    """
    # colormap
    if colormap is None:
        colormap = dict()

    # a list to align layered objects in order
    if order is None:
        order = []

    # cell name handling.
    cellname = db.keys() if cellname is None else cellname  # export all cells if cellname is not given.
    cellname = [cellname] if isinstance(cellname, str) else cellname  # convert to a list for iteration.

    fig = []
    for cn in cellname:
        _fig = plt.figure()
        pypobjs = []
        ax = _fig.add_subplot(111)
        for objname, obj in db[cn].items():
            pypobjs += _translate_obj(objname, obj, colormap, scale=scale)
        for _alignobj in order:
            for _pypobj in pypobjs:
                if _pypobj[1] == _alignobj:  # layer is matched.
                    if _pypobj[0].__class__ == matplotlib.patches.Rectangle:  # Rect
                        ax.add_patch(_pypobj[0])
        fig.append(_fig)
    if len(fig) == 1:
        fig = fig[0]

    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.show()

    return fig
