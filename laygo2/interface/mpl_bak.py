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
    Convert a layout object to corresponding matplotlib patch object.
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
                return [[rect, obj.layer[0], objname + "/" + obj.netname]]
                # ax.add_patch(rect)
            return []
    elif obj.__class__ == laygo2.object.Text:
        return [["text", obj.layer[0], obj.text, obj.xy]]
        # TODO: implement text export function.
        pass
    elif obj.__class__ == laygo2.object.Instance:
        # Invoke _laygo2_generate_instance( cv name libname cellname viewname loc orient num_rows num_cols
        # sp_rows sp_cols params params_order )
        _xy = mxy + np.dot(obj.xy, tf.Mt(mtf).T)

        _xy0 = obj.xy0
        _xy1 = np.dot(obj.size, tf.Mt(mtf).T) #* np.array([num_rows, num_cols])
        rect = matplotlib.patches.Rectangle(
            (_xy0[0], _xy0[1]),
            _xy1[0],
            _xy1[1],
            facecolor=colormap["__instance__"][1],
            edgecolor=colormap["__instance__"][0],
            alpha=colormap["__instance__"][2],
            lw=2,
        )
        #pypobjs = [[rect, "__instance__", obj.cellname + "/" + obj.name]]
        pypobjs = [[rect, "__instance__", obj.cellname + "/" + objname]]

        # Elements for array instances
        if obj.shape is not None:
            for i, e in np.ndenumerate(obj.elements):
                pypobjs += _translate_obj(e.name+"_"+str(i), e, colormap, scale=scale)

        # Instance pins
        for pn, p in obj.pins.items():
            _pypobj = _translate_obj(pn, p, colormap, scale=scale, master=master, offset=offset)
            _pypobj[0][1] = "__instance_pin__"
            _pypobj[0][0].set(edgecolor=colormap["__instance_pin__"][0])
            _pypobj[0][0].set(facecolor=colormap["__instance_pin__"][1])
            pypobjs += _pypobj
        return pypobjs
    elif obj.__class__ == laygo2.object.VirtualInstance:
        pypobjs = []
        if obj.shape is None:
            for elem_name, elem in obj.native_elements.items():
                if not elem.__class__ == laygo2.object.Pin:
                    if obj.name == None:
                        objname = "NoName"
                    else:
                        objname = obj.name
                    _pypobj = _translate_obj(objname, elem, colormap, scale=scale, master=obj)
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

        # Instance pins
        for pn, p in obj.pins.items():
            _pypobj = _translate_obj(pn, p, colormap, scale=scale, master=master, offset=offset)
            _pypobj[0][1] = "__instance_pin__"
            _pypobj[0][0].set(edgecolor=colormap["__instance_pin__"][0])
            _pypobj[0][0].set(facecolor=colormap["__instance_pin__"][1])
            pypobjs += _pypobj
        return pypobjs

    else:
        return []
        # return [obj.translate_to_matplotlib()]
    return []


def export(
    db,
    cellname=None,
    scale=1,
    colormap=None,
    order=None,
    xlim=None,
    ylim=None,
    filename=None,
):
    """
    Export a laygo2.object.database.Library or Design object to a matplotlib plot.

    Parameters
    ----------
    db: laygo2.database.Library or laygo2.database design
        The library database or design to exported.
    cellname: str or List[str]
        (optional) The name(s) of cell(s) to be exported.
    scale: float
        (optional) The scaling factor between laygo2's integer coordinates and plot coordinates.
    colormap: dict
        A dictionary that contains layer-color mapping information.
    order: list
        A list that contains the order of layers to be displayed (former is plotted first).
    xlim: list
        (optional) A list that specifies the range of plot in x-axis.
    ylim: list
        (optional) A list that specifies the range of plot in y-axis.
    filename: str
        (optional) If specified, export a output file for the plot.

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
    if isinstance(db, laygo2.database.Design):
        cellname = [db.cellname]
        db = {db.cellname:db}
    if isinstance(db, laygo2.database.Library):
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
                    if isinstance(_pypobj[0], str):
                        if _pypobj[0] == 'text':  # Text
                            # [["text", obj.layer[0], obj.text, obj.xy]]
                            color = "black"
                            ax.annotate(
                                _pypobj[2], (_pypobj[3][0], _pypobj[3][1]), color=color, weight="bold", fontsize=6, ha="center", va="center"
                            )
                    elif _pypobj[0].__class__ == matplotlib.patches.Rectangle:  # Rect
                        ax.add_patch(_pypobj[0])
                        if len(_pypobj) == 3:  # annotation.
                            ax.add_artist(_pypobj[0])
                            rx, ry = _pypobj[0].get_xy()
                            cx = rx + _pypobj[0].get_width() / 2.0
                            cy = ry + _pypobj[0].get_height() / 2.0
                            if _pypobj[1] == "__instance_pin__":
                                color = _pypobj[0].get_edgecolor()
                            else:
                                color = "black"
                            ax.annotate(
                                _pypobj[2], (cx, cy), color=color, weight="bold", fontsize=6, ha="center", va="center"
                            )
        fig.append(_fig)
    if len(fig) == 1:
        fig = fig[0]

    # scale
    plt.autoscale()
    if not (xlim == None):
        plt.xlim(xlim)
    if not (ylim == None):
        plt.ylim(ylim)


    if filename is not None:
        plt.savefig(filename)

    plt.show()

    return fig


def export_instance(
    obj,
    scale=1,
    colormap=None,
    order=None,
    xlim=None,
    ylim=None,
    filename=None,
):
    """
    Export a laygo2.object.physical.Instance object to a matplotlib plot.

    Parameters
    ----------
    obj: laygo2.object.physical.Instance
        The instance object to exported.
    scale: float
        (optional) The scaling factor between laygo2's integer coordinates and plot coordinates.
    colormap: dict
        A dictionary that contains layer-color mapping information.
    order: list
        A list that contains the order of layers to be displayed (former is plotted first).
    xlim: list
        (optional) A list that specifies the range of plot in x-axis.
    ylim: list
        (optional) A list that specifies the range of plot in y-axis.
    filename: str
        (optional) If specified, export a output file for the plot.

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

    '''
    # xlim and ylim
    if xlim is None:
        xlim = [obj.bbox[0][0] - obj.width, obj.bbox[1][0] + obj.width]
    if ylim is None:
        ylim = [obj.bbox[0][1] - obj.height, obj.bbox[1][1] + obj.height]
    '''

    fig = plt.figure()
    pypobjs = []
    ax = fig.add_subplot(111)
    pypobjs += _translate_obj(obj.name, obj, colormap, scale=scale)
    for _alignobj in order:
        for _pypobj in pypobjs:
            if _pypobj[1] == _alignobj:  # layer is matched.
                if isinstance(_pypobj[0], str):
                    if _pypobj[0] == 'text':  # Text
                        # [["text", obj.layer[0], obj.text, obj.xy]]
                        color = "black"
                        ax.annotate(
                            _pypobj[2], (_pypobj[3][0], _pypobj[3][1]), color=color, weight="bold", fontsize=6, ha="center", va="center"
                        )
                elif _pypobj[0].__class__ == matplotlib.patches.Rectangle:  # Rect
                    ax.add_patch(_pypobj[0])
                    if len(_pypobj) == 3:  # annotation.
                        ax.add_artist(_pypobj[0])
                        rx, ry = _pypobj[0].get_xy()
                        cx = rx + _pypobj[0].get_width() / 2.0
                        cy = ry + _pypobj[0].get_height() / 2.0
                        if _pypobj[1] == "__instance_pin__":
                            color = _pypobj[0].get_edgecolor()
                        else:
                            color = "black"
                        ax.annotate(
                            _pypobj[2], (cx, cy), color=color, weight="bold", fontsize=6, ha="center", va="center"
                        )
    # scale
    plt.autoscale()
    if not (xlim == None):
        plt.xlim(xlim)
    if not (ylim == None):
        plt.ylim(ylim)

    if filename is not None:
        plt.savefig(filename)

    plt.show()

    return fig


def export_grid(
    obj,
    colormap=None,
    order=None,
    xlim=None,
    ylim=None,
    filename=None,
):
    """
    Export a laygo2.object.grid.Grid object to a matplotlib plot.

    Parameters
    ----------
    obj: laygo2.object.grid.Grid
        The grid object to exported.
    scale: float
        (optional) The scaling factor between laygo2's integer coordinates and plot coordinates.
    colormap: dict
        A dictionary that contains layer-color mapping information.
    order: list
        A list that contains the order of layers to be displayed (former is plotted first).
    xlim: list
        (optional) A list that specifies the range of plot in x-axis.
    ylim: list
        (optional) A list that specifies the range of plot in y-axis.
    filename: str
        (optional) If specified, export a output file for the plot.

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

    fig = plt.figure()
    pypobjs = []
    ax = fig.add_subplot(111)
    # scope
    _xy = (obj.vgrid.range[0], obj.hgrid.range[0])
    _width = obj.vgrid.range[1] - obj.vgrid.range[0]
    _height = obj.hgrid.range[1] - obj.hgrid.range[0]
    rect = matplotlib.patches.Rectangle(_xy, _width, _height, facecolor="none", edgecolor="black", lw=2)
    ax.add_patch(rect)
    rx, ry = rect.get_xy()
    cx = rx + rect.get_width() / 2.0
    cy = ry + rect.get_height() / 2.0
    ax.annotate(
        obj.name, (cx, cy), color="black", weight="bold", fontsize=6, ha="center", va="center"
    )
    if obj.__class__ == laygo2.object.RoutingGrid:  # Routing grid
        for i in range(len(obj.vgrid.elements)):  # vertical routing grid
            ve = obj.vgrid.elements[i]
            _xy = (ve - obj.vwidth[i]/2, obj.hgrid.range[0]-obj.vextension[i])
            _width = obj.vwidth[i]
            _height = obj.hgrid.range[1] - obj.hgrid.range[0] + 2 * obj.vextension[i]
            facecolor=colormap[obj.vlayer[i][0]][1]
            edgecolor=colormap[obj.vlayer[i][0]][0]
            alpha=colormap[obj.vlayer[i][0]][2]
            rect = matplotlib.patches.Rectangle(_xy, _width, _height, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, lw=2)
            ax.add_patch(rect)
            _xy = (ve, obj.hgrid.range[0]-obj.vextension[i])
            ax.annotate(_xy[0], _xy, color="black", weight="bold", fontsize=6, ha="center", va="center")
        for i in range(len(obj.hgrid.elements)):  # horizontal routing grid
            he = obj.hgrid.elements[i]
            _xy = (obj.vgrid.range[0] - obj.hextension[i], he - obj.hwidth[i]/2)
            _width = obj.vgrid.range[1] - obj.vgrid.range[0] + 2*obj.hextension[i]
            _height = obj.hwidth[i]
            facecolor=colormap[obj.hlayer[i][0]][1]
            edgecolor=colormap[obj.hlayer[i][0]][0]
            alpha=colormap[obj.hlayer[i][0]][2]
            rect = matplotlib.patches.Rectangle(_xy, _width, _height, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, lw=2)
            ax.add_patch(rect)
            _xy = (obj.vgrid.range[0] - obj.hextension[i], he)
            ax.annotate(_xy[1], _xy, color="black", weight="bold", fontsize=6, ha="center", va="center")
        # viamap
        for i in range(len(obj.vgrid.elements)):  # vertical routing grid
            for j in range(len(obj.hgrid.elements)):  # horizontal routing grid
                v = obj.viamap[i, j]
                x = obj.vgrid.elements[i]
                y = obj.hgrid.elements[j]
                circ = matplotlib.patches.Circle((x, y), radius=2, facecolor="black", edgecolor="black") #, **kwargs)
                ax.add_patch(circ)
                ax.annotate(v.name, (x+2, y), color="black", fontsize=4, ha="left", va="bottom")

    for _alignobj in order:
        for _pypobj in pypobjs:
            if _pypobj[1] == _alignobj:  # layer is matched.
                if isinstance(_pypobj[0], str):
                    if _pypobj[0] == 'text':  # Text
                        # [["text", obj.layer[0], obj.text, obj.xy]]
                        color = "black"
                        ax.annotate(
                            _pypobj[2], (_pypobj[3][0], _pypobj[3][1]), color=color, weight="bold", fontsize=6, ha="center", va="center"
                        )
                elif _pypobj[0].__class__ == matplotlib.patches.Rectangle:  # Rect
                    ax.add_patch(_pypobj[0])
                    if len(_pypobj) == 3:  # annotation.
                        ax.add_artist(_pypobj[0])
                        rx, ry = _pypobj[0].get_xy()
                        cx = rx + _pypobj[0].get_width() / 2.0
                        cy = ry + _pypobj[0].get_height() / 2.0
                        if _pypobj[1] == "__instance_pin__":
                            color = _pypobj[0].get_edgecolor()
                        else:
                            color = "black"
                        ax.annotate(
                            _pypobj[2], (cx, cy), color=color, weight="bold", fontsize=6, ha="center", va="center"
                        )

    # scale
    plt.autoscale()
    if not (xlim == None):
        plt.xlim(xlim)
    if not (ylim == None):
        plt.ylim(ylim)

    if filename is not None:
        plt.savefig(filename)

    plt.show()

    return fig
