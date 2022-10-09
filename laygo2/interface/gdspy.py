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
This module implements interfaces with gds files via gdspy.
"""

import logging
import pprint
from math import log10
from decimal import *
import laygo2.util.transform as tf

import numpy as np

import laygo2.object


def _load_layermap(layermapfile):
    """
    Load layermap information from layermapfile (Foundry techfile can be used)

    Parameters
    ----------
    layermapfile : str
        layermap filename.

        The example file can be found in default.layermap or see below:
        #technology layer information
        #layername  layerpurpose stream# datatype
        text        drawing 100 0
        prBoundary  drawing 101 0
        M1      drawing 50  0
        M1      pin     50  10
        M2      drawing 51  0
        M2      pin     51  10

    Returns
    -------
    dict
        constructed layermap information.

    """
    layermap = dict()
    f = open(layermapfile, "r")
    for line in f:
        tokens = line.split()
        if not len(tokens) == 0:
            if not tokens[0].startswith("#"):
                name = tokens[0]
                # if not layermap.has_key(name):
                if name not in layermap:
                    layermap[name] = dict()
                layermap[name][tokens[1]] = {
                    "layer": int(tokens[2]),
                    "datatype": int(tokens[3]),
                }
    return layermap


def _translate_obj(
    objname,
    obj,
    layermap,
    scale=0.001,
    master=None,
    offset=np.array([0, 0]),
    pin_label_height=0.1,
):
    """
    Convert an object to corresponding skill commands.
    offset : np.array([int, int])
        Offsets to obj.xy
    """
    # import gdspy here to avoid unnecessary C++ compliations for non-gds options.
    import gdspy

    if master is None:
        mxy = np.array([0, 0])
        mtf = "R0"
    else:  # if the translated object has a master (e.g. VirtualInstance)
        mxy = master.xy
        mtf = master.transform

    if obj.__class__ == laygo2.object.Rect:
        ## TODO: add color handling.
        # color = obj.color # coloring function example for skill.

        _xy = np.sort(obj.xy, axis=0)  # make sure obj.xy is sorted
        _xy = mxy + np.dot(
            _xy
            + np.array(
                [[-obj.hextension, -obj.vextension], [obj.hextension, obj.vextension]]
            ),
            tf.Mt(mtf).T,
        )

        l = layermap[obj.layer[0]][obj.layer[1]]
        rect = gdspy.Rectangle((_xy[0, 0], _xy[0, 1]), (_xy[1, 0], _xy[1, 1]), **l)
        return rect
    elif obj.__class__ == laygo2.object.Path:
        # TODO: implement path export function.
        pass
    elif obj.__class__ == laygo2.object.Pin:
        if obj.elements is None:
            _objelem = [obj]
        else:
            _objelem = obj.elements
        item = []
        for idx, _obj in np.ndenumerate(_objelem):
            _xy = mxy + np.dot(_obj.xy, tf.Mt(mtf).T)
            l = layermap[_obj.layer[0]][_obj.layer[1]]
            rect = gdspy.Rectangle((_xy[0, 0], _xy[0, 1]), (_xy[1, 0], _xy[1, 1]), **l)
            _xy_c = 0.5 * (_xy[0, :] + _xy[1, :])
            text = gdspy.Label(
                _obj.netname, _xy_c, "nw", magnification=pin_label_height * 100
            )
            item += [rect, text]
        return item
    elif obj.__class__ == laygo2.object.Text:
        # TODO: implement text export function.
        pass
    elif obj.__class__ == laygo2.object.Instance:
        print(
            "[Warning] laygo2.interface.gdspy: Instance transform is not implemented yet."
        )
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
        # if obj.params is None:  # gds cannot handle pcell parameters.
        #    inst_params = "nil"
        # else:
        #    inst_params = _py2skill_inst_params(obj.params['pcell_params'])
        inst = gdspy.CellReference(obj.cellname, _xy)  # , transform)
        return inst
    elif obj.__class__ == laygo2.object.VirtualInstance:
        item = []
        if obj.shape is None:
            for elem_name, elem in obj.native_elements.items():
                if not elem.__class__ == laygo2.object.Pin:
                    if obj.name == None:
                        obj.name = "NoName"
                    else:
                        pass
                    item += [
                        _translate_obj(
                            obj.name + "_" + elem_name,
                            elem,
                            layermap=layermap,
                            master=obj,
                            scale=scale,
                            pin_label_height=pin_label_height,
                        )
                    ]
        else:  # arrayed VirtualInstance
            for i, j in np.ndindex(tuple(obj.shape.tolist())):  # iterate over obj.shape
                for elem_name, elem in obj.native_elements.items():
                    if not elem.__class__ == laygo2.object.Pin:
                        item += [
                            _translate_obj(
                                obj.name + "_" + elem_name + str(i) + "_" + str(j),
                                elem,
                                layermap=layermap,
                                master=obj[i, j],
                                scale=scale,
                                pin_label_height=pin_label_height,
                            )
                        ]
        return item
    return None
    # raise Exception("No corresponding GDS structure for:"+obj.summarize())


def export(
    db,
    filename,
    cellname=None,
    scale=1e-9,
    layermapfile="default.layermap",
    physical_unit=1e-9,
    logical_unit=0.001,
    pin_label_height=0.1,
    svg_filename=None,
    png_filename=None,
):
    """
    Export a laygo2.object.database.Library object to a gds file via gdspy.

    Parameters
    ----------
    db: laygo2.database.Library
        The library database to exported.
    filename: str, optional
        The name of output gds file.
    cellname: str or List[str]
        The name(s) of cell(s) to be exported.
    scale: float
        The scaling factor between laygo2's integer coordinats actual physical coordinates.
    layermapfile : str
        the name of layermap file.
    physical_unit : float, optional
        GDS physical unit.
    logical_unit : float, optional
        GDS logical unit.
    pin_label_height : float, optional
        the height of pin label.
    svg_filename: str, optional
        If specified, it exports a svg file with the specified filename.
    svg_filename: str, optional
        If specified, it exports a png file with the specified filename
        (svg_filename needs to be specified as well).

    Example
    --------
    >>> import laygo2
    >>> from laygo2.object.database import Design
    >>> from laygo2.object.physical import Rect, Pin, Instance, Text
    >>> # Create a design.
    >>> dsn = Design(name="mycell", libname="genlib")
    >>> # Create layout objects.
    >>> r0 = Rect(xy=[[0, 0], [100, 100]], layer=["M1", "drawing"])
    >>> p0 = Pin(xy=[[0, 0], [50, 50]], layer=["M1", "pin"], name="P")
    >>> i0 = Instance(libname="tlib", cellname="t0", name="I0", xy=[0, 0])
    >>> t0 = Text(xy=[[50, 50], [100, 100]], layer=["text", "drawing"], text="T")
    >>> # Add the layout objects to the design object.
    >>> dsn.append(r0)
    >>> dsn.append(p0)
    >>> dsn.append(i0)
    >>> dsn.append(t0)
    >>> #
    >>> # Export to a gds file.
    >>> lib = laygo2.object.database.Library(name="mylib")
    >>> lib.append(dsn)
    >>> laygo2.interface.gds.export(lib, filename="mylayout.gds")
    """
    # Compute scale parameter.
    _scale = round(1 / scale * physical_unit / logical_unit)
    # 1um in phy
    # 1um/1nm = 1000 in laygo2 if scale = 1e-9 (1nm)
    # 1000/1nm*1nm/0.001 = 1000000 in gds if physical_unit = 1e-9 (1nm) and logical_unit = 0.001

    # Load layermap file.
    layermap = _load_layermap(layermapfile)  # load layermap information

    # Construct cellname.
    cellname = (
        db.keys() if cellname is None else cellname
    )  # export all cells if cellname is not given.
    cellname = (
        [cellname] if isinstance(cellname, str) else cellname
    )  # convert to a list for iteration.

    # import gdspy here to avoid unnecessary C++ compliations for non-gds options.
    import gdspy

    # Create library.
    lib = gdspy.GdsLibrary()
    for cn in cellname:
        # Create cell.
        cell = lib.new_cell(cn)
        # Translate objects.
        for objname, obj in db[cn].items():
            tobj = _translate_obj(
                objname,
                obj,
                layermap=layermap,
                scale=_scale,
                pin_label_height=pin_label_height,
            )
            if tobj is not None:
                cell.add(tobj)
    lib.write_gds(filename)
    if svg_filename is not None:
        cell.write_svg(svg_filename)
        if svg_filename is not None:
            # import cairosvg here to avoid unnecessary lib installation for non-gds options.
            import cairosvg

            cairosvg.svg2png(url=svg_filename, write_to=png_filename, scale=1.0)
    # gdspy.LayoutViewer()
