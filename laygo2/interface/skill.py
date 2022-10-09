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
This module implements interface with virtuoso in skill language

"""
import logging
from math import log10
from decimal import *

import numpy as np
import laygo2.object
import laygo2.util.transform as tf

__author__ = ""
__maintainer__ = ""
__status__ = "Prototype"


def _py2skill_number(value, scale=0.001):
    fmt_str = "%." + "%d" % (-1 * log10(scale) + 1) + "f "  # for truncations
    return fmt_str % (value * scale)


def _py2skill_float(value, scale=0.001):
    exp = np.ceil(-1 * log10(value) + 1)
    fmt_str = "%." + "%d" % (-1 * log10(scale) + 1) + "f"  # for truncations
    # convert to exponentail expression
    print(value, exp, str(value))
    return fmt_str % (value * (10**exp) * scale) + "e" + str(int(-1 * exp))


def _py2skill_list(pylist, scale=0.001):
    """Convert a python list object to a skill list."""
    list_str = "list( "
    for item in pylist:
        if isinstance(item, list):  # nested list
            list_str += _py2skill_list(item) + " "
        elif isinstance(item, np.ndarray):  # nested list
            list_str += _py2skill_list(item, scale=scale) + " "
        elif isinstance(item, str):
            list_str += '"' + str(item) + '" '
        elif isinstance(item, bool):
            if item:
                list_str += "t "
            else:
                list_str += "nil "
        elif isinstance(item, int) or isinstance(item, np.integer):
            # fmt_str = "%."+"%d" % (-1*log10(scale)+1)+"f "  # for truncations
            # list_str += fmt_str%(item*scale) + " "
            list_str += _py2skill_number(item, scale) + " "
    list_str += ")"
    return list_str


def _py2skill_inst_params_list(pylist):
    """Convert a python list object to a skill list. (for pcell parameters)"""
    list_str = "list( "
    for item in pylist:
        if isinstance(item, list):  # nested list
            list_str += _py2skill_inst_params_list(item) + " "
        elif isinstance(item, np.ndarray):  # nested list
            list_str += _py2skill_inst_params_list(item) + " "
        elif isinstance(item, str):
            list_str += '"' + str(item) + '" '
        elif isinstance(item, float):
            list_str += str(item)
            # list_str += _py2skill_float(item, scale) + " "
        elif isinstance(item, bool):
            if item:
                list_str += "t "
            else:
                list_str += "nil "
        elif isinstance(item, int) or isinstance(item, np.integer):
            # fmt_str = "%."+"%d" % (-1*log10(scale)+1)+"f "  # for truncations
            # list_str += fmt_str%(item*scale) + " "
            # list_str += _py2skill_number(item, 1) + " "  # do not scale integers
            list_str += str(item) + " "  # do not scale integers
    list_str += ")"
    return list_str


def _py2skill_inst_params(value_dict):
    """Convert instance parameter dictionary to skill list"""
    _list = []
    for k, v in value_dict.items():
        if isinstance(v, str):
            _type = "string"
        elif isinstance(v, float):
            _type = "float"
        elif isinstance(v, bool):
            _type = "boolean"
        elif isinstance(v, int) or isinstance(v, np.integer):
            _type = "int"
        _list.append([k, _type, v])
    return _py2skill_inst_params_list(_list)


def _translate_obj(objname, obj, scale=0.001, master=None, offset=np.array([0, 0])):
    """
    Convert an object to corresponding skill commands.
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
        color = obj.color  # coloring func. added
        # Invoke _laygo2_generate_rect( cv layer bbox ) in {header_filename}
        _xy = np.sort(obj.xy, axis=0)  # make sure obj.xy is sorted
        _xy = mxy + np.dot(
            _xy
            + np.array(
                [[-obj.hextension, -obj.vextension], [obj.hextension, obj.vextension]]
            ),
            tf.Mt(mtf).T,
        )
        # _xy = mxy + np.dot(obj.xy + np.array([[-obj.hextension, -obj.vextension], [obj.hextension, obj.vextension]]),
        #                   tf.Mt(mtf).T)
        return '_laygo2_generate_rect(cv, %s, %s, "%s") ; for the Rect object %s \n' % (
            _py2skill_list(obj.layer),
            _py2skill_list(_xy, scale=scale),
            color,
            objname,
        )  # coloring func. added
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
            return (
                '_laygo2_generate_pin(cv, "%s", %s, %s ) ; for the Pin object %s \n'
                % (
                    _obj.netname,
                    _py2skill_list(_obj.layer),
                    _py2skill_list(_xy, scale=scale),
                    objname,
                )
            )
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
            sp_rows = _py2skill_number(obj.pitch[1])
            sp_cols = _py2skill_number(obj.pitch[0])
        if obj.params is None:
            inst_params = "nil"
        else:
            inst_params = _py2skill_inst_params(obj.params["pcell_params"])
            # inst_params = _py2skill_list([["Wfg", "string", "500n"], ["fingers", "string", "4"], ["l", "string", "500n"]])
        return (
            '_laygo2_generate_instance(cv, "%s", "%s", "%s", "%s", %s, "%s", %d, %d, %s, %s, %s, %s) '
            "; for the Instance object %s \n"
            % (
                objname,
                obj.libname,
                obj.cellname,
                obj.viewname,
                _py2skill_list(_xy),
                transform,
                num_rows,
                num_cols,
                sp_rows,
                sp_cols,
                inst_params,
                "nil",
                objname,
            )
        )
    elif obj.__class__ == laygo2.object.VirtualInstance:
        cmd = ""
        if obj.shape is None:
            for elem_name, elem in obj.native_elements.items():
                if not elem.__class__ == laygo2.object.Pin:
                    if obj.name == None:
                        obj.name = "NoName"
                    else:
                        pass
                    cmd += _translate_obj(
                        obj.name + "_" + elem_name, elem, scale=scale, master=obj
                    )
        else:  # arrayed VirtualInstance
            for i, j in np.ndindex(tuple(obj.shape.tolist())):  # iterate over obj.shape
                for elem_name, elem in obj.native_elements.items():
                    if not elem.__class__ == laygo2.object.Pin:
                        cmd += _translate_obj(
                            obj.name + "_" + elem_name + str(i) + "_" + str(j),
                            elem,
                            scale=scale,
                            master=obj[i, j],
                        )
        return cmd
    else:
        return obj.translate_to_skill()  #

    return ""


def export(
    db, filename=None, cellname=None, scale=1e-3, reset_library=False, tech_library=None
):
    """
    Export a laygo2.object.database.Library object to skill code.

    Parameters
    ----------
    db: laygo2.database.Library
        The library database to exported.
    filename: str, optional
        If specified, the generated skill script is stored in filename.
    cellname: str or List[str]
        The name(s) of cell(s) to be exported.
    scale: float
        The scaling factor between laygo2's integer coordinats actual physical coordinates.
    reset_library: bool, optional
        If True, the library to export the cells is reset.
    tech_library: str, optional
        The name of technology library to be attached to the resetted library.

    Returns
    -------
    str: The generated skill script.

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
    >>> # Export to skill.
    >>> lib = laygo2.object.database.Library(name="mylib")
    >>> lib.append(dsn)
    >>> scr = laygo2.interface.skill.export(lib, filename="myscript.il")
    >>> print(scr)
    ; (definitions of laygo2 skill functions)
    ; exporting mylib__mycell
    cv = _laygo2_open_layout("mylib" "mycell" "layout")
    _laygo2_generate_rect(cv, list( "M1" "drawing" ), list( list( 0.0000  0.0000  ) list( 0.1000  0.1000  ) ), "None")
    _laygo2_generate_pin(cv, "P", list( "M1" "pin" ), list( list( 0.0000  0.0000  ) list( 0.0500  0.0500  ) ) )
    _laygo2_generate_instance(cv, "I0", "tlib", "t0", "layout", list( 0.0000  0.0000  ), "R0", 1, 1, 0, 0, nil, nil)
    _laygo2_save_and_close_layout(cv)

    """
    # parse header functions.
    cmd = "; laygo2 layout export skill script.\n\n"
    import os

    header_filename = (
        os.path.abspath(laygo2.interface.__file__)[:-11] + "skill_export.il"
    )
    with open(header_filename, "r") as f:
        cmd += f.read()
        cmd += "\n"

    cellname = (
        db.keys() if cellname is None else cellname
    )  # export all cells if cellname is not given.
    cellname = (
        [cellname] if isinstance(cellname, str) else cellname
    )  # convert to a list for iteration.
    if reset_library:
        cmd += '_laygo2_reset_library("%s" "%s")\n' % (db.name, tech_library)
    for cn in cellname:
        cmd += "\n; exporting %s__%s\n" % (db.name, cn)  # open the design.
        logging.debug("Export_to_SKILL: Cellname:" + cn)
        # cmd += "let( (cv) \n"  # limit the scope of cv. Disabled to prevent overflow.
        cmd += 'cv = _laygo2_open_layout("%s" "%s" "layout")\n' % (
            db.name,
            cn,
        )  # open the design.
        # export objects
        for objname, obj in db[cn].items():
            cmd += _translate_obj(objname, obj, scale=scale)
        cmd += "_laygo2_save_and_close_layout(cv)\n"  # close the layout
        # cmd += ");let\n" Disabled to prevent overflow.
    if filename is not None:  # export to a file.
        with open(filename, "w") as f:
            f.write(cmd)
    return cmd
