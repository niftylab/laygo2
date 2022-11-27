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
This module implements interface with BAG in skill language
"""
import laygo2.interface
import yaml
import numpy as np

def export(
    db, filename, cellname=None, scale=1e-3, reset_library=False, tech_library=None
):
    """
    Export a laygo2.object.database.Library object to BAG2.

    Parameters
    ----------
    db: laygo2.object.database.Library
        The library database to be exported.
    filename: str, optional
        The path of the intermediate skill script file.
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
    -------
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
    >>> # Export to BAG.
    >>> lib = laygo2.object.database.Library(name="mylib")
    >>> lib.append(dsn)
    >>> scr = laygo2.interface.bag.export(lib, filename="myscript.il")
    >>> print(scr)
    ; (definitions of laygo2 skill functions)
    ; exporting mylib__mycell
    cv = _laygo2_open_layout("mylib" "mycell" "layout")
    _laygo2_generate_rect(cv, list( "M1" "drawing" ), list( list( 0.0000  0.0000  ) list( 0.1000  0.1000  ) ), "None")
    _laygo2_generate_pin(cv, "P", list( "M1" "pin" ), list( list( 0.0000  0.0000  ) list( 0.0500  0.0500  ) ) )
    _laygo2_generate_instance(cv, "I0", "tlib", "t0", "layout", list( 0.0000  0.0000  ), "R0", 1, 1, 0, 0, nil, nil)
    _laygo2_save_and_close_layout(cv)

    """
    skill_str = laygo2.interface.skill.export(
        db, filename, cellname, scale, reset_library, tech_library
    )
    import bag

    prj = bag.BagProject()
    prj.impl_db._eval_skill('load("' + filename + '");1\n')
    print("Your design was generated in Virtuoso.")

    return skill_str

def load(libname, cellname=None, filename=None, yaml_filename="import_skill_scratch.yaml", scale=1e-3, mpt=False):
    """
    Import virtuoso layout to a laygo2.object.database.Library object via BAG2 interface.
    """
    import bag
    prj = bag.BagProject()

    if cellname==None: #import all cells
        skill_str = laygo2.interface.skill.load_cell_list(libname, filename, yaml_filename)
        with open(yamlfile, 'r') as stream:
            ydict = yaml.load(stream)
            celllist=ydict[libname]
        prj.impl_db._eval_skill('load("' + filename + '");1\n')
    else:
        if isinstance(cellname, list): celllist=cellname
        else: celllist=[cellname]
    
    db = laygo2.object.database.Library(name=libname)
    for cn in celllist:
        skill_str = laygo2.interface.skill.load(libname, cellname=cellname, filename=filename, 
                                                  yaml_filename=yaml_filename, mpt=mpt)
        prj.impl_db._eval_skill('load("' + filename + '");1\n')
        with open(yaml_filename, 'r') as stream:
            ydict = yaml.load(stream)
        dsn = laygo2.object.database.Design(name=cn)
        db.append(dsn)
        for _r_key, _r in ydict['rects'].items():
            if 'color' in _r:  # to support MPT
                _color = _r['color']
            else:
                _color = None
            r = laygo2.object.Rect(xy=np.array(_r['bBox'])/scale, layer=_r['layer'].split(), color = _color)
            dsn.append(r)
        for _t_key, _t in ydict['labels'].items():
            t = laygo2.object.Text(text=_t['label'], xy=np.array(_t['xy'])/scale, layer=_t['layer'].split())
            dsn.append(t)
        for _i_key, _i in ydict['instances'].items():
            if not 'rows' in _i: _i['rows']=1
            if not 'cols' in _i: _i['cols']=1
            if not 'sp_rows' in _i: _i['sp_rows']=0
            if not 'sp_cols' in _i: _i['sp_cols']=0
            if not 'transform' in _i: _i['transform']='R0'
            inst = laygo2.object.Instance(xy = np.array(_i['xy'])/scale, libname = _i['lib_name'], cellname = _i['cell_name'],
                                          shape = np.array([_i['cols'], _i['rows']]), pitch = np.array([_i['sp_cols'], _i['sp_rows']]),
                                          transform = _i['transform']) 
            dsn.append(inst)
    return db


