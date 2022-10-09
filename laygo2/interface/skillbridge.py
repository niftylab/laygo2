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
This module implements interface with Skillbridge in skill language
"""
import laygo2.interface


def export(
    db,
    filename,
    cellname,
    scale=1e-3,
    reset_library=False,
    tech_library=None,
    pyserver_id=None,
):
    """
    Export a laygo2.object.database.Library object to Cadence Virtuoso via 
    skillbridge.

    Parameters
    ----------
    db: laygo2.database.Library
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
    >>> # Export to skillbridge.
    >>> lib = laygo2.object.database.Library(name="mylib")
    >>> lib.append(dsn)
    >>> scr = laygo2.interface.skillbridge.export(lib, filename="myscript.il")
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

    # Export skill script to skillbridge
    import os
    from skillbridge import Workspace

    # ws = Workspace.open(os.environ['USER'])
    if pyserver_id is None:
        ws = Workspace.open()
    else:
        ws = Workspace.open(pyserver_id)
    ws["load"](filename)

    return skill_str
