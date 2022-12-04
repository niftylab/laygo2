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

import numpy as np
from .core import CircularMapping, Grid
import laygo2.object

class RoutingGrid(Grid):
    """
    A class that implements wire connections in an abstract coordinate system.

    Notes
    -----
    **(Korean)** 추상 좌표계 상의 배선 동작을 구현하는 클래스.
    """

    type = "routing"
    """ Type of grid. Should be 'routing' for routing grids."""

    vwidth = None
    """CircularMapping: Width of vertical wires.

    Example
    -------
    >>> import laygo2
    >>> from laygo2.object.grid import CircularMapping as CM
    >>> from laygo2.object.grid import CircularMappingArray as CMA
    >>> from laygo2.object.grid import OneDimGrid, RoutingGrid
    >>> from laygo2.object.template import NativeInstanceTemplate
    >>> # Routing grid construction (not needed if laygo2_tech is set up).
    >>> gv = OneDimGrid(name="gv", scope=[0, 50], elements=[0])
    >>> gh = OneDimGrid(name="gh", scope=[0, 100], elements=[0, 40, 60])
    >>> wv = CM([10])           # vertical (xgrid) width
    >>> wh = CM([20, 10, 10])   # horizontal (ygrid) width
    >>> ev = CM([10])           # vertical (xgrid) extension
    >>> eh = CM([10, 10, 10])   # horizontal (ygrid) extension
    >>> e0v = CM([15])          # vert. extension (for zero-length wires)
    >>> e0h = CM([15, 15, 15])  # hori. extension (for zero-length wires)
    >>> lv = CM([['M1', 'drawing']], dtype=object)  # layer information
    >>> lh = CM([['M2', 'drawing']]*3, dtype=object) 
    >>> plv = CM([['M1', 'pin']], dtype=object) # pin layers
    >>> plh = CM([['M2', 'pin']]*3, dtype=object)
    >>> xcolor = CM([None], dtype=object)  # not multi-patterned 
    >>> ycolor = CM([None]*3, dtype=object) 
    >>> primary_grid = 'horizontal'
    >>> tvia = NativeInstanceTemplate(libname='tlib', cellname='via0')  # via 
    >>> viamap = CMA(elements=[[tvia, tvia, tvia]], dtype=object)
    >>> g = laygo2.object.grid.RoutingGrid(name='mygrid', vgrid=gv, hgrid=gh,
                                           vwidth=wv, hwidth=wh,
                                           vextension=ev, hextension=eh,
                                           vlayer=lv, hlayer=lh,
                                           pin_vlayer=plv, pin_hlayer=plh,
                                           viamap=viamap, primary_grid=primary_grid,
                                           xcolor=xcolor, ycolor=ycolor,
                                           vextension0=e0v, hextension0=e0h)
    >>> print(g.vwidth)
    <laygo2.object.grid.CircularMapping object > 
        class: CircularMapping, 
        elements: [10]

    .. image:: ../assets/img/object_grid_RoutingGrid_vwidth.png
           :height: 250

    See Also
    --------
    vwidth, hwidth, vextension, hextension, vextension0, hextension0

    Notes
    -----
    **(Korean)** 수직 wire들의 폭.
    """

    hwidth = None
    """CircularMapping: Width of horizontal wires.

    Example
    -------
    >>> import laygo2
    >>> from laygo2.object.grid import CircularMapping as CM
    >>> from laygo2.object.grid import CircularMappingArray as CMA
    >>> from laygo2.object.grid import OneDimGrid, RoutingGrid
    >>> from laygo2.object.template import NativeInstanceTemplate
    >>> # Routing grid construction (not needed if laygo2_tech is set up).
    >>> gv = OneDimGrid(name="gv", scope=[0, 50], elements=[0])
    >>> gh = OneDimGrid(name="gv", scope=[0, 100], elements=[0, 40, 60])
    >>> wv = CM([10])           # vertical (xgrid) width
    >>> wh = CM([20, 10, 10])   # horizontal (ygrid) width
    >>> ev = CM([10])           # vertical (xgrid) extension
    >>> eh = CM([10, 10, 10])   # horizontal (ygrid) extension
    >>> e0v = CM([15])          # vert. extension (for zero-length wires)
    >>> e0h = CM([15, 15, 15])  # hori. extension (for zero-length wires)
    >>> lv = CM([['M1', 'drawing']], dtype=object)  # layer information
    >>> lh = CM([['M2', 'drawing']]*3, dtype=object) 
    >>> plv = CM([['M1', 'pin']], dtype=object) # pin layers
    >>> plh = CM([['M2', 'pin']]*3, dtype=object)
    >>> xcolor = CM([None], dtype=object)  # not multi-patterned 
    >>> ycolor = CM([None]*3, dtype=object) 
    >>> primary_grid = 'horizontal'
    >>> tvia = NativeInstanceTemplate(libname='tlib', cellname='via0')  # via 
    >>> viamap = CMA(elements=[[tvia, tvia, tvia]], dtype=object)
    >>> g = laygo2.object.grid.RoutingGrid(name='mygrid', vgrid=gv, hgrid=gh,
                                           vwidth=wv, hwidth=wh,
                                           vextension=ev, hextension=eh,
                                           vlayer=lv, hlayer=lh,
                                           pin_vlayer=plv, pin_hlayer=plh,
                                           viamap=viamap, primary_grid=primary_grid,
                                           xcolor=xcolor, ycolor=ycolor,
                                           vextension0=e0v, hextension0=e0h)
    >>> print(g.hwidth)
    <laygo2.object.grid.CircularMapping object > 
        class: CircularMapping, 
        elements: [20, 10, 10]

    .. image:: ../assets/img/object_grid_RoutingGrid_hwidth.png
           :height: 250 

    See Also
    --------
    vwidth, hwidth, vextension, hextension, vextension0, hextension0

    Notes
    -----
    **(Korean)** 수평 wire들의 폭.
    """

    vextension = None
    """CircularMapping: Extension of vertical wires.

    Example
    -------
    >>> import laygo2
    >>> from laygo2.object.grid import CircularMapping as CM
    >>> from laygo2.object.grid import CircularMappingArray as CMA
    >>> from laygo2.object.grid import OneDimGrid, RoutingGrid
    >>> from laygo2.object.template import NativeInstanceTemplate
    >>> # Routing grid construction (not needed if laygo2_tech is set up).
    >>> gv = OneDimGrid(name="gv", scope=[0, 50], elements=[0])
    >>> gh = OneDimGrid(name="gv", scope=[0, 100], elements=[0, 40, 60])
    >>> wv = CM([10])           # vertical (xgrid) width
    >>> wh = CM([20, 10, 10])   # horizontal (ygrid) width
    >>> ev = CM([10])           # vertical (xgrid) extension
    >>> eh = CM([10, 10, 10])   # horizontal (ygrid) extension
    >>> e0v = CM([15])          # vert. extension (for zero-length wires)
    >>> e0h = CM([15, 15, 15])  # hori. extension (for zero-length wires)
    >>> lv = CM([['M1', 'drawing']], dtype=object)  # layer information
    >>> lh = CM([['M2', 'drawing']]*3, dtype=object) 
    >>> plv = CM([['M1', 'pin']], dtype=object) # pin layers
    >>> plh = CM([['M2', 'pin']]*3, dtype=object)
    >>> xcolor = CM([None], dtype=object)  # not multi-patterned 
    >>> ycolor = CM([None]*3, dtype=object) 
    >>> primary_grid = 'horizontal'
    >>> tvia = NativeInstanceTemplate(libname='tlib', cellname='via0')  # via 
    >>> viamap = CMA(elements=[[tvia, tvia, tvia]], dtype=object)
    >>> g = laygo2.object.grid.RoutingGrid(name='mygrid', vgrid=gv, hgrid=gh,
                                           vwidth=wv, hwidth=wh,
                                           vextension=ev, hextension=eh,
                                           vlayer=lv, hlayer=lh,
                                           pin_vlayer=plv, pin_hlayer=plh,
                                           viamap=viamap, primary_grid=primary_grid,
                                           xcolor=xcolor, ycolor=ycolor,
                                           vextension0=e0v, hextension0=e0h)
    >>> print(g.vextension)
    <laygo2.object.grid.CircularMapping object > 
        class: CircularMapping, 
        elements: [10]

    .. image:: ../assets/img/object_grid_RoutingGrid_vextension.png
           :height: 250

    Notes
    -----
    **(Korean)** 수직 wire들의 extension.
    """

    hextension = None
    """CircularMapping: Extension of horizontal wires.

    Example
    -------
    >>> import laygo2
    >>> from laygo2.object.grid import CircularMapping as CM
    >>> from laygo2.object.grid import CircularMappingArray as CMA
    >>> from laygo2.object.grid import OneDimGrid, RoutingGrid
    >>> from laygo2.object.template import NativeInstanceTemplate
    >>> # Routing grid construction (not needed if laygo2_tech is set up).
    >>> gv = OneDimGrid(name="gv", scope=[0, 50], elements=[0])
    >>> gh = OneDimGrid(name="gv", scope=[0, 100], elements=[0, 40, 60])
    >>> wv = CM([10])           # vertical (xgrid) width
    >>> wh = CM([20, 10, 10])   # horizontal (ygrid) width
    >>> ev = CM([10])           # vertical (xgrid) extension
    >>> eh = CM([10, 10, 10])   # horizontal (ygrid) extension
    >>> e0v = CM([15])          # vert. extension (for zero-length wires)
    >>> e0h = CM([15, 15, 15])  # hori. extension (for zero-length wires)
    >>> lv = CM([['M1', 'drawing']], dtype=object)  # layer information
    >>> lh = CM([['M2', 'drawing']]*3, dtype=object) 
    >>> plv = CM([['M1', 'pin']], dtype=object) # pin layers
    >>> plh = CM([['M2', 'pin']]*3, dtype=object)
    >>> xcolor = CM([None], dtype=object)  # not multi-patterned 
    >>> ycolor = CM([None]*3, dtype=object) 
    >>> primary_grid = 'horizontal'
    >>> tvia = NativeInstanceTemplate(libname='tlib', cellname='via0')  # via 
    >>> viamap = CMA(elements=[[tvia, tvia, tvia]], dtype=object)
    >>> g = laygo2.object.grid.RoutingGrid(name='mygrid', vgrid=gv, hgrid=gh,
                                           vwidth=wv, hwidth=wh,
                                           vextension=ev, hextension=eh,
                                           vlayer=lv, hlayer=lh,
                                           pin_vlayer=plv, pin_hlayer=plh,
                                           viamap=viamap, primary_grid=primary_grid,
                                           xcolor=xcolor, ycolor=ycolor,
                                           vextension0=e0v, hextension0=e0h)
    >>> print(g.hextension)
    <laygo2.object.grid.CircularMapping object > 
        class: CircularMapping, 
        elements: [10, 10, 10]

    .. image:: ../assets/img/object_grid_RoutingGrid_hextension.png
           :height: 250 

    Notes
    -----
    **(Korean)** 수평 wire들의 extension.
    """

    vextension0 = None
    """CircularMapping: the array containing the extension of the zero-length wires on the vertical grid.
    
    Example
    -------
    >>> import laygo2
    >>> from laygo2.object.grid import CircularMapping as CM
    >>> from laygo2.object.grid import CircularMappingArray as CMA
    >>> from laygo2.object.grid import OneDimGrid, RoutingGrid
    >>> from laygo2.object.template import NativeInstanceTemplate
    >>> # Routing grid construction (not needed if laygo2_tech is set up).
    >>> gv = OneDimGrid(name="gv", scope=[0, 50], elements=[0])
    >>> gh = OneDimGrid(name="gv", scope=[0, 100], elements=[0, 40, 60])
    >>> wv = CM([10])           # vertical (xgrid) width
    >>> wh = CM([20, 10, 10])   # horizontal (ygrid) width
    >>> ev = CM([10])           # vertical (xgrid) extension
    >>> eh = CM([10, 10, 10])   # horizontal (ygrid) extension
    >>> e0v = CM([15])          # vert. extension (for zero-length wires)
    >>> e0h = CM([15, 15, 15])  # hori. extension (for zero-length wires)
    >>> lv = CM([['M1', 'drawing']], dtype=object)  # layer information
    >>> lh = CM([['M2', 'drawing']]*3, dtype=object) 
    >>> plv = CM([['M1', 'pin']], dtype=object) # pin layers
    >>> plh = CM([['M2', 'pin']]*3, dtype=object)
    >>> xcolor = CM([None], dtype=object)  # not multi-patterned 
    >>> ycolor = CM([None]*3, dtype=object) 
    >>> primary_grid = 'horizontal'
    >>> tvia = NativeInstanceTemplate(libname='tlib', cellname='via0')  # via 
    >>> viamap = CMA(elements=[[tvia, tvia, tvia]], dtype=object)
    >>> g = laygo2.object.grid.RoutingGrid(name='mygrid', vgrid=gv, hgrid=gh,
                                           vwidth=wv, hwidth=wh,
                                           vextension=ev, hextension=eh,
                                           vlayer=lv, hlayer=lh,
                                           pin_vlayer=plv, pin_hlayer=plh,
                                           viamap=viamap, primary_grid=primary_grid,
                                           xcolor=xcolor, ycolor=ycolor,
                                           vextension0=e0v, hextension0=e0h)
    >>> print(g.vextension0)
    <laygo2.object.grid.CircularMapping object > 
        class: CircularMapping, 
        elements: [15]
    """

    hextension0 = None
    """CircularMapping: the array containing the extension of the zero-length wires on the horizontal grid. 
    
    Example
    -------
    >>> import laygo2
    >>> from laygo2.object.grid import CircularMapping as CM
    >>> from laygo2.object.grid import CircularMappingArray as CMA
    >>> from laygo2.object.grid import OneDimGrid, RoutingGrid
    >>> from laygo2.object.template import NativeInstanceTemplate
    >>> # Routing grid construction (not needed if laygo2_tech is set up).
    >>> gv = OneDimGrid(name="gv", scope=[0, 50], elements=[0])
    >>> gh = OneDimGrid(name="gv", scope=[0, 100], elements=[0, 40, 60])
    >>> wv = CM([10])           # vertical (xgrid) width
    >>> wh = CM([20, 10, 10])   # horizontal (ygrid) width
    >>> ev = CM([10])           # vertical (xgrid) extension
    >>> eh = CM([10, 10, 10])   # horizontal (ygrid) extension
    >>> e0v = CM([15])          # vert. extension (for zero-length wires)
    >>> e0h = CM([15, 15, 15])  # hori. extension (for zero-length wires)
    >>> lv = CM([['M1', 'drawing']], dtype=object)  # layer information
    >>> lh = CM([['M2', 'drawing']]*3, dtype=object) 
    >>> plv = CM([['M1', 'pin']], dtype=object) # pin layers
    >>> plh = CM([['M2', 'pin']]*3, dtype=object)
    >>> xcolor = CM([None], dtype=object)  # not multi-patterned 
    >>> ycolor = CM([None]*3, dtype=object) 
    >>> primary_grid = 'horizontal'
    >>> tvia = NativeInstanceTemplate(libname='tlib', cellname='via0')  # via 
    >>> viamap = CMA(elements=[[tvia, tvia, tvia]], dtype=object)
    >>> g = laygo2.object.grid.RoutingGrid(name='mygrid', vgrid=gv, hgrid=gh,
                                           vwidth=wv, hwidth=wh,
                                           vextension=ev, hextension=eh,
                                           vlayer=lv, hlayer=lh,
                                           pin_vlayer=plv, pin_hlayer=plh,
                                           viamap=viamap, primary_grid=primary_grid,
                                           xcolor=xcolor, ycolor=ycolor,
                                           vextension0=e0v, hextension0=e0h)
    >>> print(g.hextension0)
    <laygo2.object.grid.CircularMapping object > 
        class: CircularMapping, 
        elements: [15, 15, 15]
    """

    vlayer = None
    """CircularMapping: Layer information of vertical wires.

    Example
    -------
    >>> import laygo2
    >>> from laygo2.object.grid import CircularMapping as CM
    >>> from laygo2.object.grid import CircularMappingArray as CMA
    >>> from laygo2.object.grid import OneDimGrid, RoutingGrid
    >>> from laygo2.object.template import NativeInstanceTemplate
    >>> # Routing grid construction (not needed if laygo2_tech is set up).
    >>> gv = OneDimGrid(name="gv", scope=[0, 50], elements=[0])
    >>> gh = OneDimGrid(name="gv", scope=[0, 100], elements=[0, 40, 60])
    >>> wv = CM([10])           # vertical (xgrid) width
    >>> wh = CM([20, 10, 10])   # horizontal (ygrid) width
    >>> ev = CM([10])           # vertical (xgrid) extension
    >>> eh = CM([10, 10, 10])   # horizontal (ygrid) extension
    >>> e0v = CM([15])          # vert. extension (for zero-length wires)
    >>> e0h = CM([15, 15, 15])  # hori. extension (for zero-length wires)
    >>> lv = CM([['M1', 'drawing']], dtype=object)  # layer information
    >>> lh = CM([['M2', 'drawing']]*3, dtype=object) 
    >>> plv = CM([['M1', 'pin']], dtype=object) # pin layers
    >>> plh = CM([['M2', 'pin']]*3, dtype=object)
    >>> xcolor = CM([None], dtype=object)  # not multi-patterned 
    >>> ycolor = CM([None]*3, dtype=object) 
    >>> primary_grid = 'horizontal'
    >>> tvia = NativeInstanceTemplate(libname='tlib', cellname='via0')  # via 
    >>> viamap = CMA(elements=[[tvia, tvia, tvia]], dtype=object)
    >>> g = laygo2.object.grid.RoutingGrid(name='mygrid', vgrid=gv, hgrid=gh,
                                           vwidth=wv, hwidth=wh,
                                           vextension=ev, hextension=eh,
                                           vlayer=lv, hlayer=lh,
                                           pin_vlayer=plv, pin_hlayer=plh,
                                           viamap=viamap, primary_grid=primary_grid,
                                           xcolor=xcolor, ycolor=ycolor,
                                           vextension0=e0v, hextension0=e0h)
    >>> print(g.vlayer)
    <laygo2.object.grid.CircularMapping object > 
        class: CircularMapping, 
        elements: [['M1', 'drawing']]

    .. image:: ../assets/img/object_grid_RoutingGrid_vlayer.png
           :height: 250

    Notes
    -----
    **(Korean)** 수직 wire들의 레이어 정보.
    """

    hlayer = None
    """CircularMapping: Layer information of horizontal wires.

    Example
    -------
    >>> import laygo2
    >>> from laygo2.object.grid import CircularMapping as CM
    >>> from laygo2.object.grid import CircularMappingArray as CMA
    >>> from laygo2.object.grid import OneDimGrid, RoutingGrid
    >>> from laygo2.object.template import NativeInstanceTemplate
    >>> # Routing grid construction (not needed if laygo2_tech is set up).
    >>> gv = OneDimGrid(name="gv", scope=[0, 50], elements=[0])
    >>> gh = OneDimGrid(name="gv", scope=[0, 100], elements=[0, 40, 60])
    >>> wv = CM([10])           # vertical (xgrid) width
    >>> wh = CM([20, 10, 10])   # horizontal (ygrid) width
    >>> ev = CM([10])           # vertical (xgrid) extension
    >>> eh = CM([10, 10, 10])   # horizontal (ygrid) extension
    >>> e0v = CM([15])          # vert. extension (for zero-length wires)
    >>> e0h = CM([15, 15, 15])  # hori. extension (for zero-length wires)
    >>> lv = CM([['M1', 'drawing']], dtype=object)  # layer information
    >>> lh = CM([['M2', 'drawing']]*3, dtype=object) 
    >>> plv = CM([['M1', 'pin']], dtype=object) # pin layers
    >>> plh = CM([['M2', 'pin']]*3, dtype=object)
    >>> xcolor = CM([None], dtype=object)  # not multi-patterned 
    >>> ycolor = CM([None]*3, dtype=object) 
    >>> primary_grid = 'horizontal'
    >>> tvia = NativeInstanceTemplate(libname='tlib', cellname='via0')  # via 
    >>> viamap = CMA(elements=[[tvia, tvia, tvia]], dtype=object)
    >>> g = laygo2.object.grid.RoutingGrid(name='mygrid', vgrid=gv, hgrid=gh,
                                           vwidth=wv, hwidth=wh,
                                           vextension=ev, hextension=eh,
                                           vlayer=lv, hlayer=lh,
                                           pin_vlayer=plv, pin_hlayer=plh,
                                           viamap=viamap, primary_grid=primary_grid,
                                           xcolor=xcolor, ycolor=ycolor,
                                           vextension0=e0v, hextension0=e0h)
    >>> print(g.hlayer)
    <laygo2.object.grid.CircularMapping object > 
        class: CircularMapping, 
        elements: [['M1', 'drawing'], ['M1', 'drawing'], ['M1', 'drawing']]

    .. image:: ../assets/img/object_grid_RoutingGrid_hlayer.png
           :height: 250

    Notes
    -----
    **(Korean)** 수평 wire들의 레이어정보.
    """

    pin_vlayer = None
    """CircularMapping: Layer information of vertical pin wires.

    Example
    -------
    >>> import laygo2
    >>> from laygo2.object.grid import CircularMapping as CM
    >>> from laygo2.object.grid import CircularMappingArray as CMA
    >>> from laygo2.object.grid import OneDimGrid, RoutingGrid
    >>> from laygo2.object.template import NativeInstanceTemplate
    >>> # Routing grid construction (not needed if laygo2_tech is set up).
    >>> gv = OneDimGrid(name="gv", scope=[0, 50], elements=[0])
    >>> gh = OneDimGrid(name="gv", scope=[0, 100], elements=[0, 40, 60])
    >>> wv = CM([10])           # vertical (xgrid) width
    >>> wh = CM([20, 10, 10])   # horizontal (ygrid) width
    >>> ev = CM([10])           # vertical (xgrid) extension
    >>> eh = CM([10, 10, 10])   # horizontal (ygrid) extension
    >>> e0v = CM([15])          # vert. extension (for zero-length wires)
    >>> e0h = CM([15, 15, 15])  # hori. extension (for zero-length wires)
    >>> lv = CM([['M1', 'drawing']], dtype=object)  # layer information
    >>> lh = CM([['M2', 'drawing']]*3, dtype=object) 
    >>> plv = CM([['M1', 'pin']], dtype=object) # pin layers
    >>> plh = CM([['M2', 'pin']]*3, dtype=object)
    >>> xcolor = CM([None], dtype=object)  # not multi-patterned 
    >>> ycolor = CM([None]*3, dtype=object) 
    >>> primary_grid = 'horizontal'
    >>> tvia = NativeInstanceTemplate(libname='tlib', cellname='via0')  # via 
    >>> viamap = CMA(elements=[[tvia, tvia, tvia]], dtype=object)
    >>> g = laygo2.object.grid.RoutingGrid(name='mygrid', vgrid=gv, hgrid=gh,
                                           vwidth=wv, hwidth=wh,
                                           vextension=ev, hextension=eh,
                                           vlayer=lv, hlayer=lh,
                                           pin_vlayer=plv, pin_hlayer=plh,
                                           viamap=viamap, primary_grid=primary_grid,
                                           xcolor=xcolor, ycolor=ycolor,
                                           vextension0=e0v, hextension0=e0h)
    >>> print(g.pin_vlayer)
    <laygo2.object.grid.CircularMapping object > 
        class: CircularMapping, 
        elements: [['M1', 'pin']]

    .. image:: ../assets/img/object_grid_RoutingGrid_pin_vlayer.png
           :height: 250

    Notes
    -----
    **(Korean)** 수직 pin wire들의 레이어 정보.
    """

    pin_hlayer = None
    """CircularMapping: Layer information of horizontal pine wires.

    Example
    -------
    >>> import laygo2
    >>> from laygo2.object.grid import CircularMapping as CM
    >>> from laygo2.object.grid import CircularMappingArray as CMA
    >>> from laygo2.object.grid import OneDimGrid, RoutingGrid
    >>> from laygo2.object.template import NativeInstanceTemplate
    >>> # Routing grid construction (not needed if laygo2_tech is set up).
    >>> gv = OneDimGrid(name="gv", scope=[0, 50], elements=[0])
    >>> gh = OneDimGrid(name="gv", scope=[0, 100], elements=[0, 40, 60])
    >>> wv = CM([10])           # vertical (xgrid) width
    >>> wh = CM([20, 10, 10])   # horizontal (ygrid) width
    >>> ev = CM([10])           # vertical (xgrid) extension
    >>> eh = CM([10, 10, 10])   # horizontal (ygrid) extension
    >>> e0v = CM([15])          # vert. extension (for zero-length wires)
    >>> e0h = CM([15, 15, 15])  # hori. extension (for zero-length wires)
    >>> lv = CM([['M1', 'drawing']], dtype=object)  # layer information
    >>> lh = CM([['M2', 'drawing']]*3, dtype=object) 
    >>> plv = CM([['M1', 'pin']], dtype=object) # pin layers
    >>> plh = CM([['M2', 'pin']]*3, dtype=object)
    >>> xcolor = CM([None], dtype=object)  # not multi-patterned 
    >>> ycolor = CM([None]*3, dtype=object) 
    >>> primary_grid = 'horizontal'
    >>> tvia = NativeInstanceTemplate(libname='tlib', cellname='via0')  # via 
    >>> viamap = CMA(elements=[[tvia, tvia, tvia]], dtype=object)
    >>> g = laygo2.object.grid.RoutingGrid(name='mygrid', vgrid=gv, hgrid=gh,
                                           vwidth=wv, hwidth=wh,
                                           vextension=ev, hextension=eh,
                                           vlayer=lv, hlayer=lh,
                                           pin_vlayer=plv, pin_hlayer=plh,
                                           viamap=viamap, primary_grid=primary_grid,
                                           xcolor=xcolor, ycolor=ycolor,
                                           vextension0=e0v, hextension0=e0h)
    >>> print(g.pin_hlayer)
    <laygo2.object.grid.CircularMapping object > 
        class: CircularMapping, 
        elements: [['M1', 'pin'], ['M1', 'pin'], ['M1', 'pin']]

    .. image:: ../assets/img/object_grid_RoutingGrid_pin_hlayer.png
           :height: 250

    Notes
    -----
    **(Korean)** 수평 pin wire 들의 레이어정보.
    """

    viamap = None
    """CircularMappingArray: Array containing Via objects positioned on grid crossing points.

    Example
    -------
    >>> import laygo2
    >>> from laygo2.object.grid import CircularMapping as CM
    >>> from laygo2.object.grid import CircularMappingArray as CMA
    >>> from laygo2.object.grid import OneDimGrid, RoutingGrid
    >>> from laygo2.object.template import NativeInstanceTemplate
    >>> # Routing grid construction (not needed if laygo2_tech is set up).
    >>> gv = OneDimGrid(name="gv", scope=[0, 50], elements=[0])
    >>> gh = OneDimGrid(name="gv", scope=[0, 100], elements=[0, 40, 60])
    >>> wv = CM([10])           # vertical (xgrid) width
    >>> wh = CM([20, 10, 10])   # horizontal (ygrid) width
    >>> ev = CM([10])           # vertical (xgrid) extension
    >>> eh = CM([10, 10, 10])   # horizontal (ygrid) extension
    >>> e0v = CM([15])          # vert. extension (for zero-length wires)
    >>> e0h = CM([15, 15, 15])  # hori. extension (for zero-length wires)
    >>> lv = CM([['M1', 'drawing']], dtype=object)  # layer information
    >>> lh = CM([['M2', 'drawing']]*3, dtype=object) 
    >>> plv = CM([['M1', 'pin']], dtype=object) # pin layers
    >>> plh = CM([['M2', 'pin']]*3, dtype=object)
    >>> xcolor = CM([None], dtype=object)  # not multi-patterned 
    >>> ycolor = CM([None]*3, dtype=object) 
    >>> primary_grid = 'horizontal'
    >>> tvia = NativeInstanceTemplate(libname='tlib', cellname='via0')  # via 
    >>> viamap = CMA(elements=[[tvia, tvia, tvia]], dtype=object)
    >>> g = laygo2.object.grid.RoutingGrid(name='mygrid', vgrid=gv, hgrid=gh,
                                           vwidth=wv, hwidth=wh,
                                           vextension=ev, hextension=eh,
                                           vlayer=lv, hlayer=lh,
                                           pin_vlayer=plv, pin_hlayer=plh,
                                           viamap=viamap, primary_grid=primary_grid,
                                           xcolor=xcolor, ycolor=ycolor,
                                           vextension0=e0v, hextension0=e0h)
    >>> print(g.viamap)
    <laygo2.object.grid.CircularMappingArray object at 0x000002217F15A530> 
    class: CircularMappingArray, 
    elements: [
        [<laygo2.object.template.NativeInstanceTemplate object at 0x000002217F15ADD0>
         <laygo2.object.template.NativeInstanceTemplate object at 0x000002217F15ADD0>
         <laygo2.object.template.NativeInstanceTemplate object at 0x000002217F15ADD0>]]

    .. image:: ../assets/img/object_grid_RoutingGrid_viamap.png
           :height: 250

    Notes
    -----
    **(Korean)** 그리드 교차점에 위치하는 via개채들을 담고있는배열.
    """

    primary_grid = "vertical"
    """str: The default direction of routing 
        (Direction of wire having length 0).

    Example
    -------
    >>> import laygo2
    >>> from laygo2.object.grid import CircularMapping as CM
    >>> from laygo2.object.grid import CircularMappingArray as CMA
    >>> from laygo2.object.grid import OneDimGrid, RoutingGrid
    >>> from laygo2.object.template import NativeInstanceTemplate
    >>> # Routing grid construction (not needed if laygo2_tech is set up).
    >>> gv = OneDimGrid(name="gv", scope=[0, 50], elements=[0])
    >>> gh = OneDimGrid(name="gv", scope=[0, 100], elements=[0, 40, 60])
    >>> wv = CM([10])           # vertical (xgrid) width
    >>> wh = CM([20, 10, 10])   # horizontal (ygrid) width
    >>> ev = CM([10])           # vertical (xgrid) extension
    >>> eh = CM([10, 10, 10])   # horizontal (ygrid) extension
    >>> e0v = CM([15])          # vert. extension (for zero-length wires)
    >>> e0h = CM([15, 15, 15])  # hori. extension (for zero-length wires)
    >>> lv = CM([['M1', 'drawing']], dtype=object)  # layer information
    >>> lh = CM([['M2', 'drawing']]*3, dtype=object) 
    >>> plv = CM([['M1', 'pin']], dtype=object) # pin layers
    >>> plh = CM([['M2', 'pin']]*3, dtype=object)
    >>> xcolor = CM([None], dtype=object)  # not multi-patterned 
    >>> ycolor = CM([None]*3, dtype=object) 
    >>> primary_grid = 'horizontal'
    >>> tvia = NativeInstanceTemplate(libname='tlib', cellname='via0')  # via 
    >>> viamap = CMA(elements=[[tvia, tvia, tvia]], dtype=object)
    >>> g = laygo2.object.grid.RoutingGrid(name='mygrid', vgrid=gv, hgrid=gh,
                                           vwidth=wv, hwidth=wh,
                                           vextension=ev, hextension=eh,
                                           vlayer=lv, hlayer=lh,
                                           pin_vlayer=plv, pin_hlayer=plh,
                                           viamap=viamap, primary_grid=primary_grid,
                                           xcolor=xcolor, ycolor=ycolor,
                                           vextension0=e0v, hextension0=e0h)
    >>> print(g.primary_grid) 
    “horizontal”

    .. image:: ../assets/img/object_grid_RoutingGrid_primary_grid.png
           :height: 250

    Notes
    -----
    **(Korean)** Routing의 기본 방향 (길이가 0인 wire방향).
    """

    xcolor = None
    """CircularMapping: Color of horizontal wires.

    Example
    -------
    >>> templates = tech.load_templates() 
    >>> grids = tech.load_grids(templates=templates) 
    >>> r23   = grids['routing_23_cmos’] 
    >>> print(r23.xcolor) 
    <laygo2.object.grid.CircularMapping object> class: CircularMapping, 
        elements: [[“colorA”], [“colorB”], [“colorA”], [“colorB”], [“colorA”], [“colorB”], [“colorA”], [“colorB”]]

    .. image:: ../assets/img/object_grid_RoutingGrid_xcolor.png
           :height: 250

    Notes
    -----
    **(Korean)** 수평 wire 들의 color.
    """

    ycolor = None
    """CircularMapping: Color of vertical wires.

    Example
    -------
    >>> templates = tech.load_templates() 
    >>> grids = tech.load_grids(templates=templates) 
    >>> r23   = grids['routing_23_cmos’]
    >>> print(r23.ycolor) 
    <laygo2.object.grid.CircularMapping object> class: CircularMapping, 
        elements: [[“colorA”]]

    .. image:: ../assets/img/object_grid_RoutingGrid_ycolor.png
           :height: 250

    Notes
    -----
    **(Korean)** 수직 wire들의 color.
    """

    def __init__(
        self,
        name,
        vgrid,
        hgrid,
        vwidth,
        hwidth,
        vextension,
        hextension,
        vlayer,
        hlayer,
        pin_vlayer,
        pin_hlayer,
        viamap,
        xcolor=None,
        ycolor=None,
        primary_grid="vertical",
        vextension0=None,
        hextension0=None,
    ):
        """
        Constructor function of RoutingGrid class.

        Parameters
        ----------
        name : str
            Routing object name
        vgrid : laygo2.OneDimGrid
            OneDimGrid of x-coordinate system
        hgrid : laygo2.OneDimGrid
            OneDimGrid of y-coordinate system
        vwidth : CircularMapping
            x-coordinate system width
        hwidth : CircularMapping
            y-coordinate system width
        vextension : CircularMapping
            x-coordinate system extension
        hextension : CircularMapping
            y-coordinate system extension
        vlayer : CircularMapping
            x-coordinate system layer
        hlayer : CircularMapping
            y-coordinate system layer
        pin_vlayer : CircularMapping
            layer of x-coordinate system pin
        pin_hlayer : CircularMapping
            layer of y-coordinate system pin
        xcolor : list
            x-coordinate system color
        ycolor : list
            y-coordinate system color
        viamap : CircularMappingArray
            Via map of Grid
        primary_grid : str
            direction of wire having length 0

        Returns
        -------
        laygo2.RoutingGrid

        Example
        -------
        >>> import laygo2
        >>> from laygo2.object.grid import CircularMapping as CM
        >>> from laygo2.object.grid import CircularMappingArray as CMA
        >>> from laygo2.object.grid import OneDimGrid, RoutingGrid
        >>> from laygo2.object.template import NativeInstanceTemplate
        >>> from laygo2.object.physical import Instance
        >>> # Routing grid construction (not needed if laygo2_tech is set up).
        >>> gv = OneDimGrid(name="gv", scope=[0, 50], elements=[0])
        >>> gh = OneDimGrid(name="gv", scope=[0, 100], elements=[0, 40, 60])
        >>> wv = CM([10])           # vertical (xgrid) width
        >>> wh = CM([20, 10, 10])   # horizontal (ygrid) width
        >>> ev = CM([10])           # vertical (xgrid) extension
        >>> eh = CM([10, 10, 10])   # horizontal (ygrid) extension
        >>> e0v = CM([15])          # vert. extension (for zero-length wires)
        >>> e0h = CM([15, 15, 15])  # hori. extension (for zero-length wires)
        >>> lv = CM([['M1', 'drawing']], dtype=object)  # layer information
        >>> lh = CM([['M2', 'drawing']]*3, dtype=object)
        >>> plv = CM([['M1', 'pin']], dtype=object) # pin layers
        >>> plh = CM([['M2', 'pin']]*3, dtype=object)
        >>> xcolor = CM([None], dtype=object)  # not multi-patterned
        >>> ycolor = CM([None]*3, dtype=object)
        >>> primary_grid = 'horizontal'
        >>> tvia = NativeInstanceTemplate(libname='tlib', cellname='via0')  # via
        >>> viamap = CMA(elements=[[tvia, tvia, tvia]], dtype=object)
        >>> g = RoutingGrid(name='mygrid', vgrid=gv, hgrid=gh,
                            vwidth=wv, hwidth=wh,
                            vextension=ev, hextension=eh,
                            vlayer=lv, hlayer=lh,
                            pin_vlayer=plv, pin_hlayer=plh,
                            viamap=viamap, primary_grid=primary_grid,
                            xcolor=xcolor, ycolor=ycolor,
                            vextension0=e0v, hextension0=e0h)
        >>> # Routing on grid
        >>> mn_list = [[0, -2], [0, 1], [2, 1], [5,1] ]
        >>> route = g.route(mn=mn_list, via_tag=[True, None, True, True])
        >>> for r in route:
        >>>     print(r)
        <laygo2.object.physical.Instance object at 0x0000016939A23A90>
            name: None,
            class: Instance,
            xy: [0, -60],
            params: None,
            size: [0, 0]
            shape: None
            pitch: [0, 0]
            transform: R0
            pins: {}
        <laygo2.object.physical.Rect object at 0x0000016939A23880>
            name: None,
            class: Rect,
            xy: [[0, -60], [0, 40]],
            params: None, , layer: ['M1' 'drawing'], netname: None
        <laygo2.object.physical.Rect object at 0x0000016939A21BA0>
            name: None,
            class: Rect,
            xy: [[0, 40], [100, 40]],
            params: None, , layer: ['M2' 'drawing'], netname: None
        <laygo2.object.physical.Instance object at 0x0000016939A21B70>
            name: None,
            class: Instance,
            xy: [100, 40],
            params: None,
            size: [0, 0]
            shape: None
            pitch: [0, 0]
            transform: R0
            pins: {}
        <laygo2.object.physical.Rect object at 0x0000016939A21D80>
            name: None,
            class: Rect,
            xy: [[100, 40], [250, 40]],
            params: None, , layer: ['M2' 'drawing'], netname: None
        <laygo2.object.physical.Instance object at 0x0000016939A22350>
            name: None,
            class: Instance,
            xy: [250, 40],
            params: None,
            size: [0, 0]
            shape: None
            pitch: [0, 0]
            transform: R0
            pins: {}

        .. image:: ../assets/img/object_grid_RoutingGrid_init.png
           :height: 250

        Notes
        -----
        **(Korean)**
        RoutingGrid 클래스의 생성자함수.
        파라미터
        name(str): Routing 객체의 이름
        vgrid(laygo2.OneDimGrid): x좌표계 OneDimGrid
        hgrid(laygo2.OneDimGrid): y좌표계 OneDimGrid
        vwidth(CircularMapping): x좌표계 Width
        hwidth(CircularMapping): y좌표계 Width
        vextension(CircularMapping): x좌표계의 extension
        hextension(CircularMapping): y좌표계의 extension
        vlayer(CircularMapping): x좌표계의 layer
        hlayer(CircularMapping): y좌표계의 layer
        pin_vlayer(CircularMapping): x좌표계 pin의 layer
        pin_hlayer(CircularMapping): y좌표계 pin의 layer
        xcolor(list): x좌표계 color
        ycolor(list): y좌표계 color
        viamap(CircularMappingArray): Grid의 Via map
        primary_grid(str): 길이가 0인 Wire방향
        반환값
        laygo2.RoutingGrid
        참조
        없음
        """
        self.vwidth = vwidth
        self.hwidth = hwidth
        self.vextension = vextension
        self.hextension = hextension
        if vextension0 is None:
            self.vextension0 = vextension
        else:
            self.vextension0 = vextension0
        if hextension0 is None:
            self.hextension0 = hextension
        else:
            self.hextension0 = hextension0
        self.vlayer = vlayer
        self.hlayer = hlayer
        self.pin_vlayer = pin_vlayer
        self.pin_hlayer = pin_hlayer
        self.viamap = viamap
        self.primary_grid = primary_grid
        if xcolor is None:
            self.xcolor = CircularMapping([None]*self.vwidth.shape[0], dtype=object)
        else:
            self.xcolor = xcolor
        if ycolor is None:
            self.ycolor = CircularMapping([None]*self.hwidth.shape[0], dtype=object)
        else:
            self.ycolor = ycolor
        Grid.__init__(self, name=name, vgrid=vgrid, hgrid=hgrid)

    def route(self, mn, direction=None, via_tag=None):
        """
        Create wire object(s) for routing.

        Parameters
        ----------
        mn : list(numpy.ndarray)
            The list containing two or more mn coordinates to be connected.
        direction : str, optional.
            None or “vertical” or "horizontal". The direction of the routing object.
        via_tag : list(Boolean), optional.
            The list containing switches deciding whether to place via at the edges.

        Returns
        -------
        laygo2.object.physical.Rect or list :
            The generated routing object(s). Check the example code for details.

        Example
        -------
        >>> import laygo2
        >>> from laygo2.object.grid import CircularMapping as CM
        >>> from laygo2.object.grid import CircularMappingArray as CMA
        >>> from laygo2.object.grid import OneDimGrid, RoutingGrid
        >>> from laygo2.object.template import NativeInstanceTemplate
        >>> from laygo2.object.physical import Instance
        >>> #
        >>> # Routing grid construction (not needed if laygo2_tech is set up).
        >>> #
        >>> gv = OneDimGrid(name="gv", scope=[0, 50], elements=[0])
        >>> gh = OneDimGrid(name="gv", scope=[0, 100], elements=[0, 40, 60])
        >>> wv = CM([10])           # vertical (xgrid) width
        >>> wh = CM([20, 10, 10])   # horizontal (ygrid) width
        >>> ev = CM([10])           # vertical (xgrid) extension
        >>> eh = CM([10, 10, 10])   # horizontal (ygrid) extension
        >>> e0v = CM([15])          # vert. extension (for zero-length wires)
        >>> e0h = CM([15, 15, 15])  # hori. extension (for zero-length wires)
        >>> lv = CM([['M1', 'drawing']], dtype=object)  # layer information
        >>> lh = CM([['M2', 'drawing']]*3, dtype=object)
        >>> plv = CM([['M1', 'pin']], dtype=object) # pin layers
        >>> plh = CM([['M2', 'pin']]*3, dtype=object)
        >>> xcolor = CM([None], dtype=object)  # not multi-patterned
        >>> ycolor = CM([None]*3, dtype=object)
        >>> primary_grid = 'horizontal'
        >>> tvia = NativeInstanceTemplate(libname='tlib', cellname='via0')  # via
        >>> viamap = CMA(elements=[[tvia, tvia, tvia]], dtype=object)
        >>> g = laygo2.object.grid.RoutingGrid(name='mygrid', vgrid=gv, hgrid=gh,
                                               vwidth=wv, hwidth=wh,
                                               vextension=ev, hextension=eh,
                                               vlayer=lv, hlayer=lh,
                                               pin_vlayer=plv, pin_hlayer=plh,
                                               viamap=viamap, primary_grid=primary_grid,
                                               xcolor=xcolor, ycolor=ycolor,
                                               vextension0=e0v, hextension0=e0h)
        >>> #
        >>> # Routing on grid
        >>> #
        >>> mn_list = [[0, -2], [0, 1], [2, 1], [5,1] ]
        >>> route = g.route(mn=mn_list, via_tag=[True, None, True, True])
        >>> for r in route:
        >>>     print(r)
        <laygo2.object.physical.Instance object at 0x0000016939A23A90>
            name: None,
            class: Instance,
            xy: [0, -60],
            params: None,
            size: [0, 0]
            shape: None
            pitch: [0, 0]
            transform: R0
            pins: {}
        <laygo2.object.physical.Rect object at 0x0000016939A23880>
            name: None,
            class: Rect,
            xy: [[0, -60], [0, 40]],
            params: None, , layer: ['M1' 'drawing'], netname: None
        <laygo2.object.physical.Rect object at 0x0000016939A21BA0>
            name: None,
            class: Rect,
            xy: [[0, 40], [100, 40]],
            params: None, , layer: ['M2' 'drawing'], netname: None
        <laygo2.object.physical.Instance object at 0x0000016939A21B70>
            name: None,
            class: Instance,
            xy: [100, 40],
            params: None,
            size: [0, 0]
            shape: None
            pitch: [0, 0]
            transform: R0
            pins: {}
        <laygo2.object.physical.Rect object at 0x0000016939A21D80>
            name: None,
            class: Rect,
            xy: [[100, 40], [250, 40]],
            params: None, , layer: ['M2' 'drawing'], netname: None
        <laygo2.object.physical.Instance object at 0x0000016939A22350>
            name: None,
            class: Instance,
            xy: [250, 40],
            params: None,
            size: [0, 0]
            shape: None
            pitch: [0, 0]
            transform: R0
            pins: {}

        .. image:: ../assets/img/object_grid_RoutingGrid_route.png
           :height: 250

        Notes
        -----
        **(Korean)**
        추상 좌표 위에 라우팅을 수행 하는 함수.
        파라미터
        mn(list(numpy.ndarray)): 배선을 수행할 2개 이상의 mn 좌표를 담고 있는 list.
        direction(str): None or “vertical”; path의 방향을 결정 (수평 or 수직) [optional].
        via_tag(list(Boolean)): Path에 via를 형성 할지를 결정하는 switch들을 담고 있는 list [optional].
        반환값
        list: 생성된 routing object들을 담고 있는 list.
        """
        mn = np.asarray(mn)
        _mn = list()
        for i in range(1, mn.shape[0]):
            # when more than two points are given,
            # create a multi-point wire compose of sub-routing wires
            # connecting the points given by mn in sequence.
            _mn.append([mn[i - 1, :], mn[i, :]])
        route = list()
        # via at the starting point
        if via_tag is not None:
            if via_tag[0]:
                route.append(self.via(mn=_mn[0][0], params=None))
        # routing wires
        for i, __mn in enumerate(_mn):
            xy0 = self.abs2phy[__mn[0]]
            xy1 = self.abs2phy[__mn[1]]
            _xy = np.array([[xy0[0], xy0[1]], [xy1[0], xy1[1]]])
            if np.all(xy0 == xy1):  # if two points are identical, generate a metal stub on the bottom layer.
                if (direction == "vertical") or ((direction is None) and (self.primary_grid == "vertical")):
                    width = self.vwidth[__mn[0][0]]
                    hextension = int(width / 2)
                    vextension = self.vextension0[__mn[0][0]]
                    layer = self.vlayer[__mn[0][0]]
                    if self.xcolor is not None:
                        color = self.xcolor[
                            __mn[0][0] % self.xcolor.shape[0]
                        ]  # xcolor is determined by its grid layer.
                    else:
                        color = None
                else:
                    width = self.hwidth[__mn[0][1]]
                    hextension = self.hextension0[__mn[0][1]]
                    vextension = int(width / 2)
                    layer = self.hlayer[__mn[0][1]]
                    if self.ycolor is not None:
                        color = self.ycolor[
                            __mn[0][1] % self.ycolor.shape[0]
                        ]  # ycolor is determined by its grid layer.
                    else:
                        color = None
            else:
                if (xy0[0] == xy1[0]) or (direction == "vertical"):  # vertical routing
                    width = self.vwidth[__mn[0][0]]
                    hextension = int(width / 2)
                    vextension = self.vextension[__mn[0][0]]
                    layer = self.vlayer[__mn[0][0]]
                    if self.xcolor is not None:
                        color = self.xcolor[
                            __mn[0][0] % self.xcolor.shape[0]
                        ]  # xcolor is determined by its grid layer.
                    else:
                        color = None
                else:  # horizontal routing
                    width = self.hwidth[__mn[0][1]]
                    hextension = self.hextension[__mn[0][1]]
                    vextension = int(width / 2)
                    layer = self.hlayer[__mn[0][1]]
                    if self.ycolor is not None:
                        color = self.ycolor[
                            __mn[0][1] % self.ycolor.shape[0]
                        ]  # ycolor is determined by its grid layer.
                    else:
                        color = None
            p = laygo2.object.physical.Rect(
                xy=_xy,
                layer=layer,
                hextension=hextension,
                vextension=vextension,
                color=color,
            )
            route.append(p)
            # via placement
            if via_tag is None:
                if (i > 0) and (i < mn.shape[0] - 1):
                    route.append(self.via(mn=__mn[0], params=None))
            else:
                if via_tag[i + 1] == True:
                    route.append(self.via(mn=__mn[1], params=None))
        if len(route) == 1:  # not isinstance(mn[0][0], list):
            return route[0]
        else:
            return route

    def via(self, mn=np.array([0, 0]), params=None):
        """
        Create Via object(s) on abstract grid.

        Parameters
        ----------
        mn : list(numpy.ndarray)
            Abstract coordinate(s) that specify location(s) to insert via(s).

        Returns
        -------
        list(physical.PhysicalObject):
            The list containing the generated via objects.

        Example
        -------
        >>> import laygo2
        >>> from laygo2.object.grid import CircularMapping as CM
        >>> from laygo2.object.grid import CircularMappingArray as CMA
        >>> from laygo2.object.grid import OneDimGrid, RoutingGrid
        >>> from laygo2.object.template import NativeInstanceTemplate
        >>> from laygo2.object.physical import Instance
        >>> # Routing grid construction (not needed if laygo2_tech is set up).
        >>> gv = OneDimGrid(name="gv", scope=[0, 50], elements=[0])
        >>> gh = OneDimGrid(name="gv", scope=[0, 100], elements=[0, 40, 60])
        >>> wv = CM([10])           # vertical (xgrid) width
        >>> wh = CM([20, 10, 10])   # horizontal (ygrid) width
        >>> ev = CM([10])           # vertical (xgrid) extension
        >>> eh = CM([10, 10, 10])   # horizontal (ygrid) extension
        >>> e0v = CM([15])          # vert. extension (for zero-length wires)
        >>> e0h = CM([15, 15, 15])  # hori. extension (for zero-length wires)
        >>> lv = CM([['M1', 'drawing']], dtype=object)  # layer information
        >>> lh = CM([['M2', 'drawing']]*3, dtype=object)
        >>> plv = CM([['M1', 'pin']], dtype=object) # pin layers
        >>> plh = CM([['M2', 'pin']]*3, dtype=object)
        >>> xcolor = CM([None], dtype=object)  # not multi-patterned
        >>> ycolor = CM([None]*3, dtype=object)
        >>> primary_grid = 'horizontal'
        >>> tvia = NativeInstanceTemplate(libname='tlib', cellname='via0')  # via
        >>> viamap = CMA(elements=[[tvia, tvia, tvia]], dtype=object)
        >>> g = laygo2.object.grid.RoutingGrid(name='mygrid', vgrid=gv, hgrid=gh,
                                               vwidth=wv, hwidth=wh,
                                               vextension=ev, hextension=eh,
                                               vlayer=lv, hlayer=lh,
                                               pin_vlayer=plv, pin_hlayer=plh,
                                               viamap=viamap, primary_grid=primary_grid,
                                               xcolor=xcolor, ycolor=ycolor,
                                               vextension0=e0v, hextension0=e0h)
        >>> # Routing on grid
        >>> mn_list = [[0, -2], [1, 0], [2, 5]]
        >>> via = mygrid.via(mn=mn_list)
        >>> print(via)
        [<laygo2.object.physical.VirtualInstance object>,
         <laygo2.object.physical.VirtualInstance object>,
         <laygo2.object.physical.VirtualInstance object>]

        .. image:: ../assets/img/object_grid_RoutingGrid_via.png
           :height: 250

        Notes
        -----
        **(Korean)** via 생성함수.

        파라미터
            - mn(list(numpy.ndarray)): via를 생성할 mn좌표. 복수 개 입력 가능.
        반환값
            - list(physical.PhysicalObject)): 생성된 via들을 담고 있는 list.
        """
        # If mn contains multiple coordinates (or objects), place iteratively.
        if isinstance(mn, list):
            if isinstance(mn[0], (int, np.integer)):  # It's actually a single coordinate.
                return self.via(mn=np.asarray(mn), params=params)
            else:
                return [self.via(mn=_mn, params=params) for _mn in mn]
        elif isinstance(mn, np.ndarray):
            if isinstance(mn[0], (int, np.integer)):  # It's actually a single coordinate.
                pass
            else:
                return np.array([self.via(mn=_mn, params=params) for _mn in mn])
        if not isinstance(mn, tuple):
            mn = tuple(mn)  # viamap (CircularMapping) works only with tuples
        tvia = self.viamap[mn]
        via = tvia.generate(params=params)
        via.xy = self[mn]
        return via

    def route_via_track(self, mn, track, via_tag=[None, True]):
        """
        Perform routing on the specified track with accessing wires to mn.

        Parameters
        ----------
        mn : list(numpy.ndarray)
            list containing coordinates of the points being connected through a track
        track : numpy.ndarray
            list containing coordinate values and direction of a track.
            Vertical tracks have [v, None] format, while horizontal tracks have [None, v] format
            (v is the coordinates of the track).

        Returns
        -------
        list:
            The list containing the generated routing objects;
            The last object corresponds to the routing object on the track.

        Example
        -------
        >>> import laygo2
        >>> from laygo2.object.grid import CircularMapping as CM
        >>> from laygo2.object.grid import CircularMappingArray as CMA
        >>> from laygo2.object.grid import OneDimGrid, RoutingGrid
        >>> from laygo2.object.template import NativeInstanceTemplate
        >>> from laygo2.object.physical import Instance
        >>> # Routing grid construction (not needed if laygo2_tech is set up).
        >>> gv = OneDimGrid(name="gv", scope=[0, 50], elements=[0])
        >>> gh = OneDimGrid(name="gv", scope=[0, 100], elements=[0, 40, 60])
        >>> wv = CM([10])           # vertical (xgrid) width
        >>> wh = CM([20, 10, 10])   # horizontal (ygrid) width
        >>> ev = CM([10])           # vertical (xgrid) extension
        >>> eh = CM([10, 10, 10])   # horizontal (ygrid) extension
        >>> e0v = CM([15])          # vert. extension (for zero-length wires)
        >>> e0h = CM([15, 15, 15])  # hori. extension (for zero-length wires)
        >>> lv = CM([['M1', 'drawing']], dtype=object)  # layer information
        >>> lh = CM([['M2', 'drawing']]*3, dtype=object)
        >>> plv = CM([['M1', 'pin']], dtype=object) # pin layers
        >>> plh = CM([['M2', 'pin']]*3, dtype=object)
        >>> xcolor = CM([None], dtype=object)  # not multi-patterned
        >>> ycolor = CM([None]*3, dtype=object)
        >>> primary_grid = 'horizontal'
        >>> tvia = NativeInstanceTemplate(libname='tlib', cellname='via0')  # via
        >>> viamap = CMA(elements=[[tvia, tvia, tvia]], dtype=object)
        >>> g = laygo2.object.grid.RoutingGrid(name='mygrid', vgrid=gv, hgrid=gh,
                                               vwidth=wv, hwidth=wh,
                                               vextension=ev, hextension=eh,
                                               vlayer=lv, hlayer=lh,
                                               pin_vlayer=plv, pin_hlayer=plh,
                                               viamap=viamap, primary_grid=primary_grid,
                                               xcolor=xcolor, ycolor=ycolor,
                                               vextension0=e0v, hextension0=e0h)
        >>> # Routing on grid
        >>> mn_list = [[0, -2], [1, 0], [2, 5], [3, 4], [4, 5], [5, 5]]
        >>> track = g.route_via_track(mn=mn_list, track=[None, 0])
        >>> print(track)
        [[<laygo2.object.physical.Rect object>,
          <laygo2.object.physical.VirtualInstance object>],
          <laygo2.object.physical.VirtualInstance object>,
         [<laygo2.object.physical.Rect object>,
          <laygo2.object.physical.VirtualInstance object>],
         [<laygo2.object.physical.Rect object>,
          <laygo2.object.physical.VirtualInstance object>],
         [<laygo2.object.physical.Rect object>,
          <laygo2.object.physical.VirtualInstance object>],
         [<laygo2.object.physical.Rect object>,
          <laygo2.object.physical.VirtualInstance object>],
          <laygo2.object.physical.Rect object>]

        .. image:: ../assets/img/object_grid_RoutingGrid_route_via_track.png
           :height: 250

        Notes
        -----
        **(Korean)** wire 라우팅 함수, track을 기준점으로 routing을 진행한다.

        파라미터
            - track(numpy.ndarray): track의 좌표값과 방향을 담고 있는 list.
                수직 트랙일 경우 [v, None],
                수평 트랙일 경우 [None, v]의 형태를 가지고 있다 (v는 track의 좌표값).
            - mn(list(numpy.ndarray)): track을 통해 연결될 지점들의 좌표를 담고 있는 list.
        반환값
            - list: 생성된 routing object들을 담고 있는 list.
                마지막 object가 track위의 routing object에 해당.
        """
        mn = np.array(mn)
        route = list()

        if track[1] != None:  # x direction
            t = 0  # index of track axis
            p = 1  # index of perpendicular track
            mn_pivot = track[1]
        else:  # y direction
            t = 1
            p = 0
            mn_pivot = track[0]

        mn_b = np.array([[0, 0], [0, 0]])  # 1.branch
        min_t, max_t = mn[0][t], mn[0][t]

        for i in range(len(mn)):
            mn_b[0] = mn[i]
            mn_b[1][t] = mn_b[0][t]
            mn_b[1][p] = mn_pivot
            if np.array_equal(mn_b[0], mn_b[1]):  #### via only
                route.append(self.via(mn=mn_b[0], params=None))
            else:
                route.append(self.route(mn=[mn_b[0], mn_b[1]], via_tag=via_tag))

            center_t = mn[i][t]
            if center_t < min_t:
                min_t = center_t
            elif max_t < center_t:
                max_t = center_t

        mn_track = np.array([[0, 0], [0, 0]])  # 2.track
        mn_track[0][t], mn_track[0][p] = min_t, mn_pivot  # min
        mn_track[1][t], mn_track[1][p] = max_t, mn_pivot  # max

        if np.array_equal(mn_track[0], mn_track[1]):  # Skip
            route.append(None)
        else:
            route.append(self.route(mn=mn_track))

        return route

    def pin(self, name, mn, direction=None, netname=None, params=None):
        """
        Create a Pin object over the abstract coordinates specified by mn,
        on the specified routing grid.

        Parameters
        ----------
        name : str
            Pin name.
        mn : numpy.ndarray
            Abstract coordinates for generating Pin.
        direction : str, optional.
            Direction.
        netname : str, optional.
            Net name of Pin.
        params : dict, optional
            Pin attributes.

        Returns
        -------
        laygo2.physical.Pin: The generated pin object.

        Example
        -------
        >>> import laygo2
        >>> from laygo2.object.grid import CircularMapping as CM
        >>> from laygo2.object.grid import CircularMappingArray as CMA
        >>> from laygo2.object.grid import OneDimGrid, RoutingGrid
        >>> from laygo2.object.template import NativeInstanceTemplate
        >>> # Routing grid construction (not needed if laygo2_tech is set up).
        >>> gv = OneDimGrid(name="gv", scope=[0, 50], elements=[0])
        >>> gh = OneDimGrid(name="gv", scope=[0, 100], elements=[0, 40, 60])
        >>> wv = CM([10])           # vertical (xgrid) width
        >>> wh = CM([20, 10, 10])   # horizontal (ygrid) width
        >>> ev = CM([10])           # vertical (xgrid) extension
        >>> eh = CM([10, 10, 10])   # horizontal (ygrid) extension
        >>> e0v = CM([15])          # vert. extension (for zero-length wires)
        >>> e0h = CM([15, 15, 15])  # hori. extension (for zero-length wires)
        >>> lv = CM([['M1', 'drawing']], dtype=object)  # layer information
        >>> lh = CM([['M2', 'drawing']]*3, dtype=object)
        >>> plv = CM([['M1', 'pin']], dtype=object) # pin layers
        >>> plh = CM([['M2', 'pin']]*3, dtype=object)
        >>> xcolor = CM([None], dtype=object)  # not multi-patterned
        >>> ycolor = CM([None]*3, dtype=object)
        >>> primary_grid = 'horizontal'
        >>> tvia = NativeInstanceTemplate(libname='tlib', cellname='via0')  # via
        >>> viamap = CMA(elements=[[tvia, tvia, tvia]], dtype=object)
        >>> g = laygo2.object.grid.RoutingGrid(name='mygrid', vgrid=gv, hgrid=gh,
                                               vwidth=wv, hwidth=wh,
                                               vextension=ev, hextension=eh,
                                               vlayer=lv, hlayer=lh,
                                               pin_vlayer=plv, pin_hlayer=plh,
                                               viamap=viamap, primary_grid=primary_grid,
                                               xcolor=xcolor, ycolor=ycolor,
                                               vextension0=e0v, hextension0=e0h)
        >>> mn = [[0, 0], [10, 10]]
        >>> pin = g.pin(name="pin", grid=g, mn=mn)
        >>> print(pin)
        <laygo2.object.physical.Pin object at 0x0000028DABE3AB90>
            name: pin,
            class: Pin,
            xy: [[0, -10], [500, 350]],
            params: None, , layer: ['M2' 'pin'], netname: pin, shape: None,
            master: None

        Notes
        -----
        **(Korean)** pin 생성함수.

        파라미터
            - name(str): Pin 이름
            - mn(numpy.ndarray): Pin을 생성할 abstract 좌표
            - direction(str): 방향 [optional]
            - netname(str): Pin의 net이름 [optional]
            - params(dict): Pin 속성 [optional]
        반환값
            - laygo2.physical.Pin: Pin object
        """
        xy0 = self.abs2phy[mn[0]]
        xy1 = self.abs2phy[mn[1]]
        # _xy = np.array([[xy0[0], xy0[1]], [xy1[0], xy1[1]]])
        if np.all(xy0 == xy1):  # if two points are identical, generate a metal stub on the bottom layer.
            if (direction == "vertical") or ((direction is None) and (self.primary_grid == "vertical")):
                width = self.vwidth[mn[0][0]]
                hextension = int(width / 2)
                vextension = 0
                layer = self.pin_vlayer[mn[0][0]]
            else:
                width = self.hwidth[mn[0][1]]
                hextension = 0
                vextension = int(width / 2)
                layer = self.pin_hlayer[mn[0][1]]
        else:
            if (xy0[0] == xy1[0]) or (direction == "vertical"):  # vertical routing
                width = self.vwidth[mn[0][0]]
                hextension = int(width / 2)
                vextension = 0
                layer = self.pin_vlayer[mn[0][0]]
            else:  # horizontal routing
                width = self.hwidth[mn[0][1]]
                hextension = 0
                vextension = int(width / 2)
                layer = self.pin_hlayer[mn[0][1]]
        # TODO: pin.xy differ from tech.py.
        _xy = np.array(
            [
                [xy0[0] - hextension, xy0[1] - vextension],
                [xy1[0] + hextension, xy1[1] + vextension],
            ]
        )  ## need to check
        p = laygo2.object.physical.Pin(name=name, xy=_xy, layer=layer, netname=netname, params=params)
        return p

    def copy(self): 
        """Copy the current RoutingGrid object.
        """
        name = self.name
        vgrid = self.vgrid.copy()
        hgrid = self.hgrid.copy()
        vwidth = self.vwidth.copy()
        hwidth = self.hwidth.copy()
        vextension = self.vextension.copy()
        hextension = self.hextension.copy()
        vlayer = self.vlayer.copy()
        hlayer = self.hlayer.copy()
        pin_vlayer = self.pin_vlayer.copy()
        pin_hlayer = self.pin_hlayer.copy()
        viamap = self.viamap.copy()
        xcolor = self.xcolor.copy()
        ycolor = self.ycolor.copy()
        primary_grid = self.primary_grid
        vextension0 = self.vextension0.copy()
        hextension0 = self.hextension0.copy()

        rg = RoutingGrid(
            name = name,
            vgrid = vgrid,
            hgrid = hgrid,
            vwidth = vwidth,
            hwidth = hwidth,
            vextension = vextension,
            hextension = hextension,
            vlayer = vlayer,
            hlayer = hlayer,
            pin_vlayer = pin_vlayer,
            pin_hlayer = pin_hlayer,
            viamap = viamap,
            xcolor = xcolor,
            ycolor = ycolor,
            primary_grid = primary_grid,
            vextension0 = vextension0,
            hextension0 = hextension0,
        )
        return rg

    def vflip(self, copy=True):
        """Flip the routing grid in vertical direction."""
        if copy:
            g = self.copy()
        else:
            g = self
        g.hgrid.flip()
        g.hwidth.flip()
        g.hextension.flip()
        g.hlayer.flip()
        g.pin_hlayer.flip()
        g.ycolor.flip()
        g.viamap.flip(axis=1)
        g.hextension0.flip()
        return g

    def hflip(self, copy=True):
        """Flip the routing grid in horizontal direction."""
        if copy:
            g = self.copy()
        else:
            g = self
        g.vgrid.flip()
        g.vwidth.flip()
        g.vextension.flip()
        g.vlayer.flip()
        g.pin_vlayer.flip()
        g.xcolor.flip()
        g.viamap.flip(axis=0)
        g.vextension0.flip()
        return g

    def vstack(self, obj, copy=True):
        """Stack routing grid(s) on top of the routing grid in vertical direction."""
        if copy:
            g = self.copy()
        else:
            g = self
        if isinstance(obj, list):  # multiple stack
            obj_list = obj
        else:  # single stack
            obj_list = [obj]
        # compute the grid range first
        grid_ofst = g.hgrid.width
        for _obj in obj_list:
            g.hgrid.range[1] += _obj.hgrid.width
        # stack
        for _obj in obj_list:
            for i, h in enumerate(_obj.hgrid):
                # Check if the new grid element exist in the current grid already.
                val = (h - _obj.hgrid.range[0]) + grid_ofst
                val = val % (g.hgrid.width)  # modulo
                if not (val in g.hgrid):
                    # Unique element
                    g.hgrid.append(val + g.hgrid.range[0])
                    #g.hgrid.append(h - _obj.hgrid.range[0] + g.hgrid.range[0] + grid_ofst)
                    g.hwidth.append(_obj.hwidth[i])
                    g.hextension.append(_obj.hextension[i])
                    g.hlayer.append(_obj.hlayer[i])
                    g.pin_hlayer.append(_obj.pin_hlayer[i])
                    g.ycolor.append(_obj.ycolor[i])
                    g.hextension0.append(_obj.hextension0[i])
                    elem = np.expand_dims(_obj.viamap.elements[:, i], axis=0)
                    # hstack due to the transposition of numpy array and cartesian system.
                    g.viamap.elements = np.hstack((g.viamap.elements, elem)) 
            grid_ofst += _obj.hgrid.width  # increse offset
        # Do not use the following code, 
        # as it does not work when stacking multiple grids with elements at boundaries. 
        '''
        if isinstance(obj, list):  # Multiple stack.
            for o in obj:
                g = g.vstack(o, copy=copy)
            return g
        for i, h in enumerate(obj.hgrid):
            # Check if the new grid element exist in the current grid already.
            val = (h - obj.hgrid.range[0]) + g.hgrid.width  
            val = val % (g.hgrid.width + obj.hgrid.width)  # modulo
            if not (val in g.hgrid):
                # Unique element
                g.hgrid.append(h + g.hgrid.range[1])
                g.hwidth.append(obj.hwidth[i])
                g.hextension.append(obj.hextension[i])
                g.hlayer.append(obj.hlayer[i])
                g.pin_hlayer.append(obj.pin_hlayer[i])
                g.ycolor.append(obj.ycolor[i])
                g.hextension0.append(obj.hextension0[i])
                elem = np.expand_dims(obj.viamap.elements[:, i], axis=0)
                # hstack due to the transposition of numpy array and cartesian system.
                g.viamap.elements = np.hstack((g.viamap.elements, elem)) 
        g.hgrid.range[1] += obj.hgrid.width
        '''
        return g

    def hstack(self, obj, copy=True):
        """Stack routing grid(s) on top of the routing grid in horizontal direction."""
        if copy:
            g = self.copy()
        else:
            g = self
        if isinstance(obj, list):  # Multiple stack.
            for o in obj:
                g = g.hstack(o, copy=copy)
            return g
        for i, v in enumerate(obj.vgrid):
            # Check if the new grid element exist in the current grid already.
            val = (v - obj.vgrid.range[0]) + g.vgrid.width  
            val = val % (g.vgrid.width + obj.vgrid.width)  # modulo
            if not (val in g.vgrid):
                # Unique element
                g.vgrid.append(v + g.vgrid.range[1])
                g.vwidth.append(obj.vwidth[i])
                g.vextension.append(obj.vextension[i])
                g.vlayer.append(obj.vlayer[i])
                g.pin_vlayer.append(obj.pin_vlayer[i])
                g.xcolor.append(obj.xcolor[i])
                g.vextension0.append(obj.vextension0[i])
                elem = np.expand_dims(obj.viamap.elements[i, :], axis=0)
                # vstack due to the transposition of numpy array and cartesian system.
                g.viamap.elements = np.vstack((g.viamap.elements, elem)) 
        g.vgrid.range[1] += obj.vgrid.width
        return g


    def summarize(self):
        """Summarize object information."""
        return (
            Grid.summarize(self) 
            + " vwidth: " + str(self.vwidth) + "\n"
            + " hwidth: " + str(self.hwidth) + "\n"
            + " vextension: " + str(self.vextension) + "\n"
            + " hextension: " + str(self.hextension) + "\n"
            + " vextension0: " + str(self.vextension0) + "\n"
            + " hextension0: " + str(self.hextension0) + "\n"
            + " vlayer: " + str(self.vlayer) + "\n"
            + " hlayer: " + str(self.hlayer) + "\n"
            + " primary_grid: " + str(self.primary_grid) + "\n"
            + " xcolor: " + str(self.xcolor) + "\n"
            + " ycolor: " + str(self.ycolor) + "\n"
            + " viamap: " + str(self.viamap) + "\n"
        )
    
