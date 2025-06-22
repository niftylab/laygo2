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

#import numpy as np
#from .core import CircularMapping, Grid, OneDimGrid
#import laygo2.object
import numpy as np
from .core import CircularMapping, Grid, OneDimGrid
from ..physical import PhysicalObject, Rect, Path, Pin, Text, Instance, VirtualInstance, PhysicalObjectPointer, PhysicalObjectPointerAbsCoordinate


class RoutingGrid(Grid):
    """
    A class that implements wire connections in an abstract coordinate system.
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
        >>> route = g.route(mn=mn_list, via_tag=[True, False, True, True])
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

    def route(self, mn, direction=None, via_tag=None, netname=None):
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
        >>> route = g.route(mn=mn_list, via_tag=[True, False, True, True])
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

        """
        
        # 1. Check whether the "mn" is physical object list or coordinate list.
        __mn = []
        for _mn in mn:
            if (_mn.__class__.__name__ == "PhysicalObject") or (
                issubclass(_mn.__class__, PhysicalObject)
            ):
                __mn.append(np.array(self.center(_mn))) # Assume the point to be routed as the center of the object.
            elif (_mn.__class__.__name__ == "PhysicalObjectPointer"):
                __mn.append(np.array(_mn.evaluate(self)))
            else:
                __mn.append(np.array(_mn))
        # Construct via_tag if the first/last elements of mn is either PhysicalObject or PhysicalObjectPointer
        # and the via_tag is not provided explictly
        if via_tag is None:  # via_tag is not given
            via_tag_candidate = [False, False]
            # first element of the new via_tag
            if (mn[0].__class__.__name__ == "PhysicalObject") or \
               (issubclass(mn[0].__class__, PhysicalObject)) or \
               (mn[0].__class__.__name__ == "PhysicalObjectPointer"):
                    if mn[0].__class__.__name__ == "PhysicalObjectPointer":
                        _mn_obj = mn[0].master
                    else:
                        _mn_obj = mn[0]
                    if (_mn_obj.__class__.__name__ == "Rect") or \
                       (_mn_obj.__class__.__name__ == "Pin"):
                        if __mn[0][1] == __mn[1][1]: # horizontal
                            if self.hlayer[__mn[0][1]][0] != _mn_obj.layer[0]:
                                via_tag_candidate[0] = True
                        else:  # vertical
                            if self.vlayer[__mn[0][0]][0] != _mn_obj.layer[0]:
                                via_tag_candidate[0] = True
            # second element of the new via_tag
            if (mn[-1].__class__.__name__ == "PhysicalObject") or \
               (issubclass(mn[-1].__class__, PhysicalObject)) or \
               (mn[-1].__class__.__name__ == "PhysicalObjectPointer"):
                    if mn[-1].__class__.__name__ == "PhysicalObjectPointer":
                        _mn_obj = mn[-1].master
                    else:
                        _mn_obj = mn[-1]
                    if (_mn_obj.__class__.__name__ == "Rect") or \
                       (_mn_obj.__class__.__name__ == "Pin"):
                        if __mn[-1][1] == __mn[-2][1]:  # horizontal
                            if self.hlayer[__mn[-1][1]][0] != _mn_obj.layer[0]:
                                via_tag_candidate[1] = True
                        else:  # vertical
                            if self.vlayer[__mn[-1][0]][0] != _mn_obj.layer[0]:
                                via_tag_candidate[1] = True
                    if True in via_tag_candidate:
                        via_tag = via_tag_candidate
        
        mn_orig = mn            
        mn = list()
        # for i in range(1, __mn.shape[0]):
        for i in range(1, len(__mn)):
            # when more than two points are given,
            # create a multi-point wire compose of sub-routing wires
            # connecting the points given by mn in sequence.
            # mn.append([__mn[i - 1, :], __mn[i, :]])
            mn.append([__mn[i-1], __mn[i]])
        route = list()
        # via at the starting point
        if via_tag is not None:
            if via_tag[0]:
                route.append(self.via(mn=mn[0][0], params=None))
        # routing wires
        for i, _mn in enumerate(mn):
            if (_mn[0].ndim > 1) or (_mn[1].ndim > 1):  # if the dimension of the point is more than 1, raise error.
                raise ValueError(f"Invalid input for RoutingGrid.route(): mn should be a list of abstract coordinates or PhysicalObject. Check the value of mn: {mn_orig}") 
            xy0 = self.abs2phy[_mn[0]]
            xy1 = self.abs2phy[_mn[1]]
            _xy = np.array([[xy0[0], xy0[1]], [xy1[0], xy1[1]]])
            if np.all(xy0 == xy1):  # if two points are identical, generate a metal stub on the bottom layer.
                if (direction == "vertical") or ((direction is None) and (self.primary_grid == "vertical")):
                    width = self.vwidth[_mn[0][0]]
                    hextension = int(width / 2)
                    vextension = self.vextension0[_mn[0][0]]
                    layer = self.vlayer[_mn[0][0]]
                    if self.xcolor is not None:
                        color = self.xcolor[
                            _mn[0][0] % self.xcolor.shape[0]
                        ]  # xcolor is determined by its grid layer.
                    else:
                        color = None
                else:
                    width = self.hwidth[_mn[0][1]]
                    hextension = self.hextension0[_mn[0][1]]
                    vextension = int(width / 2)
                    layer = self.hlayer[_mn[0][1]]
                    if self.ycolor is not None:
                        color = self.ycolor[
                            _mn[0][1] % self.ycolor.shape[0]
                        ]  # ycolor is determined by its grid layer.
                    else:
                        color = None
            else:
                if (xy0[0] == xy1[0]) or (direction == "vertical"):  # vertical routing
                    width = self.vwidth[_mn[0][0]]
                    hextension = int(width / 2)
                    vextension = self.vextension[_mn[0][0]]
                    layer = self.vlayer[_mn[0][0]]
                    if self.xcolor is not None:
                        color = self.xcolor[
                            _mn[0][0] % self.xcolor.shape[0]
                        ]  # xcolor is determined by its grid layer.
                    else:
                        color = None
                else:  # horizontal routing
                    width = self.hwidth[_mn[0][1]]
                    hextension = self.hextension[_mn[0][1]]
                    vextension = int(width / 2)
                    layer = self.hlayer[_mn[0][1]]
                    if self.ycolor is not None:
                        color = self.ycolor[
                            _mn[0][1] % self.ycolor.shape[0]
                        ]  # ycolor is determined by its grid layer.
                    else:
                        color = None
            p = Rect(
                xy=_xy,
                layer=layer,
                hextension=hextension,
                vextension=vextension,
                color=color,
                netname=netname,
            )
            route.append(p)
            # via placement 
            if i < (np.asarray(mn).shape[0] - 1):  # connecting vias
                route.append(self.via(mn=_mn[1], params=None))  
            elif via_tag is not None:
                if via_tag[-1]:  # ending point
                    route.append(self.via(mn=_mn[1], params=None))
        if len(route) == 1:  # not isinstance(mn[0][0], list):
            return route[0]
        else:
            return route
        '''
        # 1. Check whether the "mn" is physical object list or coordinate list.
        __mn = []
        for _mn in mn:
            if (_mn.__class__.__name__ == "PhysicalObject") or (
                issubclass(_mn.__class__, laygo2.object.physical.PhysicalObject)
            ):
                __mn.append(np.array(self.center(_mn))) # Assume the point to be routed as the center of the object.
            else:
                __mn.append(np.array(_mn))
        

        mn = np.asarray(__mn)
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
                netname=netname,
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
        '''

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

    def route_via_track(self, mn, track, via_tag=[False, True], netname=None):
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

        """
        branch_offset=None  # compatibility with laygo3
        via_tag_branch=[False, False]  # compatibility with laygo3
        
        # Check if the track number is integer
        if (track[0] is None) and (not isinstance(track[1], int) and not isinstance(track[1], PhysicalObjectPointerAbsCoordinate)):
            if track[1].ndim > 0:
                raise ValueError(f"The format of horizontal track for route_via_track() and route() should be [None, int] or [None, PhysicalObjectPointerAbsCoordinate].")
        if (track[1] is None) and (not isinstance(track[0], int) and not isinstance(track[0], PhysicalObjectPointerAbsCoordinate)):
            if track[0].ndim > 0:
                raise ValueError(f"The format of vertical track for route_via_track() and route() should be [int, None] or [PhysicalObjectPointerAbsCoordinate, None].")

        # Preprocess: complete via_tag and via_tag_branch by adding false to empty entries.
        if via_tag is None:
            via_tag = []
        for diff in range(len(mn)-len(via_tag)):  
            via_tag.append(False)
        if branch_offset is not None: 
            for diff in range(len(branch_offset)-len(via_tag_branch)):
                via_tag_branch.append(False)

        # Preprocess: evaluate the expression of track_num (evaluated version of track)
        track_num = [0, 0]
        if isinstance(track[0], PhysicalObjectPointerAbsCoordinate):
            track_num[0] = track[0].evaluate(self)
        else:
            track_num[0] = track[0]
        if isinstance(track[1], PhysicalObjectPointerAbsCoordinate):
            track_num[1] = track[1].evaluate(self)
        else:
            track_num[1] = track[1]

        # 1. Check whether the "mn" is physical object list or coordinate list and determine the direction of the object.
        route_obj = [] # [mn, layer, direction]
        for _mn in mn:
            if (_mn.__class__.__name__ == "PhysicalObjectPointer") or \
               (_mn.__class__.__name__ == "PhysicalObject") or \
               (issubclass(_mn.__class__, PhysicalObject)):
                if (_mn.__class__.__name__ == "PhysicalObjectPointer"): # if _mn is PhysicalObjectPointer
                    _mn_obj = _mn.master
                    _mn_eval = _mn.evaluate(self)
                else: # If _mn is laygo2.object.Rect or laygo2.object.Pin             
                    _mn_obj = _mn
                    _mn_eval = self.center(_mn)

                if (_mn_obj.width > _mn_obj.height) and (_mn_obj.height > 0):
                    route_obj.append([_mn_eval, _mn_obj.layer[0], "horizontal"]) # Assume the point to be routed as the center of the object.
                elif _mn_obj.width < _mn_obj.height and (_mn_obj.width > 0):
                    route_obj.append([_mn_eval, _mn_obj.layer[0], "vertical"])
                else: 
                    # The policy for determining the direction of square/point objects was changed on Mar-29-2025. 
                    # for square objects, if the object's layer matches the track's layer, 
                    # assume the parallel direction with the track.
                    # otherwise, assume the orthogonal direction with the track.
                    if track_num[1] != None: # horizontal
                        if self.hlayer[track_num[1]][0] == _mn_obj.layer[0]:
                            _direction = "horizontal" 
                        else:
                            _direction = "vertical" 
                    else: # vertical
                        if self.vlayer[track_num[0]][0] == _mn_obj.layer[0]:
                            _direction = "vertical" 
                        else:
                            _direction = "horizontal" 
                    route_obj.append([_mn_eval, _mn_obj.layer[0], _direction])
                    '''
                    # Deprecated on Mar-29-2025.
                    # for square objects, assume the orthogonal direction of the track.
                    if track[1] != None:
                        route_obj.append([self.center(_mn), _mn.layer[0], "vertical"])
                    else:
                        route_obj.append([self.center(_mn), _mn.layer[0], "horizontal"])
                    '''
            else:
                route_obj.append([np.array(_mn), "unknown", "unknown"]) # Use the input "via_tag" variable for routing.
        
        route = []

        # 1. Define the track axis.
        if track_num[1] != None:  # Give a priority to horizontal(y) track.
            tr = 1  # index of track axis
            br = 0  # index of branch axis
            mn_pivot = track_num[1]
            track_direction = "horizontal"
        else: # Vertical(x) track
            tr = 0
            br = 1
            mn_pivot = track_num[0]
            track_direction = "vertical"

        # 2. Route bypass branches (optional)
        if branch_offset != None:
            for idx, offset in enumerate(branch_offset):
                if offset != 0:
                    _mn = route_obj[idx][0]
                    mn_new_x = _mn[0] + tr*offset
                    mn_new_y = _mn[1] + br*offset
                    mn_new = np.array([mn_new_x, mn_new_y])

                    branch_direction = route_obj[idx][2]
                    if branch_direction == "unknown":
                        vtag = [via_tag_branch[idx], True]
                    elif branch_direction == track_direction:
                        vtag = [False, True]
                    else:
                        vtag = [True, True]
                    # Destination point requires via.
                    if not np.array_equal(mn_new, _mn):
                        route.append(self.route(mn=[_mn, mn_new], via_tag=vtag))  # route mn_new, route_obj[idx][0]
                    else:
                        route.append([None, self.via(mn=_mn[0], params=None)])
                    route_obj[idx][0] = mn_new  # update route_obj[idx][0] to mn_new

        # 3. Route main branches.
        mn_b = np.array([[0, 0], [0, 0]]) 
        min_t, max_t = route_obj[0][0][br], route_obj[0][0][br] # Initialize the min, max value for the trunk routing.

        for i in range(len(route_obj)): # Iterate the given mn-list.
            mn_b[0] = route_obj[i][0] # Define the departing point
            mn_b[1][br] = mn_b[0][br] # Define the destination of the branch. 
            mn_b[1][tr] = mn_pivot

            # Via drop - TODO: the stretagy needs to be changed to a layer-based one.
            if mn_b[0][tr] == mn_pivot: # If the branch has zero length,
                if route_obj[i][2] == "unknown": # If the route_obj direction is unknown, create a via at the destination point according to the via_tag.
                    if via_tag[i]:
                        route.append([None, self.via(mn=mn_b[0], params=None)]) # Only create a via at the destination point.
                    else:
                        route.append([None, None])
                elif route_obj[i][2] == track_direction: # If the branch is orthogonal (route_obj is parallel) to the trunk, don't create a via.
                    route.append([None, None])
                else: # If the branch is parallel (route_obj is orthogonal) to the trunk, create a via at the destination point.
                    route.append([None, self.via(mn=mn_b[0], params=None)]) 
            else:
                if route_obj[i][2] == "unknown":
                    vtag = [via_tag[i], True] # If the branch direction is unknown, preserve the via_tag parameter for receiving branch coordinates for mn.
                elif route_obj[i][2] == track_direction: # If the branch is orthogonal (route_obj is parallel) to the trunk, create a via.
                    vtag = [True, True]
                else: # If the branch is parallel (route_obj is orthogonal) to the trunk, create a via at the destination point.
                    vtag = [False, True]
                route.append(self.route(mn=[mn_b[0], mn_b[1]], via_tag=vtag, netname=netname))

            comp = route_obj[i][0][br] # Update the min, max value.
            if comp < min_t:
                min_t = comp
            elif max_t < comp:
                max_t = comp

        # 4. Route a trunk wire on the track.
        mn_trunk = np.array([[0, 0], [0, 0]])
        mn_trunk[0][br], mn_trunk[0][tr] = min_t, mn_pivot # min
        mn_trunk[1][br], mn_trunk[1][tr] = max_t, mn_pivot # max

        # if np.array_equal(mn_trunk[0], mn_trunk[1]): # Skip or create wire with length is 2 ??
        zero_len = int(np.array_equal(mn_trunk[0], mn_trunk[1]))
        mn_trunk[0][0] -= 1*tr*zero_len; mn_trunk[1][0] += 1*tr*zero_len
        mn_trunk[0][1] -= 1*br*zero_len; mn_trunk[1][1] += 1*br*zero_len
        route.append(self.route(mn=mn_trunk, netname=netname, via_tag=[False, False])) # Just create trunk wire on the track, not creating via.

        return route

        
        '''
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
                route.append(self.route(mn=[mn_b[0], mn_b[1]], via_tag=via_tag, netname=netname))

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
            route.append(self.route(mn=mn_track, netname=netname))

        return route
        '''

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

        """
        if (mn.__class__.__name__ == "PhysicalObject") or \
            (issubclass(mn.__class__, laygo2.object.physical.PhysicalObject)): # object is given for the input coordinate
            mn = self.mn.bbox(mn)  # get the bounding box of the object.
        else:  # numerical coordinate
            mn = mn

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
    
