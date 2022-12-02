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
**laygo2.object.routing** module is defining functions and classes for basic routing.
"""

import numpy as np
from laygo2.object.template import *
from laygo2.object.physical import *

class RoutingMeshTemplate(Template):
    """RoutingMesh describes a two dimensional routing structure over a routing grid."""

    grid = None
    """laygo2.object.grid.RoutingGrid: A routing grid object that the RoutingMesh object refers to.
    """

    tracks = dict()
    """dict: A dictionary that contains track information.
    The format of its element is "track_name": [index, "net_name"].
    """

    nodes = []
    """list: A list that contains layout objects to be connected through the routing channel.
    The element could be either one of the following:
        laygo2.object.physical.Instance, 
        laygo2.object.physical.VirtualInstance, 
        laygo2.object.physical.Rect.
    
    For Instance or VirtualInstance elements, the netname parameters of their pins need to be set to the 
    name of net (netname) which the pins are connected to.
    For Rect elements, their netname parameters should be set.       
    """

    # hrange = [None, None]
    # """[int or None, int or None]: The minimum and maximum track indices covered by the
    # horizontal routing channel, for automatic routing.
    # """
    #
    # vrange = [None, None]
    # """[int or None, int or None]: The minimum and maximum track indices covered by the
    # vertical routing channel, for automatic routing.
    # """

    def bbox(self, params=None):
        """numpy.ndarray: (Abstract method) Return the bounding box of a template."""
        # TODO: implement this
        pass

    def pins(self, params=None):
        """dict: (Abstract method) Return dict having the collection of pins of a template."""
        # TODO: implement this
        pass

    def __init__(self, grid, tracks: dict = None, nodes: list = None):
        """The constructor function.

        Parameters
        ----------
        grid: laygo2.object.grid.RoutingGrid
            The grid object that this object refers to for constructing routing wires.
        tracks: dict
            The dictionary that contains track information.
            The format of its element is "track_name": [index, "net_name"].
        nodes: list
            The list that contains layout objects to be connected through the routing channel.
            The element could be either one of the following:
                laygo2.object.physical.Instance,
                laygo2.object.physical.VirtualInstance,
                laygo2.object.physical.Rect.
            For Instance or VirtualInstance elements, the netname parameters of their pins need to be set to the
            name of net (netname) which the pins are connected to.
            For Rect elements, their netname parameters should be set.
        """
        # Assign parameters.
        self.grid = grid
        if tracks is not None:
            self.tracks = tracks
        else:
            self.tracks = dict()
        if nodes is not None:
            self.nodes = nodes
        else:
            self.nodes = []

    def add_track(self, name: str, index: int, netname: str = None):
        """Add a track to the mesh.

        Parameters
        ----------
        name: str
            The name of the track to be added.
        index: int
            The index of the track.
            For a horizontal track, it should be [None, i] where i is the
            value of abstract coordinate to place the routing track.
            For a vertical track, it should be [i, None].
        netname: str
            The net name of the track.
        """
        if netname is None:
            netname = name
        self.tracks[name] = [index, netname]

    def add_node(self, obj):
        """Add a node object to the mesh.

        Parameters
        ----------
        obj: laygo2.object.physical.Instance or laygo2.object.physical.VirtualInstance or laygo2.object.pythiscl.Rect
            The object to be added to the mesh as a node.
        """

        if isinstance(obj, list):
            self.nodes += obj
        else:
            self.nodes.append(obj)

    def generate(self):
        """Generate a routing mesh.

        Returns
        -------
        laygo2.object.physical.VirtualInstance: the generated routing object.

        Example
        -------
        >>> import laygo2
        >>> from laygo2.object.grid import CircularMapping as CM
        >>> from laygo2.object.grid import CircularMappingArray as CMA
        >>> from laygo2.object.grid import OneDimGrid, RoutingGrid
        >>> from laygo2.object.template import NativeInstanceTemplate
        >>> from laygo2.object.physical import Instance, Rect
        >>> # Routing grid construction (not needed if laygo2_tech is set up).
        >>> gv = OneDimGrid(name="gv", scope=[0, 50], elements=[0])
        >>> gh = OneDimGrid(name="gv", scope=[0, 50], elements=[0])
        >>> wv = CM([10])           # vertical (xgrid) width
        >>> wh = CM([10])   # horizontal (ygrid) width
        >>> ev = CM([10])           # vertical (xgrid) extension
        >>> eh = CM([10])   # horizontal (ygrid) extension
        >>> e0v = CM([5])          # vert. extension (for zero-length wires)
        >>> e0h = CM([5])  # hori. extension (for zero-length wires)
        >>> lv = CM([['M1', 'drawing']], dtype=object)  # layer information
        >>> lh = CM([['M2', 'drawing']], dtype=object)
        >>> plv = CM([['M1', 'pin']], dtype=object) # pin layers
        >>> plh = CM([['M2', 'pin']], dtype=object)
        >>> xcolor = CM([None], dtype=object)  # not multi-patterned
        >>> ycolor = CM([None]*3, dtype=object)
        >>> primary_grid = 'horizontal'
        >>> tvia = NativeInstanceTemplate(libname='tlib', cellname='via0')  # via
        >>> viamap = CMA(elements=[[tvia, tvia, tvia]], dtype=object)
        >>> g = RoutingGrid(name='mygrid', vgrid=gv, hgrid=gh,
        >>>                 vwidth=wv, hwidth=wh,
        >>>                 vextension=ev, hextension=eh,
        >>>                 vlayer=lv, hlayer=lh,
        >>>                 pin_vlayer=plv, pin_hlayer=plh,
        >>>                 viamap=viamap, primary_grid=primary_grid,
        >>>                 xcolor=xcolor, ycolor=ycolor,
        >>>                 vextension0=e0v, hextension0=e0h)
        >>> # Create node objects.
        >>> r0 = Rect(xy=[[45, -5], [55, 5]], layer=['M1', 'drawing'], netname = "A")
        >>> r1 = Rect(xy=[[95, -5], [105, 5]], layer=['M1', 'drawing'], netname = "A")
        >>> i0_pins = dict()
        >>> i0_pins['X'] = laygo2.object.physical.Pin(xy=[[45, 95], [55, 105]],
        >>>     layer = ['M1', 'drawing'], netname = 'A')
        >>> i0 = laygo2.object.physical.Instance(name="I0", xy=[150,0],
        >>>     libname="mylib", cellname="mycell", shape=[3, 2], pitch=[200,200],
        >>>     unit_size=[100, 100], pins=i0_pins, transform='R0')
        >>>
        >>> # Create a routing mesh.
        >>> rm = laygo2.object.routing.RoutingMesh(grid = g)
        >>> rm.add_track(name = "A", index = [None, 5], netname = "A")
        >>> rm.add_node(r0)
        >>> rm.add_node(r1)
        >>> rm.add_node(i0)
        >>> rinst = rm.generate()
        >>>
        >>> # Print horizontal wire
        >>> print(rinst.pins["A"].xy)
        [[ 50 250]
         [200 250]]
        >>> # Vertical wire to r0
        >>> print(rinst.native_elements["A_0_0"].xy)
        [[ 50   0]
         [ 50 250]]
        >>> # Vertical wire to r1
        >>> print(rinst.native_elements["A_1_0"].xy)
        [[100   0]
         [100 250]]
        >>> # Vertical wire to i0
        >>> print(rinst.native_elements["A_2_0"].xy)
        [[200 100]
         [200 250]]
        """
        g = self.grid
        tr = self.tracks
        nds = self.nodes

        # Variables for VirtualInstance construction.
        nelements = dict()
        pins = dict()
        for tn, t in tr.items():  # for each track
            ti = t[0]  # track index
            tnn = t[1]  # track netname
            mn = []
            for n in nds:
                if isinstance(n, Instance) or isinstance(n, VirtualInstance):
                    for pn, p in n.pins.items():
                        if p.netname == tnn:
                            _mn = (g.mn(p)[0] + g.mn(p)[1])/2
                            mn.append(_mn)
                elif isinstance(n, Rect):
                    if n.netname == tnn:
                        mn.append(g.mn(n)[0])
                elif isinstance(n, Pin):
                    if n.netname == tnn:
                        mn.append(g.mn(n)[0])
            # Do routing
            # if ti[0] is None:  # horizontal track
            #    _t = [None, ti[1]]
            # else:
            #    _t = [ti[0], None]
            # r = g.route_via_track(mn=mn, track=_t)
            r = g.route_via_track(mn=mn, track=ti)

            # Wrap the generated routing structure into a VirtualInstance
            for i, _r in enumerate(r):
                if isinstance(_r, list):
                    for j, __r in enumerate(_r):
                        nelements[tn + "_" + str(i) + "_" + str(j)] = __r
                else:
                    nelements[tn + "_" + str(i)] = _r
            pins[tn] = Pin(xy=r[-1].xy, layer=r[-1].layer, netname=tnn)

        # Instantiate a VirtualInstance.
        rinst = VirtualInstance(
            name=tn,
            xy=np.array([0, 0]),
            libname="mylib",
            cellname="rtrack_" + tn + "_" + tnn,
            native_elements=nelements,
            transform="R0",
            pins=pins,
        )
        return rinst
