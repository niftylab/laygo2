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
from laygo2.object.physical import *


class RoutingMesh:
    grid = None
    """laygo2.object.grid.RoutingGrid: A routing grid object that this routing mesh refers to.
    """

    tracks = dict()
    """dict: A dictionary that contains net names as keys and horizontal routing track index as values.
    """

    nodes = []
    """list: A list that contains coordinates and layout object to connected through the routing channel.
    """

    hrange = [None, None]
    """[int or None, int or None]: The minimum and maximum track indices covered by the 
    horizontal routing channel, for automatic routing.
    """
    
    vrange = [None, None]
    """[int or None, int or None]: The minimum and maximum track indices covered by the 
    vertical routing channel, for automatic routing.
    """


    def __init__(self, grid, tracks:dict = None, nodes:list = None):
        """Constructor"""
        # Assign parameters.
        self.grid = grid
        if tracks is not None:
            self.tracks = tracks
        if nodes is not None:
            self.nodes = nodes

    def add_track(self, name:str, index:int, netname:str = None):
        if netname is None:
            netname = name
        self.tracks[name] = [index, netname]
    
    def add_node(self, obj):
        if isinstance(obj, list):
            self.nodes += obj
        else:
            self.nodes.append(obj)

    def generate(self):
        g = self.grid
        tr = self.tracks
        nds = self.nodes

        return self._construct_rdict(grid=g, tracks=tr, nodes=nds) 

    def _construct_rdict(self, grid, tracks, nodes):
        """Construct a routing directiony for each direction for generate()."""
        
        # Variables for VirtualInstance construction.
        nelements = dict()
        pins = dict()
        for tn, t in tracks.items():  # for each track
            ti = t[0]  # track index
            tnn = t[1]  # track netname
            mn = []
            for n in nodes:
                if isinstance(n, Instance) or isinstance(n, VirtualInstance):
                    for pn, p in n.pins.items():
                        if p.netname == tnn:
                            mn.append(grid.mn(p)[0])
                elif isinstance(n, Rect):
                    if n.netname == tnn:
                        mn.append(grid.mn(n)[0])
            # Do routing
            if ti[0] is None:  # horizontal track
                t = [None, ti[1]]
            else:
                t = [ti[0], None]
            r = grid.route_via_track(mn=mn, track=t)

            # Wrap the generated routing structure into a VirtualInstance
            for i, _r in enumerate(r):
                if isinstance(_r, list):
                    for j, __r in enumerate(_r):
                        nelements[tn+"_"+str(i)+"_"+str(j)]=__r
                else:
                    nelements[tn+"_"+str(i)]=_r
            pins[tn] = Pin(xy=r[-1].xy, layer=r[-1].layer, netname=tnn)
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
