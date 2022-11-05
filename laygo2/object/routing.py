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


class RoutingChannel:
    grid = None
    """laygo2.object.grid.RoutingGrid: A routing grid object that this routing channel refers to.
    """

    direction = "horizontal"
    """str: 
    """

    tracks = dict()
    """dict: A dictionary that contains net names as keys and routing track index as values.
    """

    nodes = []
    """list: A list that contains coordinates and layout object to connected through the routing channel.
    """

    range = [None, None]
    """[int or None, int or None]: The minimum and maximum track indices covered by the 
    routing channel, for automatic routing.
    """

    def __init__(self, grid, tracks:dict = None, nodes:list = None):
        """Constructor"""
        # Assign parameters.
        self.grid = grid
        if tracks is not None:
            self.tracks = tracks
        if nodes is not None:
            self.nodes = nodes

    def generate(self, tracks=None, nodes=None):
        if tracks is None:
            tracks = self.tracks
        if nodes is None:
            nodes = self.nodes

        g = self.grid
        r_list = []

        for tn, ti in tracks.items():
            # Find out routing points.
            mn = []
            for n in nodes:
                if isinstance(n, Instance) or isinstance(n, VirtualInstance):
                    for pn, p in n.pins.items():
                        if p.netname == tn:
                            mn.append(g.mn(p.xy)[0])
                elif isinstance(n, Rect):
                    if n.netname == tn:
                        mn.append(g.mn(n.xy)[0])
            # Do routing
            if direction == "horizontal":
                t = [None, ti]
            else:
                t = [ti, None]
            r = rg.route_via_track(mn=mn, track=t)
            r_list.append(r[-1])


# class RoutingMesh:
#    # TODO: implement this
#    pass
