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
from .core import CircularMapping, OneDimGrid
from .routing import RoutingGrid
import laygo2.object

class UniformRoutingGrid(RoutingGrid):
    """
    A class tahta implements wire connections in an uniform and flexible abstract coordinate system.

    Attributes
    ----------
    type : str
    xres : int
    yres : int
    xwidth : int
    ywidth : int
    xextension : int
    yextension : int
    xlayer : str
    ylayer : str
    pin_xlayer : str
    pin_ylayer : str
    via : UserDefinedTemplates
    xcolor : 
    ycolor :
    primary_grid : str

    Methods
    -------
    __init__()
    via()

    """
    type = 'uniform_routing'

    xres = None
    yres = None
    xwidth = None
    ywidth = None
    xextension = None
    yextension = None
    xextension0 = None
    yextension0 = None
    xlayer = None
    ylayer = None
    pin_xlayer = None
    pin_ylayer = None
    via = None
    xcolor = None
    ycolor = None
    primary_grid = None

    def __init__(self, name, xres, yres, xwidth, ywidth, xextension, yextension, xlayer, ylayer, pin_xlayer, pin_ylayer,
                via, xcolor, ycolor, primary_grid='horizontal', xextension0=None, yextension0=None):
        """
        Constructor function of UniformRoutingGrid class.

        Parameters
        ----------
        name : str
            Routing object name
        xres : laygo2.OneDimGrid
            OneDimGrid of x-coordinate system
        yres : laygo2.OneDimGrid
            OneDimGrid of y-coordinate system
        
        """
        xres=OneDimGrid(name='xres', scope=[0,xres], elements=[0])
        yres=OneDimGrid(name='yres', scope=[0,yres], elements=[0])
        xwidth=CircularMapping([xwidth])
        ywidth=CircularMapping([ywidth])

        xextension=CircularMapping([ywidth.elements[0]/2])
        yextension=CircularMapping([xwidth.elements[0]/2])
        xextension0=CircularMapping([ywidth.elements[0]*2])
        yextension0=CircularMapping([xwidth.elements[0]*2])
        xlayer=CircularMapping([[xlayer,'drawing']], dtype=object)
        ylayer=CircularMapping([[ylayer,'drawing']], dtype=object)
        pin_xlayer=CircularMapping([[xlayer,'pin']], dtype=object)
        pin_xlayer=CircularMapping([[xlayer,'pin']], dtype=object)
        xcolor=CircularMapping(['not MPT'], dtype=object)
        ycolor=CircularMapping(['not MPT'], dtype=object)

        self.xwidth=xwidth
        self.ywidth=ywidth
        self.xlayer=xlayer
        self.ylayer=ylayer
        self.xextension=xextension
        self.yextension=yextension

        
        RoutingGrid.__init__(self, name=name, vgrid=xres, hgrid=yres, vwidth=xwidth, hwidth=ywidth, vextension=xextension, hextension=yextension,
        vlayer=xlayer, hlayer=ylayer, pin_vlayer=pin_xlayer, pin_hlayer=pin_ylayer, viamap=via, xcolor=xcolor, ycolor=ycolor,
        primary_grid='vertical', vextension0=xextension0, hextension0=yextension0)
    
    def via(self, mn=np.array([0,0]), params=None):
        # Supporting different type of via cut patterns(default,LRG,BAR,...)
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
        tvia = self.viamap
        if params == None:
            params={'cut':'Default'}
        else:
            pass
        params['xlayer']=self.xlayer.elements[0][0]
        params['ylayer']=self.ylayer.elements[0][0]
        via = tvia.generate(params=params)
        via.xy = self[mn]
        return via

    def route(self, mn, direction=None, via_tag=None):
       
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
            if via_tag[0] is True:
                route.append(self.via(mn=_mn[0][0], params=None))
            elif via_tag[0] != False:
                route.append(self.via(mn=_mn[0][0], params={'cut':via_tag[0]}))
            else:
                pass
        # routing wires
        for i, __mn in enumerate(_mn):
            xy0 = self.abs2phy[__mn[0]]
            xy1 = self.abs2phy[__mn[1]]
            _xy = np.array([[xy0[0], xy0[1]], [xy1[0], xy1[1]]])
            if np.all(xy0 == xy1):  # if two points are identical, generate a metal stub on the bottom layer.
                if (direction == 'vertical') or ((direction is None) and (self.primary_grid == 'vertical')):
                    width = self.xwidth[__mn[0][0]]
                    yextension = int(width/2)
                    xextension = self.xextension0[__mn[0][0]]
                    layer = self.xlayer[__mn[0][0]]
                    color = self.xcolor[__mn[0][0]%self.xcolor.shape[0]] # xcolor is determined by its grid layer.
                else:
                    width = self.ywidth[__mn[0][1]]
                    yextension = self.yextension0[__mn[0][1]]
                    xextension = int(width/2)
                    layer = self.ylayer[__mn[0][1]]
                    color = self.ycolor[__mn[0][1]%self.ycolor.shape[0]] # ycolor is determined by its grid layer.
            else:
                if (xy0[0] == xy1[0]) or (direction == 'vertical'):  # vertical routing
                    width = self.xwidth[__mn[0][0]]
                    yextension = int(width/2)
                    xextension = self.xextension[__mn[0][0]]
                    layer = self.xlayer[__mn[0][0]]
                    color = self.xcolor[__mn[0][0]%self.xcolor.shape[0]] # xcolor is determined by its grid layer.

                else:  # horizontal routing
                    width = self.ywidth[__mn[0][1]]
                    yextension = self.yextension[__mn[0][1]]
                    xextension = int(width/2)
                    layer = self.ylayer[__mn[0][1]]
                    color = self.ycolor[__mn[0][1]%self.ycolor.shape[0]] # ycolor is determined by its grid layer.
            p = laygo2.object.physical.Rect(xy=_xy, layer=layer, hextension=yextension, vextension=xextension, color=color)
            route.append(p)
            # via placement
            if via_tag is None:
                if (i > 0) and (i < mn.shape[0] - 1):
                    route.append(self.via(mn=__mn[0], params=None))
            else:
                if via_tag[i + 1] == True:
                    route.append(self.via(mn=__mn[1], params=None))
                elif via_tag[i+1] != False:
                    route.append(self.via(mn=__mn[1], params={'cut':via_tag[i+1]}))
                else:
                    pass
        if len(route) == 1:  # not isinstance(mn[0][0], list):
            return route[0]
        else:
            return route