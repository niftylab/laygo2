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

from laygo2.object import *
#import laygo2.object.database
#import laygo2.object.grid

import numpy as np

# Base technology object
class BaseTechnology:
    """
    A base class that implements basic functions for 
    technology parameters and objects, such as TemplateLibrary and 
    GridLibrary objects.
    """

    tech_params = None
    """dict: Dictionary contains technology parameters"""

    tech_templates = None
    """laygo2.object.database.TemplateLibrary: The template library 
    object that contains primitive templates
    """
    tech_grids = None
    """laygo2.object.database.GridLibrary: The template library 
    object that contains primitive grids
    """

    def __init__(self, tech_params, libname=None):
        """Constructor"""
        # Library name
        if libname is None:
            ln = list(tech_params['templates'].keys())[0]
        else:
            ln = libname
        self.tech_params = tech_params
        self.tech_templates = self.load_tech_templates(libname=libname)
        self.tech_grids = self.load_tech_grids(templates = self.tech_templates,
                                               libname=libname)

    def load_tech_templates(self, libname=None):
        """Load templates and construct a template library object.
            libname: optional, str
                The name of library to be loaded. 
                By default, the first library in tech_params['templates'] is used.
        """
        
        # Library name
        if libname is None:
            ln = list(self.tech_params['templates'].keys())[0]
        else:
            ln = libname
        # Template library
        tlib    = laygo2.object.database.TemplateLibrary(name = ln)
        # Include templates as needed, as follows:
        '''
        t1 = laygo2.object.NativeInstanceTemplate()
        tlib.append(t1)
        t2 = laygo2.object.NativeInstanceTemplate()
        tlib.append(t2)
        '''
        # Bind to self.tech_templates
        self.tech_templates = tlib
        return tlib
     
    def load_tech_grids(self, templates, libname=None, params=None):
        """
        Load technology grids to a grid library object.

        Parameters
        ----------
        libname: optional, str
            The name of library to be loaded. 
            By default, the first library in tech_params['grids'] is used.
        params: optional, dict
            The dictionary that contains optional parameters for update() function.
        """
        # Library name
        if libname is None:
            ln = list(self.tech_params['grids'].keys())[0]
        else:
            ln = libname
        # Grid library
        glib = laygo2.object.database.GridLibrary(name=ln)

        for gn, gdict in self.tech_params['grids'][ln].items():
            gv = laygo2.object.grid.OneDimGrid(name=gn + '_v', scope=gdict['vertical']['scope'],
                                               elements=gdict['vertical']['elements'])
            gh = laygo2.object.grid.OneDimGrid(name=gn + '_h', scope=gdict['horizontal']['scope'],
                                               elements=gdict['horizontal']['elements'])
            if gdict['type'] == 'placement':  # placement grid
                g = laygo2.object.grid.PlacementGrid(name=gn, vgrid=gv, hgrid=gh)
                glib.append(g)
            elif gdict['type'] == 'routing':  # routing grid
                vwidth = laygo2.object.grid.CircularMapping(elements=gdict['vertical']['width'])
                hwidth = laygo2.object.grid.CircularMapping(elements=gdict['horizontal']['width'])
                vextension = laygo2.object.grid.CircularMapping(elements=gdict['vertical']['extension'])
                hextension = laygo2.object.grid.CircularMapping(elements=gdict['horizontal']['extension'])
                vextension0 = laygo2.object.grid.CircularMapping(elements=gdict['vertical']['extension0'])
                hextension0 = laygo2.object.grid.CircularMapping(elements=gdict['horizontal']['extension0'])
                vlayer = laygo2.object.grid.CircularMapping(elements=gdict['vertical']['layer'], dtype=object)
                hlayer = laygo2.object.grid.CircularMapping(elements=gdict['horizontal']['layer'], dtype=object)
                pin_vlayer = laygo2.object.grid.CircularMapping(elements=gdict['vertical']['pin_layer'], dtype=object)
                pin_hlayer = laygo2.object.grid.CircularMapping(elements=gdict['horizontal']['pin_layer'], dtype=object)
                xcolor = laygo2.object.grid.CircularMapping(elements=gdict['vertical']['xcolor'], dtype=object)
                ycolor = laygo2.object.grid.CircularMapping(elements=gdict['horizontal']['ycolor'], dtype=object)
                primary_grid = gdict['primary_grid']
                # Create the via map defined by the yaml file.
                vmap_original = gdict['via']['map']  # viamap defined in the yaml file.
                vmap_mapped = list()  # map template objects to the via map.
                for vmap_org_row in vmap_original:
                    vmap_mapped_row = []
                    for vmap_org_elem in vmap_org_row:
                        vmap_mapped_row.append(templates[vmap_org_elem])
                    vmap_mapped.append(vmap_mapped_row)
                viamap = laygo2.object.grid.CircularMappingArray(elements=vmap_mapped, dtype=object)
                g = laygo2.object.grid.RoutingGrid(name=gn, vgrid=gv, hgrid=gh,
                                                   vwidth=vwidth, hwidth=hwidth,
                                                   vextension=vextension, hextension=hextension,
                                                   vlayer=vlayer, hlayer=hlayer,
                                                   pin_vlayer=pin_vlayer, pin_hlayer=pin_hlayer,
                                                   viamap=viamap, primary_grid=primary_grid,
                                                   xcolor=xcolor, ycolor=ycolor,
                                                   vextension0=vextension0, hextension0=hextension0)
                glib.append(g)

        # Update grid based on input parameters.
        if params is not None:
            glib = self.update(glib, params=params)

        # Bind to self.tech_grid
        self.tech_grid = glib
        return glib

    def update(self, grid_lib, params):
        return grid_lib  # do nothing for base class

