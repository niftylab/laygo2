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

import laygo2.object.database
import laygo2.object
import laygo2.tech.import_yaml as iy

import numpy as np
import yaml
import pprint

# Grid library
def load_grids(templates, libname:str = None, params:dict = None, tech_fname:str = None):
    """
    Load grids to a grid library object.

    Parameters
    ----------
    templates: laygo2.object.database.TemplateLibrary
        The template library object that contains via templates.
    """
    
    libname, templates_row, grids_row = iy.load_tech_yaml(tech_fname)
    glib        = laygo2.object.database.GridLibrary( name = libname)

    for gn, gdict in grids_row.items():
        gv = laygo2.object.OneDimGrid(name=gn + '_v', scope=gdict['vertical']['scope'],
                                           elements=gdict['vertical']['elements'])
        gh = laygo2.object.OneDimGrid(name=gn + '_h', scope=gdict['horizontal']['scope'],
                                           elements=gdict['horizontal']['elements'])
        if gdict['type'] == 'placement':  # placement grid
            g = laygo2.object.PlacementGrid(name=gn, vgrid=gv, hgrid=gh)
            glib.append(g)
        elif gdict['type'] == 'routing':  # routing grid
            vwidth       = laygo2.object.CircularMapping(elements=gdict['vertical']['width'])
            hwidth       = laygo2.object.CircularMapping(elements=gdict['horizontal']['width'])
            vextension   = laygo2.object.CircularMapping(elements=gdict['vertical']['extension'])
            hextension   = laygo2.object.CircularMapping(elements=gdict['horizontal']['extension'])
            vextension0  = laygo2.object.CircularMapping(elements=gdict['vertical']['extension0'])
            hextension0  = laygo2.object.CircularMapping(elements=gdict['horizontal']['extension0'])
            vlayer       = laygo2.object.CircularMapping(elements=gdict['vertical']['layer'], dtype=object)
            hlayer       = laygo2.object.CircularMapping(elements=gdict['horizontal']['layer'], dtype=object)
            pin_vlayer   = laygo2.object.CircularMapping(elements=gdict['vertical']['pin_layer'], dtype=object)
            pin_hlayer   = laygo2.object.CircularMapping(elements=gdict['horizontal']['pin_layer'], dtype=object)
            xcolor       = laygo2.object.CircularMapping(elements=gdict['vertical']['xcolor'], dtype=object)
            ycolor       = laygo2.object.CircularMapping(elements=gdict['horizontal']['ycolor'], dtype=object)
            primary_grid = gdict['primary_grid']
            # Create the via map defined by the yaml file.
            vmap_original = gdict['via']['map']  # viamap defined in the yaml file.
            vmap_mapped   = list()  # map template objects to the via map.

            for vmap_org_row in vmap_original:
                vmap_mapped_row = []
                for vmap_org_elem in vmap_org_row:
                    vmap_mapped_row.append(templates[vmap_org_elem])
                vmap_mapped.append(vmap_mapped_row)
            viamap = laygo2.object.CircularMappingArray(elements=vmap_mapped, dtype=object)
            g = laygo2.object.RoutingGrid(name=gn, vgrid=gv, hgrid=gh,
                                               vwidth=vwidth, hwidth=hwidth,
                                               vextension=vextension, hextension=hextension,
                                               vlayer=vlayer, hlayer=hlayer,
                                               pin_vlayer=pin_vlayer, pin_hlayer=pin_hlayer,
                                               viamap=viamap, primary_grid=primary_grid,
                                               xcolor=xcolor, ycolor=ycolor,
                                               vextension0=vextension0, hextension0=hextension0)
            glib.append(g)
    if params is not None:
        glib = update(glib, params=params)
    return glib

