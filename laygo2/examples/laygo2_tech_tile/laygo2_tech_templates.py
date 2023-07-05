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

import yaml
from laygo2.object.template.tile import TileMOSTemplate, TileTapTemplate
from laygo2_tech.laygo2_tech_grids import load_grids
from laygo2.object import *

tech_fname = './laygo2_tech/laygo2_tech.yaml'

with open(tech_fname, 'r') as stream:
    try:
        tech_params = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

libname       =   list(tech_params['templates'].keys())[0]  # libname
templates_raw = tech_params['templates'][libname]
grids_raw     = tech_params['grids'][libname]


def load_templates():
    """Load template to a template library object"""
    
    tlib    = laygo2.object.database.TemplateLibrary(name = libname)

    # 1.Native template load
    for tn, tdict in templates_raw.items():
        # bounding box
        bbox = np.array(tdict['xy'])
        # pins
        pins = None
        if 'pins' in tdict:
            pins = dict()
            for pn, _pdict in tdict['pins'].items():
                pins[pn] = laygo2.object.Pin(xy=_pdict['xy'], layer=_pdict['layer'], netname=pn)
        
        t = laygo2.object.NativeInstanceTemplate(libname=libname, cellname=tn, bbox=bbox, pins=pins)
        tlib.append(t)
    

    # 2.UserDefinedTemplate load
    glib = load_grids(tlib)

    placement_pattern = [ "gbndl", "bndl", "dmyl",  "core", "dmyr", "bndr", "gbndr" ]
    transform_pattern = dict( gbndl = "R0", dmyl = "R0", bndl  = "R0", core  = "R0",
                                            dmyr = "MY", bndr  = "R0", gbndr = "R0" )
    routing_map       = dict( G = 3, S = 1, D = 2, G_extension0_x = [None,None], S_extension0_m = [None, None], D_extension0_m = [None, None])
    
    nmos_ulvt = dict(
        core  = 'nmos4_fast_center_nf2',
        dmyr  = 'nmos4_fast_dmy_nf2',
        dmyl  = 'nmos4_fast_dmy_nf2',
        bndr  = 'nmos4_fast_boundary',
        bndl  = 'nmos4_fast_boundary',
        gbndr = 'nmos4_fast_boundary',
        gbndl = 'nmos4_fast_boundary',
        grid  = "routing_12_mos",
    )

    pmos_ulvt = dict(
        core  = 'pmos4_fast_center_nf2',
        dmyr  = 'pmos4_fast_dmy_nf2',
        dmyl  = 'pmos4_fast_dmy_nf2',
        bndr  = 'pmos4_fast_boundary',
        bndl  = 'pmos4_fast_boundary',
        gbndr = 'pmos4_fast_boundary',
        gbndl = 'pmos4_fast_boundary',
        grid  = "routing_12_mos",
        )

    ptap_ulvt = dict(
        core  = 'ntap_fast_center_nf2_v2',
        dmyr  = 'nmos4_fast_dmy_nf2',
        dmyl  = 'nmos4_fast_dmy_nf2',
        bndr  = 'ntap_fast_boundary',
        bndl  = 'ntap_fast_boundary',
        gbndr = 'ntap_fast_boundary',
        gbndl = 'ntap_fast_boundary',
        grid  = "routing_12_mos",
    )

    ntap_ulvt = dict(
        core  = 'ptap_fast_center_nf2_v2',
        dmyr  = 'pmos4_fast_dmy_nf2',
        dmyl  = 'pmos4_fast_dmy_nf2',
        bndr  = 'ptap_fast_boundary',
        bndl  = 'ptap_fast_boundary',
        gbndr = 'ptap_fast_boundary',
        gbndl = 'ptap_fast_boundary',

        grid  = "routing_12_mos",
    )
   
    gen_list = [["nmos", nmos_ulvt], ["pmos", pmos_ulvt]]
    for name, placement_map in gen_list:
        grid_name  = placement_map["grid"]

        temp = TileMOSTemplate( tlib, glib, grid_name, routing_map, placement_map, placement_pattern, transform_pattern, name)
        tlib.append(temp)
    
    gen_list = [["ptap", ntap_ulvt], ["ntap", ptap_ulvt] ]
    for name, placement_map in gen_list:
        grid_name  = placement_map["grid"]
        
        temp       = TileTapTemplate( tlib, glib, grid_name, routing_map, placement_map, placement_pattern, transform_pattern, name)
        tlib.append(temp)

    return tlib

