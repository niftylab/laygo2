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

from typing import Callable

import laygo2.tech.import_yaml as iy
import laygo2.tech.tech_grids  as lg

import laygo2.object
import numpy as np

def load_native_templates(tech_fname:str):
    """
    The method of load native templates from yaml
    
    Parameters
    ----------
    tech_fname: str
        yaml name
    """

    libname, templates_row, grids_row = iy.load_tech_yaml(tech_fname)
    tlib    = laygo2.object.database.TemplateLibrary(name = libname)

    for tn, tdict in templates_row.items():
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
    
    return tlib


def load_templates(tech_fname:str , load_udf_templates: Callable):
    """
    Load complete template library 
    
    Parameters
    ----------
    tech_fname: str
         yaml name
    load_udf_templates: Callable
        the method of load udf templates
    """

    tlib    = load_native_templates(tech_fname) # native_templates lib
    grids   = lg.load_grids(templates = tlib, tech_fname = tech_fname)  # grids lib

    libname = tlib.name

    tlib    = load_udf_templates(tlib, grids, libname) # UDF libs
    
    return tlib

