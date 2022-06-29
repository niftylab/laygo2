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

"""
This module implements interfaces with gds files.
"""


# TODO: Implement import functions (similar to load in python-gdsii)

import logging
import pprint

import numpy as np

import laygo2.object


#export functions
def export_from_laygo(db, filename, cellname=None, scale = 1e-9, layermapfile="default.layermap",
                      physical_unit=1e-9, logical_unit=0.001, pin_label_height=0.1,
                      pin_annotate_layer=['text', 'drawing'], text_height=0.1,
                      abstract_instances = False, abstract_instances_layer = ['prBoundary', 'drawing']):
    """
    Export specified laygo2 object(s) to a GDS file

    Parameters
    ----------
    db : laygo.object.database.Library
        a library object that designs to be exported.
    filename : str
        the name of the output file.
    cellname : list or str or None.
        the name of cells to be exported. If None, all cells in the libname are exported.
    scale : float
        the scaling factor that converts integer coordinates to actual ones (mostly 1e-6, to convert 1 to 1um).
    layermapfile : str
        the name of layermap file.
    physical_unit : float, optional
        GDS physical unit.
    logical_unit : float, optional
        GDS logical unit.
    pin_label_height : float, optional
        the height of pin label.
    pin_annotate_layer : [str, str], optional
        pin annotate layer name (used when pinname is different from netname).
    text_height : float, optional
        the height of text
    """
    raise Exception("GDS support is currently disabled. Will be reinvented by using gds-tk.")


def export(db, filename, cellname=None, scale = 1e-9, layermapfile="default.layermap",
           physical_unit=1e-9, logical_unit=0.001, pin_label_height=0.1,
           pin_annotate_layer=['text', 'drawing'], text_height=0.1,
           abstract_instances=False, abstract_instances_layer=['prBoundary', 'drawing']):
    """See laygo2.interface.gds.export_from_laygo for details."""
    export_from_laygo(db, filename, cellname, scale, layermapfile,
                      physical_unit, logical_unit, pin_label_height,
                      pin_annotate_layer, text_height,
                      abstract_instances, abstract_instances_layer)


