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
This module implements interfaces with yaml files.

"""

__author__ = "Jaeduk Han"
__maintainer__ = "Jaeduk Han"
__status__ = "Prototype"

import yaml
import os.path
import laygo2

def export_template(template, filename, mode='append'):
    libname = template.libname
    cellname = template.cellname
    pins = template.pins()

    db = dict()
    if mode == 'append':  # in append mode, the template is appended to 'filename' if the file exists.
        if os.path.exists(filename):
            with open(filename, 'r') as stream:
                db = yaml.load(stream, Loader=yaml.FullLoader)
    if libname not in db:
        db[libname] = dict()
    db[libname][cellname] = template.export_to_dict()
    with open(filename, 'w') as stream:
        yaml.dump(db, stream)
    print("Your design was translated into YAML format.")

#filename=libname+'_templates.yaml'

def import_template(filename):
    # load yaml file
    if os.path.exists(filename):
        with open(filename, 'r') as stream:
            db = yaml.load(stream, Loader=yaml.FullLoader)
    libname = list(db.keys())[0]  # assuming there's only one library defined in each file.
    # create template library
    tlib = laygo2.object.database.TemplateLibrary(name=libname)
    # read out the yaml file entries and build template objects
    for tn, tdict in db[libname].items():
        pins = dict()
        if 'pins' in tdict:
            for pinname, pdict in tdict['pins'].items():
                pins[pinname] = laygo2.object.Pin(xy=pdict['xy'], layer=pdict['layer'], netname=pdict['netname'])
        t = laygo2.object.NativeInstanceTemplate(libname=libname, cellname=tn, bbox=tdict['bbox'], pins=pins)
        tlib.append(t)
    return tlib

