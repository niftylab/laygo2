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
import laygo2.object.template
import laygo2.object.physical
import laygo2.object.database

import numpy as np
import yaml, pprint

# Load laygo1 technology parameters

def convert_laygo1_laygo2( laygo1_template_fname, laygo1_grid_fname, scale, tech_fname):
    # Script to convert laygo1 template/grid files to laygo2 tech file


    if laygo1_template_fname is not None:
        with open(laygo1_template_fname, 'r') as stream:
            try:
                laygo1_template_params = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
    if laygo1_grid_fname is not None:
        with open(laygo1_grid_fname, 'r') as stream:
            try:
                laygo1_grid_params = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
    #pprint.pprint(laygo1_template_params)

    # Construct tech parameter databse
    tech_params = dict()
    # Construct template database
    if laygo1_template_fname is not None:
        templates = dict()
        for ln, ldict in laygo1_template_params.items():
            templates[ln] = dict()
            for tn, tdict in ldict.items():
                tdict_new = dict()
                unit_size = np.round(scale*(np.array(tdict['xy1'])-np.array(tdict['xy0'])))  # scale up
                tdict_new['unit_size'] = np.array(unit_size, dtype=np.int).tolist()
                xy = np.round(scale*(np.array([tdict['xy0'], tdict['xy1']]))) 
                tdict_new['xy'] = np.array(xy, dtype=np.int).tolist()
                # Construct pin dictionary
                tdict_new['pins'] = dict()
                if 'pins' in tdict:
                    for pn, pdict in tdict['pins'].items():
                        tdict_new['pins'][pn] = dict()
                        if 'netname' in pdict:
                            tdict_new['pins'][pn]['netname'] = pdict['netname']
                        if 'layer' in pdict:
                            tdict_new['pins'][pn]['layer'] = pdict['layer']
                        xy = np.round(scale*(np.array([pdict['xy0'], pdict['xy1']])))
                        tdict_new['pins'][pn]['xy'] = np.array(xy, dtype=np.int).tolist()
                templates[ln][tn] = tdict_new
        #pprint.pprint(templates)
        tech_params['templates'] = templates

    # Constructe grid database
    if laygo1_grid_fname is not None:
        grids = dict()
        for ln, ldict in laygo1_grid_params.items():
            grids[ln] = dict()
            for gn, gdict in ldict.items():

                print(gn)
                gdict_new = dict()
                gdict_new['type'] = gdict['type']  # grid type
                if gdict_new['type'] == 'route':  # routing grid
                    gdict_new['type'] = 'routing'
                    gn = 'routing' + '_' + gn[7] + gn[10:]  # modify grid naming
                if gdict_new['type'] == 'routing':  # routing grid
                    gdict_new['primary_grid'] = 'horizontal'  # primary routing direction
                # vertical grid
                vdict = dict()
                _scope = np.round([scale*gdict['xy0'][0], scale*gdict['xy1'][0]])
                vdict['scope'] = np.array(_scope, dtype=np.int).tolist()
                _elements = np.round(scale*np.array(gdict['xgrid']))
                vdict['elements'] = np.array(_elements, dtype=np.int).tolist()
                if gdict_new['type'] == 'routing':  # routing grid
                    vdict['layer'] = np.asarray(gdict['xlayer']).tolist()
                    vdict['pin_layer'] = [[i[0], 'pin'] for i in gdict['xlayer']]
                    _width = np.round(scale*np.array(gdict['xwidth']))
                    vdict['width'] = np.array(_width, dtype=np.int).tolist()
                    # assume nominal extension is 0.5*width,and 2*width for zero-length wires.
                    vdict['extension'] = np.array(_width/2, dtype=np.int).tolist()
                    vdict['extension0'] = np.array(_width*2, dtype=np.int).tolist()
                    if 'xcolor' in gdict:
                        vdict['xcolor'] = np.asarray(gdict['xcolor']).tolist()
                    else:
                        vdict['xcolor'] = ['not MPT']*len(vdict['width'])
                gdict_new['vertical'] = vdict 

                # horizontal grid
                hdict = dict()
                _scope = np.round([scale*gdict['xy0'][1], scale*gdict['xy1'][1]])
                hdict['scope'] = np.array(_scope, dtype=np.int).tolist()
                _elements = np.round(scale*np.array(gdict['ygrid']))
                hdict['elements'] = np.array(_elements, dtype=np.int).tolist()
                if gdict_new['type'] == 'routing':  # routing grid
                    hdict['layer'] = np.asarray(gdict['ylayer']).tolist()
                    hdict['pin_layer'] = [[i[0], 'pin'] for i in gdict['ylayer']]
                    _width = np.round(scale*np.array(gdict['ywidth']))
                    hdict['width'] = np.array(_width, dtype=np.int).tolist()
                    # assume nominal extension is 0.5*width,and 2*width for zero-length wires.
                    hdict['extension'] = np.array(_width/2, dtype=np.int).tolist()
                    hdict['extension0'] = np.array(_width*2, dtype=np.int).tolist()
                    if 'ycolor' in hdict:
                        hdict['ycolor'] = np.asarray(gdict['ycolor']).tolist() # coloring function added
                    else:
                        hdict['ycolor'] = ['not MPT']*len(hdict['width'])
                gdict_new['horizontal'] = hdict 
                # via map

                if gdict_new['type'] == 'routing':  # routing grid
                    gdict_new['via'] = dict()
                    gdict_new['via']['map'] = \
                        np.zeros((len(vdict['elements']), len(hdict['elements'])), dtype=np.object)
                    for vn, lvcoord in gdict['viamap'].items():
                        if isinstance(lvcoord[0], int):
                            _lvcoord = [lvcoord]  # convert to list
                        else:
                            _lvcoord = lvcoord
                        for _vcoord in _lvcoord:
                            gdict_new['via']['map'][tuple(_vcoord)] = vn
                    gdict_new['via']['map'] = gdict_new['via']['map'].tolist()
                    # add the via object to template dictionary
                    if laygo1_template_fname is not None:
                        if vn in tech_params['templates'][ln]:
                            tech_params['templates'][ln][vn] = {
                                'unit_size': [0, 0],
                                'xy': [[0, 0], [0, 0]],
                                }

                grids[ln][gn] = gdict_new
        #pprint.pprint(templates)
        tech_params['grids'] = grids
            
    # Export to a yaml file
    with open(tech_fname, 'w') as stream:
        try:
            yaml.dump(tech_params, stream)
        except yaml.YAMLError as exc:
            print(exc)

