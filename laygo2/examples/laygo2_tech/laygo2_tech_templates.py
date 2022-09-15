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

"""Template library for the advanced example technology. (advtech)"""

import numpy as np

import laygo2.object.template
import laygo2.object.physical
import laygo2.object.database
import laygo2_tech as tech


tech_params = tech.tech_params

# Primitive devices
def mos_bbox_func(devtype, params):
    """Computes x and y coordinate values from params."""
    unit_size = tech_params['templates']['mos'][devtype]['unit_size']
    xy0 = [0, 0]
    xy1 = [unit_size[0]*params['nf'], unit_size[1]]
    return np.array([xy0, xy1])


def pmos_bbox_func(params):
    return mos_bbox_func(devtype='pmos', params=params)


def nmos_bbox_func(params):
    return mos_bbox_func(devtype='nmos', params=params)


def mos_pins_func(devtype, params):
    """Generate a pin dictionary from params."""
    unit_size = tech_params['templates']['mos'][devtype]['unit_size']
    base_params = tech_params['templates']['mos'][devtype]['pins']
    nf = params['nf']
    sd_swap = params['sd_swap'] if 'sd_swap' in params.keys() else False
    pins = dict()
    lpin_default = 'S'
    rpin_default = 'D'
    if sd_swap:
        lpin = rpin_default
        rpin = lpin_default
    else:
        lpin = lpin_default
        rpin = rpin_default
    # pins starting from the left-most diffusion. Source if sd_swap=False, else Drain.
    _elem = []
    for fidx in range(nf // 2 + 1):
        l_xy = base_params[lpin_default]['xy'] + np.array([fidx, 0]) * unit_size*2
        l_layer = base_params[lpin_default]['layer']
        _elem.append(laygo2.object.Pin(xy=l_xy, layer=l_layer, netname=lpin))
    pins[lpin] = laygo2.object.Pin(xy=base_params[lpin_default]['xy'], layer=l_layer, netname=lpin, elements=np.array(_elem))
    # pins starting from the the other side of diffusion. Drain if sd_swap=False, else Source.
    _elem = []
    for fidx in range(((nf + 1) // 2)):
        r_xy = base_params[rpin_default]['xy'] + np.array([fidx, 0]) * unit_size*2
        r_layer = base_params[rpin_default]['layer']
        _elem.append(laygo2.object.Pin(xy=r_xy, layer=r_layer, netname=rpin))
    pins[rpin] = laygo2.object.Pin(xy=base_params[rpin_default]['xy'], layer=r_layer, netname=rpin, elements=np.array(_elem))
    # gate
    g_xy = np.array([base_params['G']['xy'][0], base_params['G']['xy'][1] + np.array([nf - 1, 0]) * unit_size])
    g_layer = base_params['G']['layer']
    pins['G'] = laygo2.object.Pin(xy=g_xy, layer=g_layer, netname='G')
    #g_xy = base_params['G']['xy'] + np.array([nf, 0]) * unit_size
    #_elem = []
    #for fidx in range(nf):
    #    g_xy = base_params['G']['xy'] + np.array([fidx, 0]) * unit_size
    #    g_layer = base_params['G']['layer']
    #    _elem.append(laygo2.object.Pin(xy=g_xy, layer=g_layer, netname='G'))
    #pins['G'] = laygo2.object.Pin(xy=base_params['G']['xy'], layer=g_layer, netname='G', elements=np.array(_elem))
    return pins


def pmos_pins_func(params):
    return mos_pins_func(devtype='pmos', params=params)


def nmos_pins_func(params):
    return mos_pins_func(devtype='nmos', params=params)


def mos_generate_func(devtype, name=None, shape=None, pitch=np.array([0, 0]), transform='R0', params=None):
    """Generates an instance from the input parameters."""
    # Compute parameters
    nf = params['nf'] if 'nf' in params.keys() else 1
    unit_size = np.array(tech_params['templates']['mos'][devtype]['unit_size'])

    # Create the base mosfet structure.
    nelements = dict()
    for rn, rpar in tech_params['templates']['mos'][devtype]['rects'].items():
        # Computing loop index
        idx_start = 0 if 'idx_start' not in rpar else rpar['idx_start']
        idx_end = 0 if 'idx_end' not in rpar else rpar['idx_end']
        idx_step = 1 if 'idx_step' not in rpar else rpar['idx_step']
        for i in range(idx_start, nf+idx_end, idx_step):
            rxy = rpar['xy'] + np.array([i, 0]) * unit_size
            nelements[rn+str(i)] = laygo2.object.Rect(xy=rxy, layer=rpar['layer'], name=rn+str(i))
    #if 'rects_merged' in tech_params['templates']['mos'][devtype]:
    #    for rn, rpar in tech_params['templates']['mos'][devtype]['rects_merged'].items():
    #        rxy = np.array([[rpar['xy'][0][0], rpar['xy'][0][1]], [rpar['xy'][1][0] + unit_size[0] * (nf - 1), rpar['xy'][1][1]]])
    #        nelements[rn] = laygo2.object.Rect(xy=rxy, layer=rpar['layer'], name=rn)

    # Create pins
    pins = mos_pins_func(devtype=devtype, params=params)
    nelements.update(pins)  # Add physical pin structures to the virtual object.

    inst_unit_size = unit_size * np.array([nf, 1])
    # Generate and return the final instance
    inst = laygo2.object.VirtualInstance(name=name, xy=np.array([0, 0]), libname='mylib', cellname='myvcell_'+devtype,
                                         native_elements=nelements, shape=shape, pitch=pitch,
                                         transform=transform, unit_size=inst_unit_size, pins=pins)
    return inst


def pmos_generate_func(name=None, shape=None, pitch=np.array([0, 0]), transform='R0', params=None):
    return mos_generate_func(devtype='pmos', name=name, shape=shape, pitch=pitch, transform=transform, params=params)


def nmos_generate_func(name=None, shape=None, pitch=np.array([0, 0]), transform='R0', params=None):
    return mos_generate_func(devtype='nmos', name=name, shape=shape, pitch=pitch, transform=transform, params=params)


# Routing vias
def via_generate_func(devtype, name=None, shape=None, pitch=np.array([0, 0]), transform='R0', params=None):
    """Generates an instance from the input parameters."""
    # Create the via structure.
    nelements = dict()
    for rn, rpar in tech_params['templates']['via'][devtype]['rects'].items():
        nelements[rn] = laygo2.object.Rect(xy=rpar['xy'], layer=rpar['layer'], name=rn)

    # Generate and return the final instance
    inst = laygo2.object.VirtualInstance(name=name, xy=np.array([0, 0]), libname='mylib', cellname='myvcell_devtype',
                                         native_elements=nelements, shape=shape, pitch=pitch,
                                         transform=transform, unit_size=np.array([0, 0]), pins=None)
    return inst


def via_r12_default_generate_func(name=None, shape=None, pitch=np.array([0, 0]), transform='R0', params=None):
    return via_generate_func(devtype='via_r12_default', name=name, shape=shape, pitch=pitch, transform=transform,
                             params=params)


def via_r12_bottomplug_generate_func(name=None, shape=None, pitch=np.array([0, 0]), transform='R0', params=None):
    return via_generate_func(devtype='via_r12_bottomplug', name=name, shape=shape, pitch=pitch, transform=transform,
                             params=params)


def via_r12_topplug_generate_func(name=None, shape=None, pitch=np.array([0, 0]), transform='R0', params=None):
    return via_generate_func(devtype='via_r12_topplug', name=name, shape=shape, pitch=pitch, transform=transform,
                             params=params)


def via_r23_default_generate_func(name=None, shape=None, pitch=np.array([0, 0]), transform='R0', params=None):
    return via_generate_func(devtype='via_r23_default', name=name, shape=shape, pitch=pitch, transform=transform,
                             params=params)


# Template library
def load_templates():
    """Load template to a template library object"""
    tlib = laygo2.object.database.TemplateLibrary(name='advtech_templates')
    # Transistors
    # Transistor layouts are created in laygo and stored as a virtual instance.
    tnmos = laygo2.object.template.UserDefinedTemplate(name='nmos', bbox_func=nmos_bbox_func, pins_func=nmos_pins_func,
                                                       generate_func=nmos_generate_func)
    tpmos = laygo2.object.template.UserDefinedTemplate(name='pmos', bbox_func=pmos_bbox_func, pins_func=pmos_pins_func,
                                                       generate_func=pmos_generate_func)
    tlib.append([tpmos, tnmos])
    # Vias
    # Via layouts are created in laygo and stored as a virtual instance.
    # tvia_r12_0 = laygo2.object.template.NativeInstanceTemplate(libname='advtech_templates', cellname='via_r12_0')
    # tlib.append([tvia_r12_0])
    tvia_r12_default = laygo2.object.template.UserDefinedTemplate(name='via_r12_default', bbox_func=lambda params: np.array([0, 0]),
                                                                  pins_func=lambda params: None,
                                                                  generate_func=via_r12_default_generate_func)
    tlib.append([tvia_r12_default])
    tvia_r12_topplug = laygo2.object.template.UserDefinedTemplate(name='via_r12_topplug', bbox_func=lambda params: np.array([0, 0]),
                                                                  pins_func=lambda params: None,
                                                                  generate_func=via_r12_topplug_generate_func)
    tlib.append([tvia_r12_topplug])
    tvia_r12_bottomplug = laygo2.object.template.UserDefinedTemplate(name='via_r12_bottomplug', bbox_func=lambda params: np.array([0, 0]),
                                                                     pins_func=lambda params: None,
                                                                     generate_func=via_r12_bottomplug_generate_func)
    tlib.append([tvia_r12_bottomplug])
    tvia_r23_default = laygo2.object.template.UserDefinedTemplate(name='via_r23_default', bbox_func=lambda params: np.array([0, 0]),
                                                                  pins_func=lambda params: None,
                                                                  generate_func=via_r23_default_generate_func)
    tlib.append([tvia_r23_default])
    return tlib


# Tests
if __name__ == '__main__':
    # Create templates.
    print("Create templates")
    templates = load_templates()
    print(templates['nmos'])
    print(templates['pmos'])
