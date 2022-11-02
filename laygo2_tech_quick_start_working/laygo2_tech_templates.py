# -*- coding: utf-8 -*-
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

"""Template library for the advanced example technology. (advtech)"""

import numpy as np

import laygo2.object.template
import laygo2.object.physical
import laygo2.object.database

# Should be `import laygo2_tech as tech` for actual use.
# import laygo2_tech as tech
import laygo2_tech_quick_start as tech

tech_params = tech.tech_params
templates = tech_params["templates"]
grids = tech_params["grids"]

# Template functions for primitive devices
def _mos_update_params(params):
    """Make a complete parameter table for mos"""
    unit_size = tech_params["templates"]["mos"]["common"]["unit_size"]
    unit_size_half = np.array([unit_size[0]/2, unit_size[1]])
    if "nf" not in params:  # number of fingers
        params["nf"] = 1
    if "nfdmyl" not in params:  # number of left-dummy fingers
        params["nfdmyl"] = 0
    if "nfdmyr" not in params:  # number of right-dummy fingers
        params["nfdmyr"] = 0
    if "trackswap" not in params:  # source-drain track swap
        params["trackswap"] = False
    if "tie" not in params:  # tie to power rail
        params["tie"] = None
    if "bndl" not in params:  # create local left boundary
        params["bndl"] = True
    if "bndr" not in params:  # create local right boundary
        params["bndr"] = True
    if "gbndl" not in params:  # create global left boundary
        params["gbndl"] = False
    if "gbndr" not in params:  # create global right boundary
        params["gbndr"] = False
    if "unit_size_core" not in params:  # core unit size
        params["unit_size_core"] = unit_size
    if "unit_size_dmy" not in params:  # dummy size
        params["unit_size_dmy"] = unit_size
    if "unit_size_bndl" not in params:  # left boundary unit size
        params["unit_size_bndl"] = unit_size_half
    if "unit_size_bndr" not in params:  # right boundary unit size
        params["unit_size_bndr"] = unit_size_half
    if "unit_size_gbndl" not in params:  # left boundary unit size
        params["unit_size_gbndl"] = unit_size_half
    if "unit_size_gbndr" not in params:  # right boundary unit size
        params["unit_size_gbndr"] = unit_size_half
    return params


def mos_bbox_func(params):
    """Computes x and y coordinate values from params."""
    params = _mos_update_params(params)
    unit_size = params["unit_size_core"]
    xy0 = [0, 0]
    xy1 = [unit_size[0] * params["nf"], unit_size[1]]
    xy = np.array([xy0, xy1])
    # Increment bbox based on boundary / dummy parameters.
    if params["gbndl"]:
        xy[1, 0] += params["unit_size_gbndl"][0]
    if params["bndl"]:
        xy[1, 0] += params["unit_size_bndl"][0]
    if params["nfdmyl"] > 0:
        xy[1, 0] += params["unit_size_dmy"][0] * round(params["nfdmyl"])
    if params["nfdmyr"] > 0:
        xy[1, 0] += params["unit_size_dmy"][0] * round(params["nfdmyr"])
    if params["bndr"]:
        xy[1, 0] += params["unit_size_bndr"][0]
    if params["gbndr"]:
        xy[1, 0] += params["unit_size_gbndr"][0]
    return xy


def _mos_route(devtype, params, offset=[0, 0]):
    """internal function to create routing structure of mosfets"""
    params = _mos_update_params(params)

    # Routing offsets
    yoffset = offset[1]
    offset = np.array([0, yoffset])
    offset_rail = np.array([0, 0])
    offset_dmyl = np.array([0, yoffset])
    offset_dmyr = np.array([0, yoffset])
    if params["gbndl"]:
        offset[0] += params["unit_size_gbndl"][0]
    offset_rail[0] = offset[0]
    if params["bndl"]:
        offset[0] += params["unit_size_bndl"][0]
        offset_rail[0] -= params["unit_size_bndl"][0]
    offset_dmyl[0] = offset[0]
    offset[0] += params["unit_size_dmy"][0] * round(params["nfdmyl"])
    offset_dmyr[0] = offset[0] + params["unit_size_core"][0] * round(params["nf"])
    nelements = dict()
    # Base parameters
    ref_temp_name = "nmos"  # template for param calculations
    name_list = ["G", "S", "D"]
    if params["trackswap"]:  # source-drain track swap
        yidx_list = [3, 2, 1]  # y-track list
    else:
        yidx_list = [3, 1, 2]
        #pin_name_list = ["G", "S", "D"]  # pin name list to connect
    for _name, _yidx in zip(name_list, yidx_list):
        if params["tie"] == _name:
            continue  # do not generate routing elements
        # compute routing cooridnates
        x0 = templates['mos'][ref_temp_name]["pins"][_name]["xy"][0][0]
        x1 = templates['mos'][ref_temp_name]["pins"][_name]["xy"][1][0]
        if _name == "G":
            x1 += params["unit_size_core"][0] * (params["nf"] - 1)
        if _name == "S":
            x1 += params["unit_size_core"][0] * (2 * round((params["nf"]-1)/2))
        if _name == "D":
            x1 += params["unit_size_core"][0] * (2 * (round(params["nf"]/2)-1))
        y = grids["routing_12_cmos"]["horizontal"]["elements"][_yidx] + offset[1]
        vextension = round(grids["routing_12_cmos"]["horizontal"]["width"][_yidx]/2)
        hextension = 0
        if x0 == x1:  # zero-size wire
            hextension = grids["routing_12_cmos"]["vertical"]["extension0"][0]
            if _name == "G" and params["nf"] == 2:  # extend G route when nf is 2 to avoid DRC errror
                hextension = hextension + 55
            elif (
                _name == "D" and params["nf"] == 2 and params["tie"] != "D"
            ):  # extend D route when nf is 2 and not tied with D
                hextension = hextension + 55
            elif (
                _name == "S" and params["nf"] == 2 and params["tie"] != "S"
            ):  # extend S route when nf is 2 and not tied with S
                hextension = hextension + 55
            else:
                hextension = grids["routing_12_cmos"]["vertical"]["extension"][0] + 25
        rxy = [[x0, y], [x1, y]]
        rlayer = grids["routing_12_cmos"]["horizontal"]["layer"][_yidx]
        # horizontal metal routing
        color = "not MPT"
        rg = laygo2.object.Rect(
            xy=rxy, layer=rlayer, name="R" + _name + "0", hextension=hextension, vextension=vextension, color=color
        )
        nelements["R" + _name + "0"] = rg
        # via
        x0 = templates['mos'][ref_temp_name]["pins"][_name]["xy"][0][0]
        x1 = templates['mos'][ref_temp_name]["pins"][_name]["xy"][1][0]
        x = (x0+x1)/2
        if _name == "G":
            idx = params["nf"]
        if _name == "S":
            idx = int(params["nf"]/2) + 1
        if _name == "D":
            idx = int((params["nf"]-1)/2) + 1
        if _name == "G":
            ivia_pitch=params["unit_size_core"]
        else:
            ivia_pitch=np.array([params["unit_size_core"][0]*2, params["unit_size_core"][1]])
        for ivia_idx in range(0, idx):
            for rn, rpar in tech_params["templates"]["via"]["via_r12_default"]["rects"].items():
                #ivia_nelements[rn] = laygo2.object.Rect(xy=rpar["xy"], layer=rpar["layer"], name=rn)
                _xy = np.array([x + ivia_pitch[0]*ivia_idx, y]) + rpar["xy"]
                nelements["IV" + _name + rn + str(ivia_idx)] = laygo2.object.Rect(xy=_xy, layer=rpar["layer"], name=rn)
    # Horizontal rail
    x0 = templates['mos'][ref_temp_name]["pins"]["S"]["xy"][0][0]
    x1 = templates['mos'][ref_temp_name]["pins"]["S"]["xy"][1][0]
    x = round((x0 + x1) / 2) + offset_rail[0]  # center coordinate
    x0 = x
    x1 = x + params["unit_size_core"][0] * round(params["nf"])
    x1 += params["unit_size_dmy"][0] * round(params["nfdmyl"])
    x1 += params["unit_size_dmy"][0] * round(params["nfdmyr"])
    if params["bndl"]:
        x1 += params["unit_size_bndl"][0]
    if params["bndr"]:
        x1 += params["unit_size_bndr"][0]
    y = grids["routing_12_cmos"]["horizontal"]["elements"][0] + offset_rail[1]
    vextension = round(grids["routing_12_cmos"]["horizontal"]["width"][0] / 2)
    hextension = grids["routing_12_cmos"]["vertical"]["extension"][0]
    rxy = [[x0, y], [x1, y]]
    rlayer = grids["routing_12_cmos"]["horizontal"]["layer"][0]
    color = "not MPT"
    # metal routing
    rg = laygo2.object.Rect(
        xy=rxy, layer=rlayer, name="RRAIL0", hextension=hextension, vextension=vextension, color=color
    )
    nelements["RRAIL0"] = rg
    '''
    # Tie to rail
    if params["tie"] is not None:
        # routing
        if params["tie"] == "D":
            idx = round(params["nf"] / 2)
            _pin_name = "D0"
        if params["tie"] == "S":
            idx = round(params["nf"] / 2) + 1
            _pin_name = "S0"
        if params["tie"] == "TAP0":
            idx = round(params["nf"] / 2) + 1
            _pin_name = "TAP0"
        if params["tie"] == "TAP1":
            idx = round(params["nf"] / 2)
            _pin_name = "TAP1"
        x0 = templates[ref_temp_name]["pins"][_pin_name]["xy"][0][0]
        x1 = templates[ref_temp_name]["pins"][_pin_name]["xy"][1][0]
        x = round((x0 + x1) / 2) + offset[0]  # center coordinate
        _x = x
        for i in range(idx):
            hextension = round(grids["routing_12_cmos"]["vertical"]["width"][0] / 2)
            vextension = grids["routing_12_cmos"]["horizontal"]["extension"][0]
            y0 = grids["routing_12_cmos"]["horizontal"]["elements"][0] + offset_rail[1]
            y1 = grids["routing_12_cmos"]["horizontal"]["elements"][1] + offset[1]
            rxy = [[_x, y0], [_x, y1]]
            rlayer = grids["routing_12_cmos"]["vertical"]["layer"][0]
            color = grids["routing_12_cmos"]["vertical"]["xcolor"][0]
            rg = laygo2.object.Rect(
                xy=rxy, layer=rlayer, name="RTIE" + str(i), hextension=hextension, vextension=vextension, color=color
            )
            nelements["RTIE" + str(i)] = rg
            _x += params["unit_size_core"][0]
        # via
        vname = grids["routing_12_cmos"]["via"]["map"][0][0]
        ivia = laygo2.object.Instance(
            name="IVTIE0",
            xy=[x, y0],
            libname=libname,
            cellname=vname,
            shape=[idx, 1],
            pitch=params["unit_size_core"],
            unit_size=[0, 0],
            pins=None,
            transform="R0",
        )
        nelements["IVTIE" + _name + "0"] = ivia
    # Tie to rail - dummy left
    if params["nfdmyl"] > 0:
        if devtype == "nmos" or devtype == "pmos":
            if params["bndl"]:  # terminated by boundary
                pin_name = "S0"
                idx_offset = 0
            else:
                pin_name = "G0"
                idx_offset = -1
        elif devtype == "ptap" or devtype == "ntap":
            if params["bndl"]:  # terminated by boundary
                pin_name = "TAP0"
                idx_offset = 0
            else:
                pin_name = "TAP1"
                idx_offset = -1
        x0 = templates[ref_dmy_temp_name]["pins"][pin_name]["xy"][0][0]
        x1 = templates[ref_dmy_temp_name]["pins"][pin_name]["xy"][1][0]
        x = round((x0 + x1) / 2) + offset_dmyl[0]  # center coordinate
        _x = x
        idx = round(params["nfdmyl"]) + idx_offset
        for i in range(idx):
            hextension = round(grids["routing_12_cmos"]["vertical"]["width"][0] / 2)
            vextension = grids["routing_12_cmos"]["horizontal"]["extension"][0]
            y0 = grids["routing_12_cmos"]["horizontal"]["elements"][0] + offset_rail[1]
            y1 = grids["routing_12_cmos"]["horizontal"]["elements"][1] + offset_dmyl[1]
            rxy = [[_x, y0], [_x, y1]]
            rlayer = grids["routing_12_cmos"]["vertical"]["layer"][0]
            color = grids["routing_12_cmos"]["vertical"]["xcolor"][0]
            rg = laygo2.object.Rect(
                xy=rxy,
                layer=rlayer,
                name="RTIEDMYL" + str(i),
                hextension=hextension,
                vextension=vextension,
                color=color,
            )
            nelements["RTIEDMYL" + str(i)] = rg
            _x = _x + round(params["unit_size_dmy"][0] / 2)
        # via
        vname = grids["routing_12_cmos"]["via"]["map"][0][0]
        ivia = laygo2.object.Instance(
            name="IVTIEDMYL0",
            xy=[x, y0],
            libname=libname,
            cellname=vname,
            shape=[idx, 1],
            pitch=params["unit_size_dmy"] * np.array([0.5, 1]),
            unit_size=[0, 0],
            pins=None,
            transform="R0",
        )
        nelements["IVTIEDMYL" + _name + "0"] = ivia
    # Tie to rail - dummy right
    if params["nfdmyr"] > 0:
        if devtype == "nmos" or devtype == "pmos":
            if params["bndr"]:  # terminated by boundary
                pin_name = "G0"
                idx_offset = 0
            else:
                pin_name = "G0"
                idx_offset = -1
        elif devtype == "ptap" or devtype == "ntap":
            if params["bndr"]:  # terminated by boundary
                pin_name = "TAP1"
                idx_offset = 0
            else:
                pin_name = "TAP1"
                idx_offset = -1
        x0 = templates[ref_dmy_temp_name]["pins"][pin_name]["xy"][0][0]
        x1 = templates[ref_dmy_temp_name]["pins"][pin_name]["xy"][1][0]
        x = round((x0 + x1) / 2) + offset_dmyr[0]  # center coordinate
        _x = x
        idx = round(params["nfdmyr"]) + idx_offset
        for i in range(idx):
            hextension = round(grids["routing_12_cmos"]["vertical"]["width"][0] / 2)
            vextension = grids["routing_12_cmos"]["horizontal"]["extension"][0]
            y0 = grids["routing_12_cmos"]["horizontal"]["elements"][0] + offset_rail[1]
            y1 = grids["routing_12_cmos"]["horizontal"]["elements"][1] + offset_dmyr[1]
            rxy = [[_x, y0], [_x, y1]]
            rlayer = grids["routing_12_cmos"]["vertical"]["layer"][0]
            color = grids["routing_12_cmos"]["vertical"]["xcolor"][0]
            rg = laygo2.object.Rect(
                xy=rxy,
                layer=rlayer,
                name="RTIEDMYR" + str(i),
                hextension=hextension,
                vextension=vextension,
                color=color,
            )
        '''
    return nelements


def mos_pins_func(devtype, params):
    """Generate a pin dictionary from params."""
    unit_size = tech_params["templates"]["mos"][devtype]["unit_size"]
    base_params = tech_params["templates"]["mos"][devtype]["pins"]
    nf = params["nf"]
    sd_swap = params["sd_swap"] if "sd_swap" in params.keys() else False
    pins = dict()
    lpin_default = "S"
    rpin_default = "D"
    if sd_swap:
        lpin = rpin_default
        rpin = lpin_default
    else:
        lpin = lpin_default
        rpin = rpin_default
    # pins starting from the left-most diffusion. Source if sd_swap=False, else Drain.
    _elem = []
    for fidx in range(nf // 2 + 1):
        l_xy = base_params[lpin_default]["xy"] + np.array([fidx, 0]) * unit_size * 2
        l_layer = base_params[lpin_default]["layer"]
        _elem.append(laygo2.object.Pin(xy=l_xy, layer=l_layer, netname=lpin))
    pins[lpin] = laygo2.object.Pin(
        xy=base_params[lpin_default]["xy"], layer=l_layer, netname=lpin, elements=np.array(_elem)
    )
    # pins starting from the the other side of diffusion. Drain if sd_swap=False, else Source.
    _elem = []
    for fidx in range(((nf + 1) // 2)):
        r_xy = base_params[rpin_default]["xy"] + np.array([fidx, 0]) * unit_size * 2
        r_layer = base_params[rpin_default]["layer"]
        _elem.append(laygo2.object.Pin(xy=r_xy, layer=r_layer, netname=rpin))
    pins[rpin] = laygo2.object.Pin(
        xy=base_params[rpin_default]["xy"], layer=r_layer, netname=rpin, elements=np.array(_elem)
    )
    # gate
    g_xy = np.array([base_params["G"]["xy"][0], base_params["G"]["xy"][1] + np.array([nf - 1, 0]) * unit_size])
    g_layer = base_params["G"]["layer"]
    pins["G"] = laygo2.object.Pin(xy=g_xy, layer=g_layer, netname="G")
    # g_xy = base_params['G']['xy'] + np.array([nf, 0]) * unit_size
    # _elem = []
    # for fidx in range(nf):
    #    g_xy = base_params['G']['xy'] + np.array([fidx, 0]) * unit_size
    #    g_layer = base_params['G']['layer']
    #    _elem.append(laygo2.object.Pin(xy=g_xy, layer=g_layer, netname='G'))
    # pins['G'] = laygo2.object.Pin(xy=base_params['G']['xy'], layer=g_layer, netname='G', elements=np.array(_elem))
    return pins


def mos_generate_func(devtype, name=None, shape=None, pitch=np.array([0, 0]), transform="R0", params=None):
    """Generates an instance from the input parameters."""
    # Compute parameters
    nf = params["nf"] if "nf" in params.keys() else 1
    unit_size = np.array(tech_params["templates"]["mos"][devtype]["unit_size"])

    # Create the base mosfet structure.
    nelements = dict()
    for rn, rpar in tech_params["templates"]["mos"][devtype]["rects"].items():
        # Computing loop index
        idx_start = 0 if "idx_start" not in rpar else rpar["idx_start"]
        idx_end = 0 if "idx_end" not in rpar else rpar["idx_end"]
        idx_step = 1 if "idx_step" not in rpar else rpar["idx_step"]
        for i in range(idx_start, nf + idx_end, idx_step):
            rxy = rpar["xy"] + np.array([i, 0]) * unit_size
            nelements[rn + str(i)] = laygo2.object.Rect(xy=rxy, layer=rpar["layer"], name=rn + str(i))
    
    # Create routing structures 
    nelements.update(_mos_route(devtype=devtype, params=params))

    # Create pins
    # pins = mos_pins_func(devtype=devtype, params=params)
    # nelements.update(pins)  # Add physical pin structures to the virtual object.

    #inst_unit_size = unit_size * np.array([nf, 1])
    # Unit size
    inst_xy = mos_bbox_func(params=params)
    inst_unit_size = [inst_xy[1, 0] - inst_xy[0, 0], inst_xy[1, 1] - inst_xy[0, 1]]
    
    # Generate and return the final instance
    inst = laygo2.object.VirtualInstance(
        name=name,
        xy=np.array([0, 0]),
        libname="mylib",
        cellname="myvcell_" + devtype,
        native_elements=nelements,
        shape=shape,
        pitch=pitch,
        transform=transform,
        unit_size=inst_unit_size,
        #pins=pins,
    )
    return inst


def pmos_generate_func(name=None, shape=None, pitch=np.array([0, 0]), transform="R0", params=None):
    return mos_generate_func(devtype="pmos", name=name, shape=shape, pitch=pitch, transform=transform, params=params)


def nmos_generate_func(name=None, shape=None, pitch=np.array([0, 0]), transform="R0", params=None):
    return mos_generate_func(devtype="nmos", name=name, shape=shape, pitch=pitch, transform=transform, params=params)


# Routing vias
def via_generate_func(devtype, name=None, shape=None, pitch=np.array([0, 0]), transform="R0", params=None):
    """Generates an instance from the input parameters."""
    # Create the via structure.
    nelements = dict()
    for rn, rpar in tech_params["templates"]["via"][devtype]["rects"].items():
        nelements[rn] = laygo2.object.Rect(xy=rpar["xy"], layer=rpar["layer"], name=rn)

    # Generate and return the final instance
    inst = laygo2.object.VirtualInstance(
        name=name,
        xy=np.array([0, 0]),
        libname="mylib",
        cellname="myvcell_devtype",
        native_elements=nelements,
        shape=shape,
        pitch=pitch,
        transform=transform,
        unit_size=np.array([0, 0]),
        pins=None,
    )
    return inst


def via_r12_default_generate_func(name=None, shape=None, pitch=np.array([0, 0]), transform="R0", params=None):
    return via_generate_func(
        devtype="via_r12_default", name=name, shape=shape, pitch=pitch, transform=transform, params=params
    )


def via_r12_bottomplug_generate_func(name=None, shape=None, pitch=np.array([0, 0]), transform="R0", params=None):
    return via_generate_func(
        devtype="via_r12_bottomplug", name=name, shape=shape, pitch=pitch, transform=transform, params=params
    )


def via_r12_topplug_generate_func(name=None, shape=None, pitch=np.array([0, 0]), transform="R0", params=None):
    return via_generate_func(
        devtype="via_r12_topplug", name=name, shape=shape, pitch=pitch, transform=transform, params=params
    )


def via_r23_default_generate_func(name=None, shape=None, pitch=np.array([0, 0]), transform="R0", params=None):
    return via_generate_func(
        devtype="via_r23_default", name=name, shape=shape, pitch=pitch, transform=transform, params=params
    )


# Template library
def load_templates():
    """Load template to a template library object"""
    tlib = laygo2.object.database.TemplateLibrary(name="advtech_templates")
    # Transistors
    # Transistor layouts are created in laygo and stored as a virtual instance.
    tnmos = laygo2.object.template.UserDefinedTemplate(
        name="nmos", bbox_func=mos_bbox_func, pins_func=mos_pins_func, generate_func=nmos_generate_func
    )
    tpmos = laygo2.object.template.UserDefinedTemplate(
        name="pmos", bbox_func=mos_bbox_func, pins_func=mos_pins_func, generate_func=pmos_generate_func
    )
    tlib.append([tpmos, tnmos])
    # Vias
    # Via layouts are created in laygo and stored as a virtual instance.
    # tvia_r12_0 = laygo2.object.template.NativeInstanceTemplate(libname='advtech_templates', cellname='via_r12_0')
    # tlib.append([tvia_r12_0])
    tvia_r12_default = laygo2.object.template.UserDefinedTemplate(
        name="via_r12_default",
        bbox_func=lambda params: np.array([0, 0]),
        pins_func=lambda params: None,
        generate_func=via_r12_default_generate_func,
    )
    tlib.append([tvia_r12_default])
    tvia_r12_topplug = laygo2.object.template.UserDefinedTemplate(
        name="via_r12_topplug",
        bbox_func=lambda params: np.array([0, 0]),
        pins_func=lambda params: None,
        generate_func=via_r12_topplug_generate_func,
    )
    tlib.append([tvia_r12_topplug])
    tvia_r12_bottomplug = laygo2.object.template.UserDefinedTemplate(
        name="via_r12_bottomplug",
        bbox_func=lambda params: np.array([0, 0]),
        pins_func=lambda params: None,
        generate_func=via_r12_bottomplug_generate_func,
    )
    tlib.append([tvia_r12_bottomplug])
    tvia_r23_default = laygo2.object.template.UserDefinedTemplate(
        name="via_r23_default",
        bbox_func=lambda params: np.array([0, 0]),
        pins_func=lambda params: None,
        generate_func=via_r23_default_generate_func,
    )
    tlib.append([tvia_r23_default])
    return tlib


# Tests
if __name__ == "__main__":
    # Create templates.
    print("Create templates")
    templates = load_templates()
    print(templates["nmos"])
    print(templates["pmos"])
