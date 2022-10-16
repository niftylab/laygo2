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
import pprint
import laygo2
import laygo2.interface
import laygo2_tech as tech

# Parameter definitions ##############
# Templates
tpmos_name = 'pmos'
tnmos_name = 'nmos'
tntap_name = 'ntap'
tptap_name = 'ptap'
# Grids
pg_name = 'placement_basic'
r12_name = 'routing_12_cmos_flipped'
r23_name = 'routing_23_cmos_flipped'
r34_name = 'routing_34_basic'
# Design hierarchy
libname = 'laygo2_test'
cellname = 'salatch_tapopt2'
# Design parameters
#nf_tap = 4  # tap
nf_csh = 4  # current source
nf_in = 8   # input
nf_rgn = 6  # regeneration - n
nf_rgp = 4  # regeneration - p
nf_rs = 2  # reset
nf_max = max(nf_csh, nf_in, nf_rgn, nf_rgp + nf_rs) + 2  # insert additional 2 dummies
nf_csh_dmy = max(nf_max - nf_csh, 0)
nf_in_dmy = max(nf_max - nf_in, 0)
nf_rgn_dmy = max(nf_max - nf_rgn, 0)
nf_rgp_dmy = max(nf_max - nf_rgp - nf_rs, 0)
# End of parameter definitions #######

# Generation start ###################
# 1. Load templates and grids.
print("Load templates")
templates = tech.load_templates()
tpmos, tnmos = templates[tpmos_name], templates[tnmos_name]
tntap, tptap = templates[tntap_name], templates[tptap_name]
#print(templates[tpmos_name], templates[tnmos_name], sep="\n")

print("Load grids")
grids = tech.load_grids(templates=templates)
pg, r12, r23, r34= grids[pg_name], grids[r12_name], grids[r23_name], grids[r34_name]
#print(grids[pg_name], grids[r12_name], grids[r23_name], sep="\n")

# 2. Create a design hierarchy.
lib = laygo2.object.database.Library(name=libname)
dsn = laygo2.object.database.Design(name=cellname, libname=libname)
lib.append(dsn)

# 3. Create instances.
print("Create instances")
iptapp = tptap.generate(name='IPTAPP', params={'nf': nf_max, 'tie': 'TAP0'}, transform='MX')
iptapn = tptap.generate(name='IPTAPN', params={'nf': nf_max, 'tie': 'TAP0'}, transform='MX')
incsp = tnmos.generate(name='MNCSP', params={'nf': nf_csh, 'nfdmyl': nf_csh_dmy, 'nfdmyr':2, 'bndr': None, 'tie': 'S'})
incsn = tnmos.generate(name='MNCSN', params={'nf': nf_csh, 'nfdmyr': nf_csh_dmy, 'bndl': None, 'tie': 'S'})
ininp = tnmos.generate(name='MNINP', params={'nf': nf_in, 'nfdmyl': nf_in_dmy, 'nfdmyr':2, 'bndr': None, 'trackswap': True}, transform='MX')
ininn = tnmos.generate(name='MNINN', params={'nf': nf_in, 'nfdmyr': nf_in_dmy, 'bndl': None, 'trackswap': True}, transform='MX')
inrgp = tnmos.generate(name='MNRGP', params={'nf': nf_rgn, 'nfdmyl': nf_rgn_dmy, 'nfdmyr':2, 'bndr': None})
inrgn = tnmos.generate(name='MNRGN', params={'nf': nf_rgn, 'nfdmyr': nf_rgn_dmy, 'bndl': None})
iprgp = tpmos.generate(name='MPRGP', params={'nf': nf_rgp, 'bndr': False, 'nfdmyl': nf_rgp_dmy, 'tie':'S'}, transform='MX')
iprsp = tpmos.generate(name='MPRSP', params={'nf': nf_rs, 'bndl': False, 'tie':'S', 'nfdmyr':2, 'bndr': None}, transform='MX')
iprsn = tpmos.generate(name='MPRSN', params={'nf': nf_rs, 'bndr': False, 'tie':'S', 'bndl': None}, transform='MX')
iprgn = tpmos.generate(name='MPRGN', params={'nf': nf_rgp, 'bndl': False, 'nfdmyr': nf_rgp_dmy, 'tie':'S'}, transform='MX')
intapp = tntap.generate(name='INTAPP', params={'nf': nf_max, 'tie': 'TAP0'})
intapn = tntap.generate(name='INTAPN', params={'nf': nf_max, 'tie': 'TAP0'})

# 4. Place instances.
dsn.place(grid=pg,  inst=iptapp,    mn=pg.mn.height_vec(iptapp))
dsn.place(grid=pg,  inst=iptapn,    mn=pg.mn.top_right(iptapp))
dsn.place(grid=pg,  inst=incsp,     mn=pg.mn.top_left(iptapp))
dsn.place(grid=pg,  inst=incsn,     mn=pg.mn.bottom_right(incsp))
dsn.place(grid=pg,  inst=ininp,     mn=pg.mn.top_left(incsp) + pg.mn.height_vec(ininp))
dsn.place(grid=pg,  inst=ininn,     mn=pg.mn.top_right(ininp))
dsn.place(grid=pg,  inst=inrgp,     mn=pg.mn.top_left(ininp))
dsn.place(grid=pg,  inst=inrgn,     mn=pg.mn.bottom_right(inrgp))
dsn.place(grid=pg,  inst=iprgp,     mn=pg.mn.top_left(inrgp) + pg.mn.height_vec(iprgp))
dsn.place(grid=pg,  inst=iprsp,     mn=pg.mn.top_right(iprgp))
dsn.place(grid=pg,  inst=iprsn,     mn=pg.mn.top_right(iprsp))
dsn.place(grid=pg,  inst=iprgn,     mn=pg.mn.top_right(iprsn))
dsn.place(grid=pg,  inst=intapp,    mn=pg.mn.top_left(iprgp))
dsn.place(grid=pg,  inst=intapn,    mn=pg.mn.bottom_right(intapp))

# 5. Create and place wires.
print("Create wires")
# CS - tail
_mn = [r23.mn(incsp.pins['D'])[0], r23.mn(incsn.pins['D'])[1]]
rcs0 = dsn.route(grid=r23, mn=_mn)
# CS - clk
_mn = [r23.mn(incsp.pins['G'])[0], r23.mn(incsn.pins['G'])[1]]
rcs1 = dsn.route(grid=r23, mn=_mn)
# IN - tail
_mn = [r23.mn(ininp.pins['S'])[0], r23.mn(ininn.pins['S'])[1]]
rin1 = dsn.route(grid=r23, mn=_mn)
# CS-IN - tail
trk_num = round(min(nf_csh, nf_in)/2)
trk_ref = [r23.mn(incsp.pins['D'])[1][0], r23.mn(incsn.pins['D'])[0][0]]
_mn0 = [r23.mn(incsp.pins['D'])[0], r23.mn(ininp.pins['S'])[0]]
_mn1 = [r23.mn(incsn.pins['D'])[0], r23.mn(ininn.pins['S'])[0]]
for i in range(trk_num):
    _, v0, r0, v1, _= dsn.route_via_track(grid=r23, mn=_mn0, track=[trk_ref[0]-2*i, None])
    _, v0, r0, v1, _= dsn.route_via_track(grid=r23, mn=_mn1, track=[trk_ref[1]+2*i, None])
# IN-RGN
trk_num = round(min(nf_in, nf_rgn)/2) - 1  # -1 due to routing offset
trk_ref = [r23.mn(ininp.pins['D'])[1][0], r23.mn(ininn.pins['D'])[0][0]]
_mn0 = [r23.mn(ininp.pins['D'])[0], r23.mn(inrgp.pins['S'])[0]]
_mn1 = [r23.mn(ininn.pins['D'])[0], r23.mn(inrgn.pins['S'])[0]]
for i in range(trk_num):
    _, v0, r0, v1, _= dsn.route_via_track(grid=r23, mn=_mn0, track=[trk_ref[0]-2*i-1, None])
    _, v0, r0, v1, _= dsn.route_via_track(grid=r23, mn=_mn1, track=[trk_ref[1]+2*i+1, None])
# RGN-RGP - out
trk_num = round(min(nf_rgn, nf_rgp)/2)
trk_ref = [r23.mn(inrgp.pins['D'])[1][0], r23.mn(inrgn.pins['D'])[0][0]]
_mn0 = [r23.mn(inrgp.pins['D'])[0], r23.mn(iprgp.pins['D'])[0]]
_mn1 = [r23.mn(inrgn.pins['D'])[0], r23.mn(iprgn.pins['D'])[0]]
rgp_out = []
rgn_out = []
for i in range(trk_num):
    _, v0, r0, v1, _= dsn.route_via_track(grid=r23, mn=_mn0, track=[trk_ref[0]-2*i, None])
    rgp_out.append(r0)
    _, v0, r0, v1, _= dsn.route_via_track(grid=r23, mn=_mn1, track=[trk_ref[1]+2*i, None])
    rgn_out.append(r0)
# RGN-RGP - in
trk_num = round(min(nf_rgn, nf_rgp)/2) - 1
trk_ref = [r23.mn(inrgp.pins['G'])[0][0], r23.mn(inrgn.pins['G'])[1][0]]
_mn0 = [r23.mn(inrgp.pins['G'])[0], r23.mn(iprgp.pins['G'])[0]]
_mn1 = [r23.mn(inrgn.pins['G'])[0], r23.mn(iprgn.pins['G'])[0]]
rgp_in = []
rgn_in = []
for i in range(trk_num):
    _, v0, r0, v1, _= dsn.route_via_track(grid=r23, mn=_mn0, track=[trk_ref[0]+2*i+1, None])
    rgp_in.append(r0)
    _, v0, r0, v1, _= dsn.route_via_track(grid=r23, mn=_mn1, track=[trk_ref[1]-2*i-1, None])
    rgn_in.append(r0)
# cross-couple
_mn = [r34.mn(rgp_in[0])[0].tolist(), r34.mn(rgn_in[-1])[0].tolist()]
routp = dsn.route(grid=r34, mn=_mn)
for r in rgp_in:
    dsn.via(r34, r34.mn.overlap(routp, r))
for r in rgn_out:
    dsn.via(r34, r34.mn.overlap(routp, r))
_mn = np.array(_mn) + np.array([[0, 1], [0, 1]])
routn = dsn.route(grid=r34, mn=_mn)
for r in rgn_in:
    dsn.via(r34, r34.mn.overlap(routn, r))
for r in rgp_out:
    dsn.via(r34, r34.mn.overlap(routn, r))
# RST - clk
_mn = [r23.mn(iprsp.pins['G'])[0], r23.mn(iprsn.pins['G'])[1]]
rrs0 = dsn.route(grid=r23, mn=_mn)
# Vertical clock route
trk = int(round((r23.mn.bottom_left(rrs0)[0]+r23.mn.bottom_right(rrs0)[0])/2))
_mn = [r23.mn(rcs1)[0], r23.mn(rrs0)[0]]
_, v0, rck, v1, _= dsn.route_via_track(grid=r23, mn=_mn, track=[trk, None])
# VSS
rvss0 = dsn.route(grid=r12, mn=[r12.mn(incsp.pins['RAIL'])[0], r12.mn(incsn.pins['RAIL'])[1]])
rvss1 = dsn.route(grid=r12, mn=[r12.mn(ininp.pins['RAIL'])[0], r12.mn(ininn.pins['RAIL'])[1]])
rvss2 = dsn.route(grid=r12, mn=[r12.mn(inrgp.pins['RAIL'])[0], r12.mn(inrgn.pins['RAIL'])[1]])
# VDD
rvdd0 = dsn.route(grid=r12, mn=[r12.mn(iprgp.pins['RAIL'])[0], r12.mn(iprgn.pins['RAIL'])[1]])

# 6. Create pins.
pinp = dsn.pin(name='INP', grid=r23, mn=r23.mn.bbox(ininp.pins['G']))
pinn = dsn.pin(name='INN', grid=r23, mn=r23.mn.bbox(ininn.pins['G']))
poutp = dsn.pin(name='OUTP', grid=r34, mn=r34.mn.bbox(routp))
poutn = dsn.pin(name='OUTN', grid=r34, mn=r34.mn.bbox(routn))
pck0 = dsn.pin(name='CK', grid=r23, mn=r23.mn.bbox(rck))
pvss0 = dsn.pin(name='VSS0', grid=r12, mn=r12.mn.bbox(rvss0), netname='VSS:')
pvss1 = dsn.pin(name='VSS1', grid=r12, mn=r12.mn.bbox(rvss1), netname='VSS:')
pvss2 = dsn.pin(name='VSS2', grid=r12, mn=r12.mn.bbox(rvss2), netname='VSS:')
pvdd0 = dsn.pin(name='VDD', grid=r12, mn=r12.mn.bbox(rvdd0))

# 7. Export to physical database.
print("Export design")
#print(dsn)
# Uncomment for GDS export
'''
#abstract = False  # export abstract
#laygo2.interface.gds.export(lib, filename=libname+'_'+cellname+'.gds', cellname=None, scale=1e-9,
#                            layermapfile="../technology_example/technology_example.layermap", physical_unit=1e-9, logical_unit=0.001,
#                            pin_label_height=0.1, pin_annotate_layer=['text', 'drawing'], text_height=0.1,
#                            abstract_instances=abstract)
'''

# Uncomment for SKILL export
'''
#skill_str = laygo2.interface.skill.export(lib, filename=libname+'_'+cellname+'.il', cellname=None, scale=1e-3)
#print(skill_str)
'''

# Uncomment for BAG export
laygo2.interface.bag.export(lib, filename=libname+'_'+cellname+'.il', cellname=None, scale=1e-3, reset_library=False, tech_library=tech.name)

# 7-a. Import the GDS file back and display
#with open('nand_generate.gds', 'rb') as stream:
#    pprint.pprint(laygo2.interface.gds.readout(stream, scale=1e-9))

# 8. Export to a template database file.
nat_temp = dsn.export_to_template()
laygo2.interface.yaml.export_template(nat_temp, filename=libname+'_templates.yaml', mode='append')

