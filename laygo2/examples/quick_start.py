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

import numpy as np
import pprint
import laygo2
import laygo2.interface
import laygo2_tech as tech

# Parameter definitions ##############
# Templates
tpmos_name = 'pmos'
tnmos_name = 'nmos'
# Grids
pg_name = 'placement_cmos'
r12_name = 'routing_12_cmos'
r23_name = 'routing_23_cmos'
# Design hierarchy
libname = 'laygo2_test'
cellname = 'nand2'
# Design parameters
nf_a = 3
nf_b = 4
# End of parameter definitions #######

# Generation start ###################
# 1. Load templates and grids.
print("Load templates")
templates = tech.load_templates()
tpmos, tnmos = templates[tpmos_name], templates[tnmos_name]
print(templates[tpmos_name], templates[tnmos_name], sep="\n")

print("Load grids")
grids = tech.load_grids(templates=templates)
pg, r12, r23 = grids[pg_name], grids[r12_name], grids[r23_name]
print(grids[pg_name], grids[r12_name], grids[r23_name], sep="\n")

# 2. Create a design hierarchy.
lib = laygo2.object.database.Library(name=libname)
dsn = laygo2.object.database.Design(name=cellname)
lib.append(dsn)

# 3. Create instances.
print("Create instances")
in0 = tnmos.generate(name='MN0', params={'nf': nf_b})
sd_swap = False if nf_b % 2 == 1 else True
in1 = tnmos.generate(name='MN1', params={'nf': nf_a, 'sd_swap': sd_swap})
ip0 = tpmos.generate(name='MP0', transform='MX', params={'nf': nf_b})
sd_swap = True if nf_b % 2 == 1 else False
ip1 = tpmos.generate(name='MP1', transform='MX', params={'nf': nf_a, 'sd_swap': sd_swap})

# 4. Place instances.
dsn.place(grid=pg, inst=in0, mn=pg.mn[0, 0])
dsn.place(grid=pg, inst=in1, mn=pg.mn.bottom_right(in0))  # same with pg == in0.bottom_right
dsn.place(grid=pg, inst=ip0, mn=pg.mn.top_left(in0) + np.array([0, pg.mn.height(ip0)]))  # +height due to MX transform
dsn.place(grid=pg, inst=ip1, mn=pg.mn.top_right(ip0))

# 5. Create and place wires.
print("Create wires")
# A
ra0 = dsn.route(grid=r12, mn=r12.mn.bbox(in1.pins['G']))
va0 = dsn.via(grid=r12, mn=r12.mn.overlap(ra0, in1.pins['G'], type='array'))
ra1 = dsn.route(grid=r12, mn=r12.mn.bbox(ip1.pins['G']))
va1 = dsn.via(grid=r12, mn=r12.mn.overlap(ra1, ip1.pins['G'], type='array'))
va3, ra2, va4 = dsn.route(grid=r23, mn=[r23.mn.bottom_left(ra0), r23.mn.top_left(ra1)], via_tag=[True, True])
# B
rb0 = dsn.route(grid=r12, mn=r12.mn.bbox(in0.pins['G']))
vb0 = dsn.via(grid=r12, mn=r12.mn.overlap(rb0, in0.pins['G'], type='array'))
rb1 = dsn.route(grid=r12, mn=r12.mn.bbox(ip0.pins['G']))
vb1 = dsn.via(grid=r12, mn=r12.mn.overlap(rb1, ip0.pins['G'], type='array'))
vb3, rb2, vb4 = dsn.route(grid=r23, mn=[r23.mn.bottom_left(rb0), r23.mn.top_left(rb1)], via_tag=[True, True])
# Internal
if not (nf_a == 1 and nf_b == 1):
    ri0 = dsn.route(grid=r12, mn=[r12.mn.bottom_left(in0.pins['D'][0]) + np.array([0, 1]),
                                  r12.mn.bottom_right(in1.pins['S'][-1]) + np.array([0, 1])])
    vi0 = [dsn.via(grid=r12, mn=r12.mn.overlap(ri0, i, type='point')) for i in in0.pins['D']]
    vi1 = [dsn.via(grid=r12, mn=r12.mn.overlap(ri0, i, type='point')) for i in in1.pins['S']]
# Output
ron0 = dsn.route(grid=r12, mn=[r12.mn.bottom_left(in1.pins['D'][0]) + np.array([0, 2]),
                               r12.mn.bottom_right(in1.pins['D'][-1]) + np.array([0, 2])])
von0 = [dsn.via(grid=r12, mn=r12.mn.overlap(ron0, i, type='point')) for i in in1.pins['D']]
rop0 = dsn.route(grid=r12, mn=[r12.mn.bottom_left(ip0.pins['D'][0]),
                               r12.mn.bottom_right(ip1.pins['D'][-1])])
vop0 = [dsn.via(grid=r12, mn=r12.mn.overlap(rop0, i, type='point')) for i in ip0.pins['D']]
vop1 = [dsn.via(grid=r12, mn=r12.mn.overlap(rop0, i, type='point')) for i in ip1.pins['D']]
m = r23.mn.bottom_right(ra2)[0] + 1
vo0, ro0, vo1 = dsn.route(grid=r23, mn=np.array([[m, r23.mn.bottom_right(ron0)[1]], [m, r23.mn.bottom_right(rop0)[1]]]),
                via_tag=[True, True])
# VSS
rvss0 = dsn.route(grid=r12, mn=[r12.mn.bottom_left(in0.pins['S'][0]), r12.mn.bottom_left(in1.pins['S'][0])])
vvss = [dsn.via(grid=r12, mn=r12.mn.overlap(rvss0, s, type='point')) for s in in0.pins['S']]
# VDD
rvdd0 = dsn.route(grid=r12, mn=[r12.mn.top_left(ip0.pins['S'][0]), r12.mn.top_right(ip1.pins['S'][-1])])
vvdd = [dsn.via(grid=r12, mn=r12.mn.overlap(rvdd0, s, type='point')) for s in ip0.pins['S']]
vvdd += [dsn.via(grid=r12, mn=r12.mn.overlap(rvdd0, s, type='point')) for s in ip1.pins['S']]

# 6. Create pins.
pa0 = dsn.pin(name='A', grid=r23, mn=r23.mn.bbox(ra2))
pb0 = dsn.pin(name='B', grid=r23, mn=r23.mn.bbox(rb2))
po0 = dsn.pin(name='O', grid=r23, mn=r23.mn.bbox(ro0))
pvss0 = dsn.pin(name='VSS', grid=r12, mn=r12.mn.bbox(rvss0))
pvdd0 = dsn.pin(name='VDD', grid=r12, mn=r12.mn.bbox(rvdd0))

print(dsn)

# 7. Export to physical database.
print("Export design")
abstract = False  # export abstract
laygo2.interface.gds.export(lib, filename=libname+'_'+cellname+'.gds', cellname=None, scale=1e-9,
                            layermapfile="./quick_start_tech/technology_example.layermap", physical_unit=1e-9, logical_unit=0.001,
                            pin_label_height=0.1, pin_annotate_layer=['text', 'drawing'], text_height=0.1,
                            abstract_instances=abstract)
skill_str = laygo2.interface.skill.export(lib, filename=libname+'_'+cellname+'.il', cellname=None, scale=1e-3)
print(skill_str)

# 7-a. Import the GDS file back and display
with open(libname+'_'+cellname+'.gds', 'rb') as stream:
    pprint.pprint(laygo2.interface.gds.readout(stream, scale=1e-9))

# 8. Export to a template database file.
nat_temp = dsn.export_to_template()
laygo2.interface.yaml.export_template(nat_temp, filename=libname+'_templates.yaml', mode='append')

