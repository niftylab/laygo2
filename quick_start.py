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

import laygo2
import laygo2.interface

# should be 'import laygo2_tech as tech' for actual use.
import laygo2_tech_quick_start as tech

# Parameter definitions ##############
# Templates
tpmos_name = "pmos"
tnmos_name = "nmos"
# Grids
pg_name = "placement_cmos"
r12_name = "routing_12_cmos"
r23_name = "routing_23_cmos"
# Design hierarchy
libname = "laygo2_test"
cellname = "nand2"
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
lib = laygo2.Library(name=libname)
dsn = laygo2.Design(name=cellname)
lib.append(dsn)

# 3. Create instances.
print("Create instances")
in0 = tnmos.generate(name="MN0", params={"nf": nf_b, "tie": "S"})
ip0 = tpmos.generate(name="MP0", transform="MX", params={"nf": nf_b, "tie": "S"})
in1 = tnmos.generate(name="MN1", params={"nf": nf_a, "trackswap": True})
ip1 = tpmos.generate(name="MP1", transform="MX", params={"nf": nf_a, "tie": "S"})

# 4. Place instances.
## option 1 - vector-based placement
# dsn.place(grid=pg, inst=in0, mn=[0, 0])
# dsn.place(grid=pg, inst=ip0, mn=pg.mn.top_left(in0) + pg.mn.height_vec(ip0))
# dsn.place(grid=pg, inst=in1, mn=pg.mn.bottom_right(in0))
# dsn.place(grid=pg, inst=ip1, mn=pg.mn.top_right(ip0))
## option 2 - anchor-based placement
# dsn.place(grid=pg, inst=in0, mn=[0, 0])
# dsn.place(grid=pg, inst=ip0, anchor_xy=[in0.top_left, ip0.bottom_left])
# dsn.place(grid=pg, inst=in1, anchor_xy=[in0.bottom_right, in1.bottom_left])
# dsn.place(grid=pg, inst=ip1, anchor_xy=[in1.top_left, ip1.bottom_left])
## option 3 - array-based placement
dsn.place(grid=pg, inst=[[in0, in1], [ip0, ip1]], mn=[0, 0])

# 5. Create and place wires.
print("Create wires")
# A
_mn = [r23.mn(in1.pins["G"])[0], r23.mn(ip1.pins["G"])[0]]
va0, ra0, va1 = dsn.route(grid=r23, mn=_mn, via_tag=[True, True])
# B
_mn = [r23.mn(in0.pins["G"])[0], r23.mn(ip0.pins["G"])[0]]
vb0, rb0, vb1 = dsn.route(grid=r23, mn=_mn, via_tag=[True, True])
# internal
_mn = [r12.mn(in0.pins["D"])[1], r12.mn(in1.pins["S"])[0]]
dsn.route(grid=r23, mn=_mn)
# output
_mn = [r12.mn(ip0.pins["D"])[1], r12.mn(ip1.pins["D"])[0]]
dsn.route(grid=r23, mn=_mn)
_mn = [r23.mn(in1.pins["D"])[1], r23.mn(ip1.pins["D"])[1]]
_, rout0, _ = dsn.route(grid=r23, mn=_mn, via_tag=[True, True])

# 6. Create pins.
pB = dsn.pin(name="B", grid=r23, mn=r23.mn(rb0))
pA = dsn.pin(name="A", grid=r23, mn=r23.mn(ra0))
pout0 = dsn.pin(name="O", grid=r23, mn=r23.mn(rout0))
_mn = [r12.mn(in0.pins["RAIL"])[0], r12.mn(in1.pins["RAIL"])[1]]
pvss0 = dsn.pin(name="VSS", grid=r12, mn=_mn)
_mn = [r12.mn(ip0.pins["RAIL"])[0], r12.mn(ip1.pins["RAIL"])[1]]
pvdd0 = dsn.pin(name="VDD", grid=r12, mn=_mn)

# 7. Export to physical database.
print("Export design")
# matplotlib export
mpl_params = tech.tech_params["mpl"]
fig = laygo2.interface.mpl.export(
    lib,
    colormap=mpl_params["colormap"],
    order=mpl_params["order"],
)
filename = libname + "_" + cellname
# gds export
laygo2.interface.gdspy.export(
    lib,
    filename=filename + ".gds",
    scale=1e-9,
    layermapfile="./laygo2_tech_quick_start/laygo2_tech.layermap",
    physical_unit=1e-9,
    logical_unit=0.001,
    pin_label_height=0.1,
    svg_filename=filename + ".svg",
    png_filename=filename + ".png",
    # pin_annotation_layer=['text', 'drawing'], text_height=0.1,abstract_instances=abstract,
)
# skill export
skill_str = laygo2.interface.skill.export(lib, filename=libname + "_" + cellname + ".il", cellname=None, scale=1e-3)
# print(skill_str)

# 8. Export to a template database file.
nat_temp = dsn.export_to_template()
laygo2.interface.yaml.export_template(nat_temp, filename=libname + "_templates.yaml", mode="append")
