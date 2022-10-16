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
# Grids
pg_name = 'placement_basic'
# Design hierarchy
libname = 'laygo2_test'
cellname = 'transistors'
# End of parameter definitions #######

# Generation start ###################
# 1. Load templates and grids.
print("Load templates")
templates = tech.load_templates()
tpmos, tnmos = templates[tpmos_name], templates[tnmos_name]
print(templates[tpmos_name], templates[tnmos_name], sep="\n")

print("Load grids")
grids = tech.load_grids(templates=templates)
pg = grids[pg_name]
print(grids[pg_name])

# 2. Create a design hierarchy.
lib = laygo2.object.database.Library(name=libname)
dsn = laygo2.object.database.Design(name=cellname, libname=libname)
lib.append(dsn)

# 3. Create instances.
print("Create instances")
in0 = tnmos.generate(name='MN0', params={'nf': 4})  # vanilla case
in1 = tnmos.generate(name='MN1', params={'nf': 8})  # resize
in2 = tnmos.generate(name='MN2', params={'nf': 4, 'tie': 'D'})  # tie to VSS or VDD
in3 = tnmos.generate(name='MN3', params={'nf': 4, 'trackswap': True})  # swap horizontal tracks
in4 = tnmos.generate(name='MN4', params={'nf': 4, 'nfdmyl': 2, 'nfdmyr': 4})  # dummy
in5 = tnmos.generate(name='MN5', params={'nf': 4, 'bndl': False, 'bndr': False})  # no boundary
in6 = tnmos.generate(name='MN6', params={'nf': 4, 'gbndl': True, 'gbndr': True})  # global boundary
ip0 = tpmos.generate(name='MP0', transform='MX', params={'nf': 4, 'tie': 'S'}) # transform

# 4. Place instances.
dsn.place(grid=pg, inst=in0, mn=[0, 0])                  # basic placement
dsn.place(grid=pg, inst=in1, mn=[8, 0])                  # abstract placement
dsn.place(grid=pg, inst=in2, mn=pg.mn.bottom_right(in1)) # same with pg == in1.bottom_right
dsn.place(grid=pg, inst=in3, mn=pg.mn.bottom_right(in2))  
dsn.place(grid=pg, inst=in4, mn=pg.mn.bottom_right(in3))  
dsn.place(grid=pg, inst=in5, mn=pg.mn.bottom_right(in4))  
dsn.place(grid=pg, inst=in6, mn=pg.mn.bottom_right(in5))  
dsn.place(grid=pg, inst=ip0, mn=pg.mn.top_left(in0) + pg.mn.height_vec(ip0))  # +height_vec due to MX transform
print(pg.mn.bottom_right(in1))
# 7. Export to physical database.
print("Export design")
# Uncomment for GDS export
"""
#abstract = False  # export abstract
#laygo2.interface.gds.export(lib, filename=libname+'_'+cellname+'.gds', cellname=None, scale=1e-9,
#                            layermapfile="../technology_example/technology_example.layermap", physical_unit=1e-9, logical_unit=0.001,

#                            abstract_instances=abstract)
"""

# Uncomment for SKILL export
"""
#skill_str = laygo2.interface.skill.export(lib, filename=libname+'_'+cellname+'.il', cellname=None, scale=1e-3)
#print(skill_str)
"""

# Uncomment for BAG export
laygo2.interface.bag.export(lib, filename=libname+'_'+cellname+'.il', cellname=None, scale=1e-3, reset_library=True, tech_library=tech.name)


# 7-a. Import the GDS file back and display
#with open('nand_generate.gds', 'rb') as stream:
#    pprint.pprint(laygo2.interface.gds.readout(stream, scale=1e-9))

# 8. Export to a template database file.
nat_temp = dsn.export_to_template()
laygo2.interface.yaml.export_template(nat_temp, filename=libname+'_templates.yaml', mode='append')

