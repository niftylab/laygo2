##########################################################
#                                                        #
#            StrongARM Latch Layout Generator            #
#     Contributors:                                      #
#                 Last Update: 2022-11-09                #
#                                                        #
##########################################################

import numpy as np
import laygo2
import laygo2.interface
import laygo2_tech_quick_start as tech

# Parameter definitions #############
# Design Variables
cellname = "salatch"

# Templates
tpmos_name = "pmos"
tnmos_name = "nmos"

# Grids
pg_name = "placement_basic"
r12_name = "routing_12_mos"
r23_name = "routing_23_mos"
r23_cmos_name = "routing_23_cmos"
r34_name = "routing_34_basic"

# Design hierarchy
libname = "logic_generated"
# End of parameter definitions ######

# Generation start ##################
# 1. Load templates and grids.
print("Load templates")
templates = tech.load_templates()
tpmos, tnmos = templates[tpmos_name], templates[tnmos_name]
# print(templates[tpmos_name], templates[tnmos_name], sep="\n") # Uncomment if you want to print templates

print("Load grids")
grids = tech.load_grids(templates=templates)
pg, r12, r23, r23c, r34 = grids[pg_name], grids[r12_name], grids[r23_name], grids[r23_cmos_name], grids[r34_name]
# print(grids[pg_name], grids[r12_name], grids[r23_name], grids[r34_name], sep="\n") # Uncomment if you want to print grids.
print(r12)