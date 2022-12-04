##########################################################

import laygo2
import laygo2.interface
import laygo2_tech_quick_start as tech


# (optional) matplotlib export
mpl_params = tech.tech_params["mpl"]
# End of parameter definitions ######

# Parameter definitions #############
# Design Variables
nf = 2
cellname = "inv_" + str(nf) + "x"

# Templates
tpmos_name = "pmos"
tnmos_name = "nmos"

# Grids
pg_name = "placement_basic"
r12_name = "routing_12_cmos"
r23_name = "routing_23_cmos"

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
pg, r12, r23 = grids[pg_name], grids[r12_name], grids[r23_name]
# print(grids[pg_name], grids[r12_name], grids[r23_name], sep="\n") # Uncomment if you want to print grids

fig = laygo2.interface.mpl.export_grid(r12, colormap=mpl_params["colormap"], order=mpl_params["order"], xlim=[-100, 200], ylim=[-100, 400])
#fig = laygo2.interface.mpl.export_grid(pg, colormap=mpl_params["colormap"], order=mpl_params["order"], xlim=[-100, 100], ylim=[-100, 100])
#print(inst0.pins['in'])
#print(inst0.pins['in'][0, 1])
