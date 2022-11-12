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

# 2. Create a design hierarchy
lib = laygo2.object.database.Library(name=libname)
dsn = laygo2.object.database.Design(name=cellname, libname=libname)
lib.append(dsn)

# 3. Create istances.
print("Create instances")
nf_ckh = 4
nf_in = 6
nf_rgn = 4 
nf_rgp = 2 
nf_dmy = 2
nf_tot = max(nf_ckh, nf_in) + nf_dmy
inckhl0 = tnmos.generate(name="MNCKHL0", params={"nf": nf_ckh, "nfdmyl": nf_tot-nf_ckh, "tie": "S"}, netmap={"D":"tail", "G":"CLK", "S":"VSS"})
inckhr0 = tnmos.generate(name="MNCKHR0", params={"nf": nf_ckh, "nfdmyr": nf_tot-nf_ckh, "tie": "S"}, netmap={"D":"tail", "G":"CLK", "S":"VSS"})
ininl0  = tnmos.generate(name="MNINL0",  params={"nf": nf_in,  "nfdmyl": nf_tot-nf_in},              netmap={"D":"intn", "G":"INP", "S":"tail"})
ininr0  = tnmos.generate(name="MNINR0",  params={"nf": nf_in,  "nfdmyr": nf_tot-nf_in},              netmap={"D":"intp", "G":"INN", "S":"tail"})
inrgl0  = tnmos.generate(name="MNRGL0",  params={"nf": nf_rgn, "nfdmyl": nf_tot-nf_rgn},             netmap={"D":"OUTN", "G":"OUTP", "S":"intn"})
inrgr0  = tnmos.generate(name="MNRGR0",  params={"nf": nf_rgn, "nfdmyr": nf_tot-nf_rgn},             netmap={"D":"OUTP", "G":"OUTN", "S":"intp"})
iprgl0  = tpmos.generate(name="MPRGL0",  params={"nf": nf_rgp, "nfdmyl": nf_tot-nf_rgp, "tie": "S"}, netmap={"D":"OUTN", "G":"OUTP", "S":"VDD"}, transform = "MX")
iprgr0  = tpmos.generate(name="MPRGR0",  params={"nf": nf_rgp, "nfdmyr": nf_tot-nf_rgp, "tie": "S"}, netmap={"D":"OUTP", "G":"OUTN", "S":"VDD"}, transform = "MX")
dsn.place(grid=pg, inst=[[inckhl0, inckhr0], [ininl0, ininr0], [inrgl0, inrgr0], [iprgl0, iprgr0]], mn=[0, 0])

# 5. Create and place wires.
print("Create wires")
# first row
# vss
_mn = [r23.mn(inckhl0.pins["S"])[0], r23.mn(inckhr0.pins["S"])[1]]
rvss0 = dsn.route(grid=r23, mn=_mn, via_tag=[False, False])
# tail
_mn = [r23.mn(inckhl0.pins["D"])[0], r23.mn(inckhr0.pins["D"])[1]]
rtail0 = dsn.route(grid=r23, mn=_mn, via_tag=[False, False])
# clk
_mn = [r23.mn(inckhl0.pins["G"])[0], r23.mn(inckhr0.pins["G"])[1]]
rck0 = dsn.route(grid=r23, mn=_mn, via_tag=[False, False])
# second row
# tail
_mn = [r23.mn(ininl0.pins["S"])[0], r23.mn(ininr0.pins["S"])[1]]
rtail1 = dsn.route(grid=r23, mn=_mn, via_tag=[False, False])

# vertical connections
# tail
_mn = [r23.mn(inckhl0.pins["D"]), r23.mn(ininl0.pins["S"])]
_mn_min = min(_mn[0][0, 0], _mn[1][0, 0])
_mn_max = _mn[1][1, 0]
for t in range(_mn_max, _mn_min - 1, -2):
    _track = [t, None]
    rv = dsn.route_via_track(grid=r23, mn=[_mn[0][1], _mn[1][1]], track=_track)
# int
_mn = [r23.mn(ininl0.pins["D"]), r23.mn(inrgl0.pins["S"])]
_mn_min = min(_mn[0][0, 0], _mn[1][0, 0])
_mn_max = _mn[0][1, 0]
for t in range(_mn_max, _mn_min - 1, -2):
    _track = [t, None]
    rv = dsn.route_via_track(grid=r23, mn=[_mn[0][1], _mn[1][1]], track=_track)
# out
_mn = [r23c.mn(inrgl0.pins["D"]), r23c.mn(iprgl0.pins["D"])]
_mn_min = min(_mn[0][0, 0], _mn[1][0, 0])
_mn_max = _mn[0][1, 0]
for t in range(_mn_max, _mn_min - 1, -2):
    _track = [t, None]
    rv = dsn.route_via_track(grid=r23, mn=[_mn[0][1], _mn[1][1]], track=_track)

## IN
#_mn = [r23.mn(in0.pins["G"])[0], r23.mn(ip0.pins["G"])[0]]
#_track = [r23.mn(in0.pins["G"])[0, 0] - 1, None]
#rin0 = dsn.route_via_track(grid=r23, mn=_mn, track=_track)
#
## OUT
#_mn = [r23.mn(in0.pins["D"])[1], r23.mn(ip0.pins["D"])[1]]
#vout0, rout0, vout1 = dsn.route(grid=r23, mn=_mn, via_tag=[True, True])

'''
_trkl = r23.mn(inckhl0.pins["D"])[1, 0]
_trkr = r23.mn(inckhr0.pins["D"])[0, 0]
#print(_trk)
rc = laygo2.object.template.RoutingMeshTemplate(grid=r23)
rc.add_track(name="taill", index=[_trkl, None], netname="tail")
rc.add_track(name="tailr", index=[_trkr, None], netname="tail")
rc.add_track(name="intn", index=[_trkl-1, None], netname="intn")
rc.add_track(name="intp", index=[_trkr+1, None], netname="intp")
rc.add_node([inckhl0, inckhr0, ininl0, ininr0, inrgl0, inrgr0])  # Add all instances to the routing mesh as nodes
rinst = rc.generate()
dsn.place(grid=pg, inst=rinst)
'''

#dsn.place(grid=pg, inst=rinst)
#rc = laygo2.object.template.RoutingMeshTemplate(grid=r23c)
#rc.add_track(name="OUTN", index=[_trkl, None], netname="OUTN")
#rc.add_track(name="OUTP", index=[_trkr, None], netname="OUTP")
#rc.add_node([inrgl0, inrgr0, iprgl0, iprgr0])  # Add all instances to the routing mesh as nodes
#rinst = rc.generate()
#dsn.place(grid=pg, inst=rinst)
'''

# 5. Create and place wires.
print("Create wires")
_trk = r34.mn(inv1.pins["O"])[0, 1] - 2
rc = laygo2.object.template.RoutingMeshTemplate(grid=r34)
rc.add_track(name="ICLK", index=[None, _trk], netname="ICLK")
rc.add_track(name="ICLKB", index=[None, _trk + 1], netname="ICLKB")
rc.add_track(name="FLCH", index=[None, _trk + 2], netname="FLCH")
rc.add_track(name="BLCH", index=[None, _trk + 2], netname="BLCH")
rc.add_track(name="LCH", index=[None, _trk + 3], netname="LCH")
rc.add_track(name="OUT", index=[None, _trk + 3], netname="OUT")
rc.add_node(list(dsn.instances.values()))  # Add all instances to the routing mesh as nodes
rinst = rc.generate()
dsn.place(grid=pg, inst=rinst)

# VSS
rvss0 = dsn.route(grid=r12, mn=[r12.mn.bottom_left(inv0), r12.mn.bottom_right(inv3)])
# VDD
rvdd0 = dsn.route(grid=r12, mn=[r12.mn.top_left(inv0), r12.mn.top_right(inv3)])

# 6. Create pins.
pin0 = dsn.pin(name="I", grid=r23, mn=r23.mn.bbox(tinv0.pins["I"]))
pclk0 = dsn.pin(name="CLK", grid=r23, mn=r23.mn.bbox(inv0.pins["I"]))
pout0 = dsn.pin(name="O", grid=r23, mn=r23.mn.bbox(inv3.pins["O"]))
pvss0 = dsn.pin(name="VSS", grid=r12, mn=r12.mn.bbox(rvss0))
pvdd0 = dsn.pin(name="VDD", grid=r12, mn=r12.mn.bbox(rvdd0))
'''

# 7. Export to physical database.
print("Export design")
print("")
# matplotlib export
mpl_params = tech.tech_params["mpl"]
fig = laygo2.interface.mpl.export(
    lib,
    cellname=cellname,
    colormap=mpl_params["colormap"],
    order=mpl_params["order"],
    xlim=[-100, 900],
    ylim=[-100, 500],
    filename="dff_2x.png",
)
# skill export
skill_str = laygo2.interface.skill.export(lib, filename=libname + "_" + cellname + ".il", cellname=None, scale=1e-3)
# Filename example: logic_generated_dff_2x.il

# 8. Export to a template database file.
nat_temp = dsn.export_to_template()
laygo2.interface.yaml.export_template(nat_temp, filename=libname + "_templates.yaml", mode="append")
# Filename example: ./laygo2_generators_private/logic/logic_generated_templates.yaml
