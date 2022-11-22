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
pg, r12, r23, r34 = grids[pg_name], grids[r12_name], grids[r23_name], grids[r34_name]
# print(grids[pg_name], grids[r12_name], grids[r23_name], grids[r34_name], sep="\n") # Uncomment if you want to print grids.
r23v = laygo2.object.grid.vstack([r23, r23, r23, r23.vflip(), r23.vflip()])

# 2. Create a design hierarchy
lib = laygo2.object.database.Library(name=libname)
dsn = laygo2.object.database.Design(name=cellname, libname=libname)
lib.append(dsn)

# 3. Create istances.
print("Create instances")
nf_ckh = 4
nf_in = 6
nf_rgn = 4 
nf_rgp = 4 
nf_rst = 2
nf_dmy = 2
nf_tot = max(nf_ckh, nf_in, nf_rgn, nf_rgp, nf_rst) + nf_dmy
inckhl0 = tnmos.generate(name="MNCKHL0",  params={"nf": nf_ckh, "nfdmyl": nf_tot-nf_ckh, "tie": "S"}, netmap={"D":"tail", "G":"CLK", "S":"VSS"})
inckhr0 = tnmos.generate(name="MNCKHR0",  params={"nf": nf_ckh, "nfdmyr": nf_tot-nf_ckh, "tie": "S"}, netmap={"D":"tail", "G":"CLK", "S":"VSS"})
ininl0  = tnmos.generate(name="MNINL0",   params={"nf": nf_in,  "nfdmyl": nf_tot-nf_in},              netmap={"D":"intn", "G":"INP", "S":"tail"})
ininr0  = tnmos.generate(name="MNINR0",   params={"nf": nf_in,  "nfdmyr": nf_tot-nf_in},              netmap={"D":"intp", "G":"INN", "S":"tail"})
inrgl0  = tnmos.generate(name="MNRGL0",   params={"nf": nf_rgn, "nfdmyl": nf_tot-nf_rgn},             netmap={"D":"OUTN", "G":"OUTP_G", "S":"intn"})
inrgr0  = tnmos.generate(name="MNRGR0",   params={"nf": nf_rgn, "nfdmyr": nf_tot-nf_rgn},             netmap={"D":"OUTP", "G":"OUTN_G", "S":"intp"})
iprgl0  = tpmos.generate(name="MPRGL0",   params={"nf": nf_rgp, "nfdmyl": nf_tot-nf_rgp, "tie": "S"}, netmap={"D":"OUTN", "G":"OUTP_G", "S":"VDD"}, transform = "MX")
iprgr0  = tpmos.generate(name="MPRGR0",   params={"nf": nf_rgp, "nfdmyr": nf_tot-nf_rgp, "tie": "S"}, netmap={"D":"OUTP", "G":"OUTN_G", "S":"VDD"}, transform = "MX")
iprstl0  = tpmos.generate(name="MPRSTL0", params={"nf": nf_rst, "nfdmyl": nf_tot-nf_rst, "tie": "S"}, netmap={"D":"intn", "G":"CLK", "S":"VDD"}, transform = "MX")
iprstr0  = tpmos.generate(name="MPRSTR0", params={"nf": nf_rst, "nfdmyr": nf_tot-nf_rst, "tie": "S"}, netmap={"D":"intp", "G":"CLK", "S":"VDD"}, transform = "MX")
iprstl1  = tpmos.generate(name="MPRSTL1", params={"nf": nf_rst, "nfdmyl": nf_tot-nf_rst, "tie": "S"}, netmap={"D":"OUTN", "G":"CLK", "S":"VDD"}, transform = "MX")
iprstr1  = tpmos.generate(name="MPRSTR1", params={"nf": nf_rst, "nfdmyr": nf_tot-nf_rst, "tie": "S"}, netmap={"D":"OUTP", "G":"CLK", "S":"VDD"}, transform = "MX")
dsn.place(grid=pg, inst=[[inckhl0, inckhr0], [ininl0, ininr0], [inrgl0, inrgr0], [iprstl0, iprstr0], [iprgl0, iprgr0], [iprstl1, iprstr1]], mn=[0, 0])
#dsn.place(grid=pg, inst=[iprstl0, iprstl1, iprstr1, iprstr0], mn=pg.mn(iprgl0.top_left))

# 5. Create and place wires.
print("Create wires")
_trkl = r23v.mn(inckhl0.pins["D"])[1, 0]
_trkr = r23v.mn(inckhr0.pins["D"])[0, 0]
nwire = int((nf_tot - nf_dmy)/ 2)

rc = laygo2.object.template.RoutingMeshTemplate(grid=r23v)
for i in range(nwire):
    rc.add_track(name="taill"+str(i), index=[_trkl-2*i, None], netname="tail")
    rc.add_track(name="tailr"+str(i), index=[_trkr+2*i, None], netname="tail")
    rc.add_track(name="intn"+str(i), index=[_trkl-2*i-1, None], netname="intn")
    rc.add_track(name="intp"+str(i), index=[_trkr+2*i+1, None], netname="intp")
    rc.add_track(name="OUTN"+str(i), index=[_trkl-2*i, None], netname="OUTN")
    rc.add_track(name="OUTP"+str(i), index=[_trkr+2*i, None], netname="OUTP")
rc.add_track(name="OUTP_G", index=[_trkl+1, None], netname="OUTP_G")
rc.add_track(name="OUTN_G", index=[_trkr-1, None], netname="OUTN_G")
rc.add_node([inckhl0, inckhr0, ininl0, ininr0, inrgl0, inrgr0])  # Add all instances to the routing mesh as nodes
rc.add_node([inrgl0, inrgr0, iprstl0, iprstr0, iprgl0, iprgr0, iprstl1, iprstr1])
rinst = rc.generate()
dsn.place(grid=pg, inst=rinst)

#_n = r34.mn(rc.pins["OUTP_G"])[0, 1]
_mn = [r34.mn(rinst.pins["OUTP_G"])[0], r34.mn(rinst.pins["OUTP0"])[1]]
_mn[0][1] += 2
_mn[1][1] = _mn[0][1]
vout0, rout0, vout1 = dsn.route(grid=r34, mn=_mn, via_tag=[True, True])
_mn = [r34.mn(rinst.pins["OUTN_G"])[0], r34.mn(rinst.pins["OUTN0"])[1]]
_mn[0][1] += 4
_mn[1][1] = _mn[0][1]
vout0, rout0, vout1 = dsn.route(grid=r34, mn=_mn, via_tag=[True, True])

'''
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
    ylim=[-100, 700],
    filename="dff_2x.png",
)
# skill export
skill_str = laygo2.interface.skill.export(lib, filename=libname + "_" + cellname + ".il", cellname=None, scale=1e-3)
# Filename example: logic_generated_dff_2x.il

# 8. Export to a template database file.
nat_temp = dsn.export_to_template()
laygo2.interface.yaml.export_template(nat_temp, filename=libname + "_templates.yaml", mode="append")
# Filename example: ./laygo2_generators_private/logic/logic_generated_templates.yaml
