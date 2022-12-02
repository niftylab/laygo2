##########################################################
#                                                        #
#            StrongARM Latch Layout Generator            #
#     Contributors:                                      #
#                 Last Update: 2022-11-09                #
#                                                        #
##########################################################

import laygo2
import laygo2.interface
import laygo2_tech_quick_start as tech

# Parameter definitions #############
# Design Variables
cellname = "salatch"
nf_ckh = 4
nf_in = 6
nf_rgn = 4 
nf_rgp = 4 
nf_rst = 2
nf_dmy = 2
nf_tot = max(nf_ckh, nf_in, nf_rgn, nf_rgp, nf_rst) + nf_dmy

# Templates
tpmos_name = "pmos"
tnmos_name = "nmos"

# Grids
pg_name = "placement_basic"
r12_name = "routing_12_mos"
r23_name = "routing_23_mos"

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
r23v = laygo2.object.grid.vstack([r23, r23, r23, r23.vflip(), r23.vflip(), r23.vflip()])

# 2. Create a design hierarchy
lib = laygo2.object.database.Library(name=libname)
dsn = laygo2.object.database.Design(name=cellname, libname=libname)
lib.append(dsn)

# 3. Create instances.
print("Create instances")

# Instance parameters
params_inckhl0 = {"nf": nf_ckh, "nfdmyl": nf_tot-nf_ckh, "nfdmyr": 2,             "bndr": False, "tie": "S"}
params_inckhr0 = {"nf": nf_ckh, "nfdmyl": 2,             "nfdmyr": nf_tot-nf_ckh, "bndl": False, "tie": "S"}
params_ininl0  = {"nf": nf_in,  "nfdmyl": nf_tot-nf_in,  "nfdmyr": 2,             "bndr": False}
params_ininr0  = {"nf": nf_in,  "nfdmyl": 2,             "nfdmyr": nf_tot-nf_in , "bndl": False}
params_inrgl0  = {"nf": nf_rgn, "nfdmyl": nf_tot-nf_rgn, "nfdmyr": 2,             "bndr": False}
params_inrgr0  = {"nf": nf_rgn, "nfdmyl": 2,             "nfdmyr": nf_tot-nf_rgn, "bndl": False}
params_iprgl0  = {"nf": nf_rgp, "nfdmyl": nf_tot-nf_rgp, "nfdmyr": 2,             "bndr": False, "tie": "S"}
params_iprgr0  = {"nf": nf_rgp, "nfdmyl": 2,             "nfdmyr": nf_tot-nf_rgp, "bndl": False, "tie": "S"}
params_iprstl0 = {"nf": nf_rst, "nfdmyl": nf_tot-nf_rst, "nfdmyr": 2,             "bndr": False, "tie": "S"}
params_iprstr0 = {"nf": nf_rst, "nfdmyl": 2,             "nfdmyr": nf_tot-nf_rst, "bndl": False, "tie": "S"}
params_iprstl1 = {"nf": nf_rst, "nfdmyl": nf_tot-nf_rst, "nfdmyr": 2,             "bndr": False, "tie": "S"}
params_iprstr1 = {"nf": nf_rst, "nfdmyl": 2,             "nfdmyr": nf_tot-nf_rst, "bndl": False, "tie": "S"}

# Net mappings
netmap_inckhl0 = {"D":"tail", "G":"CLK", "S":"VSS"}
netmap_inckhr0 = {"D":"tail", "G":"CLK", "S":"VSS"}
netmap_ininl0  = {"D":"intn", "G":"INP", "S":"tail"}
netmap_ininr0  = {"D":"intp", "G":"INN", "S":"tail"}
netmap_inrgl0  = {"D":"OUTN", "G":"OUTP", "S":"intn"}
netmap_inrgr0  = {"D":"OUTP", "G":"OUTN", "S":"intp"}
netmap_iprgl0  = {"D":"OUTN", "G":"OUTP", "S":"VDD"}
netmap_iprgr0  = {"D":"OUTP", "G":"OUTN", "S":"VDD"}
netmap_iprstl0 = {"D":"intn", "G":"CLK", "S":"VDD"}
netmap_iprstr0 = {"D":"intp", "G":"CLK", "S":"VDD"}
netmap_iprstl1 = {"D":"OUTN", "G":"CLK", "S":"VDD"}
netmap_iprstr1 = {"D":"OUTP", "G":"CLK", "S":"VDD"}

# Generate instances
inckhl0 = tnmos.generate(name="MNCKHL0", params=params_inckhl0, netmap=netmap_inckhl0)
inckhr0 = tnmos.generate(name="MNCKHR0", params=params_inckhr0, netmap=netmap_inckhr0)
ininl0  = tnmos.generate(name="MNINL0",  params=params_ininl0,  netmap=netmap_ininl0)
ininr0  = tnmos.generate(name="MNINR0",  params=params_ininr0,  netmap=netmap_ininr0)
inrgl0  = tnmos.generate(name="MNRGL0",  params=params_inrgl0,  netmap=netmap_inrgl0)
inrgr0  = tnmos.generate(name="MNRGR0",  params=params_inrgr0,  netmap=netmap_inrgr0)
iprgl0  = tpmos.generate(name="MPRGL0",  params=params_iprgl0,  netmap=netmap_iprgl0,  transform = "MX")
iprgr0  = tpmos.generate(name="MPRGR0",  params=params_iprgr0,  netmap=netmap_iprgr0,  transform = "MX")
iprstl0 = tpmos.generate(name="MPRSTL0", params=params_iprstl0, netmap=netmap_iprstl0, transform = "MX")
iprstr0 = tpmos.generate(name="MPRSTR0", params=params_iprstr0, netmap=netmap_iprstr0, transform = "MX")
iprstl1 = tpmos.generate(name="MPRSTL1", params=params_iprstl1, netmap=netmap_iprstl1, transform = "MX")
iprstr1 = tpmos.generate(name="MPRSTR1", params=params_iprstr1, netmap=netmap_iprstr1, transform = "MX")

# Place instances
dsn.place(grid=pg, inst=[[inckhl0, inckhr0], [ininl0, ininr0], [inrgl0, inrgr0], [iprstl0, iprstr0], [iprgl0, iprgr0], [iprstl1, iprstr1]], mn=[0, 0])
#dsn.place(grid=pg, inst=[iprstl0, iprstl1, iprstr1, iprstr0], mn=pg.mn(iprgl0.top_left))

# 5. Create and place wires.
print("Create wires")
# Track indices
_trkl = r23v.mn(inckhl0.pins["D"])[1, 0]
_trkr = r23v.mn(inckhr0.pins["D"])[0, 0]
_trkc = r23v.mn(inckhl0.bbox)[1, 0]
nwire = int((nf_tot - nf_dmy)/ 2)
rc0 = laygo2.object.template.RoutingMeshTemplate(grid=r23v)
for i in range(nwire):  # route multiple wires to reduce R
    rc0.add_track(name="taill"+str(i), index=[_trkl-2*i, None], netname="tail")
    rc0.add_track(name="tailr"+str(i), index=[_trkr+2*i, None], netname="tail")
    rc0.add_track(name="intn"+str(i), index=[_trkl-2*i-1, None], netname="intn")
    rc0.add_track(name="intp"+str(i), index=[_trkr+2*i+1, None], netname="intp")
    rc0.add_track(name="OUTN"+str(i), index=[_trkl-2*i, None], netname="OUTN")
    rc0.add_track(name="OUTP"+str(i), index=[_trkr+2*i, None], netname="OUTP")
rc0.add_track(name="CLK0", index=[_trkc, None], netname="CLK")
# Add all instances to the routing mesh as nodes
rc0.add_node([inckhl0, inckhr0])
rc0.add_node([ininl0, ininr0])
rc0.add_node([inrgl0.pins["D"], inrgr0.pins["D"]]) # Do not connect gate pins
rc0.add_node([iprstl0, iprstr0])
rc0.add_node([iprgl0.pins["D"], iprgr0.pins["D"]])
rc0.add_node([iprstl1, iprstr1])
rinst = rc0.generate()
dsn.place(grid=pg, inst=rinst)

# Crosscoupled inverter outputs
rc1 = laygo2.object.template.RoutingMeshTemplate(grid=r23v)
rc1.add_track(name="OUTPG0", index=[_trkl+2, None], netname="OUTP")
rc1.add_track(name="OUTNG0", index=[_trkr-2, None], netname="OUTN")
rc1.add_node([inrgl0.pins["G"], inrgr0.pins["G"]])
rc1.add_node([iprgl0.pins["G"], iprgr0.pins["G"]])
rc1.add_node([inrgl0.pins["D"], iprgr0.pins["D"]])
rinst1 = rc1.generate()
dsn.place(grid=pg, inst=rinst1)

# Input wires
_mn = [[_trkl+1, 0], [_trkl+1, r23v.mn(ininl0.pins["G"])[0, 1]]]
rinp0, v0 = dsn.route(grid=r23v, mn=_mn, via_tag=[False, True])
_mn = [[_trkl+1, r23v.mn(ininl0.pins["G"])[0, 1]], r23v.mn(ininl0.pins["G"])[0]]
rinp1 = dsn.route(grid=r23v, mn=_mn)
_mn = [[_trkr-1, 0], [_trkr-1, r23v.mn(ininr0.pins["G"])[0, 1]]]
rinn0, v0 = dsn.route(grid=r23v, mn=_mn, via_tag=[False, True])
_mn = [[_trkr-1, r23v.mn(ininr0.pins["G"])[0, 1]], r23v.mn(ininr0.pins["G"])[0]]
rinn1 = dsn.route(grid=r23v, mn=_mn)

# VSS
rvss0 = dsn.route(grid=r23v, mn=[r23v.mn.bottom_left(inckhl0), r23v.mn.bottom_right(inckhr0)])
rvss1 = dsn.route(grid=r23v, mn=[r23v.mn.bottom_left(ininl0), r23v.mn.bottom_right(ininr0)])
rvss2 = dsn.route(grid=r23v, mn=[r23v.mn.bottom_left(inrgl0), r23v.mn.bottom_right(inrgr0)])
# VDD
rvdd0 = dsn.route(grid=r23v, mn=[r23v.mn.top_left(iprstl1), r23v.mn.top_right(iprstr1)])
rvdd1 = dsn.route(grid=r23v, mn=[r23v.mn.top_left(iprgl0), r23v.mn.top_right(iprgr0)])
rvdd2 = dsn.route(grid=r23v, mn=[r23v.mn.top_left(iprstl0), r23v.mn.top_right(iprstr0)])

# 6. Create pins.
pclk0 = dsn.pin(name="CLK", grid=r23v, mn=r23v.mn.bbox(rinst.pins["CLK0"]))
pinp0 = dsn.pin(name="INP", grid=r23v, mn=r23v.mn.bbox(rinp0))
pinn0 = dsn.pin(name="INN", grid=r23v, mn=r23v.mn.bbox(rinn0))
poutp0 = dsn.pin(name="OUTP", grid=r23v, mn=r23v.mn.bbox(rinst.pins["OUTP0"]))
poutn0 = dsn.pin(name="OUTN", grid=r23v, mn=r23v.mn.bbox(rinst.pins["OUTN0"]))
pvss0 = dsn.pin(name="VSS0", grid=r23v, mn=r23v.mn.bbox(rvss0), netname="VSS:")
pvss1 = dsn.pin(name="VSS1", grid=r23v, mn=r23v.mn.bbox(rvss1), netname="VSS:")
pvss2 = dsn.pin(name="VSS2", grid=r23v, mn=r23v.mn.bbox(rvss2), netname="VSS:")
pvdd0 = dsn.pin(name="VDD0", grid=r23v, mn=r23v.mn.bbox(rvdd0), netname="VDD:")
pvdd1 = dsn.pin(name="VDD1", grid=r23v, mn=r23v.mn.bbox(rvdd1), netname="VDD:")
pvdd2 = dsn.pin(name="VDD2", grid=r23v, mn=r23v.mn.bbox(rvdd2), netname="VDD:")

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
    filename="salatch.png",
)
# skill export
skill_str = laygo2.interface.skill.export(lib, filename=libname + "_" + cellname + ".il", cellname=None, scale=1e-3)
# Filename example: logic_generated_dff_2x.il

# 8. Export to a template database file.
nat_temp = dsn.export_to_template()
laygo2.interface.yaml.export_template(nat_temp, filename=libname + "_templates.yaml", mode="append")
# Filename example: ./laygo2_generators_private/logic/logic_generated_templates.yaml
