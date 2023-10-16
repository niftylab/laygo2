##########################################################
#                                                        #
#              D-Flip Flop Layout Generator              #
#     Contributors: T. Shin, S. Park, Y. Oh, T. Kang     #
#                 Last Update: 2022-05-27                #
#                                                        #
##########################################################

import laygo2
import laygo2.interface
import laygo2_tech_quick_start as tech

# Parameter definitions #############
# Design Variables
cellname = "dff_2x"

# Templates
tpmos_name = "pmos"
tnmos_name = "nmos"

# Grids
pg_name = "placement_basic"
r12_name = "routing_12_cmos"
r23_name = "routing_23_cmos"
r34_name = "routing_34_basic"

# Design hierarchy
libname = "logic_generated"
# End of parameter definitions ######

# Generation start ##################
# 1. Load templates and grids.
print("Load templates")
templates = tech.load_templates()
# tpmos, tnmos = templates[tpmos_name], templates[tnmos_name]
tlib = laygo2.interface.yaml.import_template(filename="logic_generated_templates.yaml")

print("Load grids")
grids = tech.load_grids(templates=templates)
pg, r12, r23, r34 = grids[pg_name], grids[r12_name], grids[r23_name], grids[r34_name]
# print(grids[pg_name], grids[r12_name], grids[r23_name], grids[r34_name], sep="\n") # Uncomment if you want to print grids.

# 2. Create a design hierarchy
lib = laygo2.object.database.Library(name=libname)
dsn = laygo2.object.database.Design(name=cellname, libname=libname)
lib.append(dsn)

# 3. Create istances.
print("Create instances")
inv0 = tlib["inv_2x"].generate(name="inv0", netmap={"I": "CLK", "O": "ICLKB"})
inv1 = tlib["inv_2x"].generate(name="inv1", netmap={"I": "ICLKB", "O": "ICLK"})
inv2 = tlib["inv_2x"].generate(name="inv2", netmap={"I": "FLCH", "O": "LCH"})
inv3 = tlib["inv_2x"].generate(name="inv3", netmap={"I": "BLCH", "O": "OUT"})
tinv0 = tlib["tinv_2x"].generate(name="tinv0", netmap={"O": "FLCH", "EN": "ICLKB", "ENB": "ICLK"})
tinv1 = tlib["tinv_2x"].generate(name="tinv1", netmap={"I": "LCH", "O": "BLCH", "EN": "ICLK", "ENB": "ICLKB"})
tinv_small0 = tlib["tinv_1x"].generate(name="tinv_small0", netmap={"I": "LCH", "O": "FLCH", "EN": "ICLK", "ENB": "ICLKB"})
tinv_small1 = tlib["tinv_1x"].generate(name="tinv_small1", netmap={"I": "OUT", "O": "BLCH", "EN": "ICLKB", "ENB": "ICLK"})

dsn.place(grid=pg, inst=[inv0, inv1, tinv0, tinv_small0, inv2, tinv1, tinv_small1, inv3], mn=[0, 0])

# 5. Create and place wires.
print("Create wires")
_trk = r34.mn(inv1.pins["O"])[0, 1] - 2
# rc = laygo2.object.core.RoutingMeshTemplate(grid=r34)
rc = laygo2.object.template.routing.RoutingMeshTemplate(grid=r34)
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
    filename="dff_2x.png",
)
# skill export
skill_str = laygo2.interface.skill.export(lib, filename=libname + "_" + cellname + ".il", cellname=None, scale=1e-3)
# Filename example: logic_generated_dff_2x.il

# 8. Export to a template database file.
nat_temp = dsn.export_to_template()
laygo2.interface.yaml.export_template(nat_temp, filename=libname + "_templates.yaml", mode="append")
# Filename example: ./laygo2_generators_private/logic/logic_generated_templates.yaml
