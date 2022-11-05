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
cell_type = "dff_2x"

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
export_path = "."  # Layout generation path
export_path_skill = export_path  # SKILL file generation path
# End of parameter definitions ######

# Generation start ##################
# 1. Load templates and grids.
print("Load templates")
# templates = tech.load_templates()
# tpmos, tnmos = templates[tpmos_name], templates[tnmos_name]
tlib = laygo2.interface.yaml.import_template(filename=export_path + "logic_generated_templates.yaml")

print("Load grids")
grids = tech.load_grids(templates=templates)
pg, r12, r23, r34 = grids[pg_name], grids[r12_name], grids[r23_name], grids[r34_name]
# print(grids[pg_name], grids[r12_name], grids[r23_name], grids[r34_name], sep="\n") # Uncomment if you want to print grids.

for nf in nf_list:
    cellname = cell_type + "_" + str(nf) + "x"
    print("--------------------")
    print("Now Creating " + cellname)

    # 2. Create a design hierarchy
    lib = laygo2.object.database.Library(name=libname)
    dsn = laygo2.object.database.Design(name=cellname, libname=libname)
    lib.append(dsn)

    # 3. Create istances.
    print("Create instances")
    inv0 = tlib["inv_2x"].generate(name="inv0")
    inv1 = tlib["inv_2x"].generate(name="inv1")
    inv2 = tlib["inv_2x"].generate(name="inv2")
    inv3 = tlib["inv_2x"].generate(name="inv3")

    tinv0 = tlib["tinv_2x"].generate(name="tinv0")
    tinv1 = tlib["tinv_2x"].generate(name="tinv1")

    tinv_small0 = tlib["tinv_small_1x"].generate(name="tinv_small0")
    tinv_small1 = tlib["tinv_small_1x"].generate(name="tinv_small1")

    # 4. Place instances.
    dsn.place(grid=pg, inst=inv0, mn=[0, 0])
    dsn.place(grid=pg, inst=inv1, mn=pg.mn.bottom_right(inv0))
    dsn.place(grid=pg, inst=tinv0, mn=pg.mn.bottom_right(inv1))
    dsn.place(grid=pg, inst=tinv_small0, mn=pg.mn.bottom_right(tinv0))
    dsn.place(grid=pg, inst=inv2, mn=pg.mn.bottom_right(tinv_small0))
    dsn.place(grid=pg, inst=tinv1, mn=pg.mn.bottom_right(inv2))
    dsn.place(grid=pg, inst=tinv_small1, mn=pg.mn.bottom_right(tinv1))
    dsn.place(grid=pg, inst=inv3, mn=pg.mn.bottom_right(tinv_small1))

    # 5. Create and place wires.
    print("Create wires")

    # ICLK
    _mn = [r34.mn(inv1.pins["O"])[0], r34.mn(tinv_small1.pins["ENB"])[0]]
    _track = [None, r34.mn(inv1.pins["O"])[0, 1] - 2]
    mn_list = []
    mn_list.append(r34.mn(inv1.pins["O"])[0])
    mn_list.append(r34.mn(tinv0.pins["ENB"])[0])
    mn_list.append(r34.mn(tinv1.pins["EN"])[0])
    mn_list.append(r34.mn(tinv_small0.pins["EN"])[0])
    mn_list.append(r34.mn(tinv_small1.pins["ENB"])[0])
    dsn.route_via_track(grid=r34, mn=mn_list, track=_track)

    # ICLKB
    _track[1] += 1
    mn_list = []
    mn_list.append(r34.mn(inv0.pins["O"])[0])
    mn_list.append(r34.mn(inv1.pins["I"])[0])
    mn_list.append(r34.mn(tinv0.pins["EN"])[0])
    mn_list.append(r34.mn(tinv1.pins["ENB"])[0])
    mn_list.append(r34.mn(tinv_small0.pins["ENB"])[0])
    mn_list.append(r34.mn(tinv_small1.pins["EN"])[0])
    dsn.route_via_track(grid=r34, mn=mn_list, track=_track)

    # Front LATCH
    _track[1] += 1
    mn_list = []
    mn_list.append(r34.mn(inv2.pins["I"])[0])
    mn_list.append(r34.mn(tinv0.pins["O"])[0])
    mn_list.append(r34.mn(tinv_small0.pins["O"])[0])
    dsn.route_via_track(grid=r34, mn=mn_list, track=_track)

    # Back LATCH
    mn_list = []
    mn_list.append(r34.mn(inv3.pins["I"])[0])
    mn_list.append(r34.mn(tinv1.pins["O"])[0])
    mn_list.append(r34.mn(tinv_small1.pins["O"])[0])
    dsn.route_via_track(grid=r34, mn=mn_list, track=_track)

    # LATCH
    _track[1] += 1
    mn_list = []
    mn_list.append(r34.mn(inv2.pins["O"])[0])
    mn_list.append(r34.mn(tinv1.pins["I"])[0])
    mn_list.append(r34.mn(tinv_small0.pins["I"])[0])
    dsn.route_via_track(grid=r34, mn=mn_list, track=_track)

    # OUT
    mn_list = []
    mn_list.append(r34.mn(inv3.pins["O"])[0])
    mn_list.append(r34.mn(tinv_small1.pins["I"])[0])
    dsn.route_via_track(grid=r34, mn=mn_list, track=_track)

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
    laygo2.interface.bag.export(
        lib,
        filename=export_path_skill + libname + "_" + cellname + ".il",
        cellname=None,
        scale=1e-3,
        reset_library=False,
        tech_library=tech.name,
    )
    # Filename example: ./laygo2_generators_private/logic/skill/logic_generated_dff_2x.il

    # 8. Export to a template database file.
    nat_temp = dsn.export_to_template()
    laygo2.interface.yaml.export_template(nat_temp, filename=export_path + libname + "_templates.yaml", mode="append")
    # Filename example: ./laygo2_generators_private/logic/logic_generated_templates.yaml
