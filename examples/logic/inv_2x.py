##########################################################
#                                                        #
#                Inverter Layout Generator               #
#     Contributors: T. Shin, S. Park, Y. Oh, T. Kang     #
#                 Last Update: 2022-11-04                #
#                                                        #
##########################################################

import laygo2
import laygo2.interface
import laygo2_tech_quick_start as tech

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

# 2. Create a design hierarchy
lib = laygo2.object.database.Library(name=libname)
dsn = laygo2.object.database.Design(name=cellname, libname=libname)
lib.append(dsn)

# 3. Create instances.
print("Create instances")
in0 = tnmos.generate(name="MN0", params={"nf": nf, "tie": "S"})
ip0 = tpmos.generate(name="MP0", transform="MX", params={"nf": nf, "tie": "S"})

# 4. Place instances.
dsn.place(grid=pg, inst=[[in0], [ip0]], mn=[0, 0])

# 5. Create and place wires.
print("Create wires")

# IN
_mn = [r23.mn(in0.pins["G"])[0], r23.mn(ip0.pins["G"])[0]]
_track = [r23.mn(in0.pins["G"])[0, 0] - 1, None]
rin0 = dsn.route_via_track(grid=r23, mn=_mn, track=_track)

# OUT
_mn = [r23.mn(in0.pins["D"])[1], r23.mn(ip0.pins["D"])[1]]
vout0, rout0, vout1 = dsn.route(grid=r23, mn=_mn, via_tag=[True, True])

# VSS
rvss0 = dsn.route(grid=r12, mn=[r12.mn(in0.pins["RAIL"])[0], r12.mn(in0.pins["RAIL"])[1]])

# VDD
rvdd0 = dsn.route(grid=r12, mn=[r12.mn(ip0.pins["RAIL"])[0], r12.mn(ip0.pins["RAIL"])[1]])

# 6. Create pins.
pin0 = dsn.pin(name="I", grid=r23, mn=r23.mn.bbox(rin0[2]))
pout0 = dsn.pin(name="O", grid=r23, mn=r23.mn.bbox(rout0))
pvss0 = dsn.pin(name="VSS", grid=r12, mn=r12.mn.bbox(rvss0))
pvdd0 = dsn.pin(name="VDD", grid=r12, mn=r12.mn.bbox(rvdd0))

# 7. Export to physical database.
print("Export design")
# matplotlib export
mpl_params = tech.tech_params["mpl"]
fig = laygo2.interface.mpl.export(
    lib,
    cellname=cellname,
    colormap=mpl_params["colormap"],
    order=mpl_params["order"],
    filename="inv_2x.png",
)
# skill export
skill_str = laygo2.interface.skill.export(lib, filename=libname + "_" + cellname + ".il", cellname=None, scale=1e-3)
# Filename example: logic_generated_inv_2x.il

# 8. Export to a template database file.
nat_temp = dsn.export_to_template()
laygo2.interface.yaml.export_template(nat_temp, filename=libname + "_templates.yaml", mode="append")
# Filename example: logic_generated_templates.yaml
