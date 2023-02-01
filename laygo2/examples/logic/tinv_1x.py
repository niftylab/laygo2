##########################################################
#                                                        #
#           Tri-State Inverter Layout Generator          #
#     Contributors: T. Shin, S. Park, Y. Oh, T. Kang     #
#                 Last Update: 2022-11-04                #
#                                                        #
##########################################################

import laygo2
import laygo2.interface
import laygo2_tech_quick_start as tech

# Parameter definitions #############
# Design Variables
nf = 1
cellname = "tinv_" + str(nf) + "x"

# Templates
tpmos_name = "pmos"
tnmos_name = "nmos"
tpspace_name = "pspace"
tnspace_name = "nspace"

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
tpspace, tnspace = templates[tpspace_name], templates[tnspace_name]
# print(templates[tpmos_name], templates[tnmos_name], sep="\n") # Uncomment if you want to print templates

print("Load grids")
grids = tech.load_grids(templates=templates)
pg, r12, r23 = grids[pg_name], grids[r12_name], grids[r23_name]
# print(grids[pg_name], grids[r12_name], grids[r23_name], sep="\n") # Uncomment if you want to print grids

# 2. Create a design hierarchy
lib = laygo2.object.database.Library(name=libname)
dsn = laygo2.object.database.Design(name=cellname, libname=libname)
lib.append(dsn)

# 3. Create istances.
print("Create instances")

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
tpspace, tnspace = templates[tpspace_name], templates[tnspace_name]
# print(templates[tpmos_name], templates[tnmos_name], sep="\n") # Uncomment if you want to print templates

print("Load grids")
grids = tech.load_grids(templates=templates)
pg, r12, r23 = grids[pg_name], grids[r12_name], grids[r23_name]
# print(grids[pg_name], grids[r12_name], grids[r23_name], sep="\n") # Uncomment if you want to print grids

# 2. Create a design hierarchy
lib = laygo2.object.database.Library(name=libname)
dsn = laygo2.object.database.Design(name=cellname, libname=libname)
lib.append(dsn)

# 3. Create istances.
print("Create instances")
in0 = tnmos.generate(name="MN0", params={"nf": nf, "tie": "S"})
ip0 = tpmos.generate(name="MP0", transform="MX", params={"nf": nf, "tie": "S"})
in1 = tnmos.generate(name="MN1", params={"nf": nf, "trackswap": True})
ip1 = tpmos.generate(name="MP1", transform="MX", params={"nf": nf, "trackswap": True})
in2 = tnspace.generate(name="MN2", params={"nf": nf})
ip2 = tpspace.generate(name="MP2", transform="MX", params={"nf": nf})

# 4. Place instances.
dsn.place(grid=pg, inst=[[in0, in1, in2], [ip0, ip1, ip2]], mn=[0, 0])

# 5. Create and place wires.
print("Create wires")

# IN
_mn = [r23.mn(in0.pins["G"])[0], r23.mn(ip0.pins["G"])[0]]
v0, rin0, v1 = dsn.route(grid=r23, mn=_mn, via_tag=[True, True])

# OUT
_mn = [r23.mn(in1.pins["D"])[1], r23.mn(ip1.pins["D"])[1]]
vout0, rout0, vout1 = dsn.route(grid=r23, mn=_mn, via_tag=[True, True])

# EN
_mn = [r23.mn(in1.pins["G"])[1] + [1, 0], r23.mn(ip1.pins["G"])[1] + [1, 0]]
ven0, ren0 = dsn.route(grid=r23, mn=_mn, via_tag=[True, False])
_mn = [r23.mn(in1.pins["G"])[1], r23.mn(in1.pins["G"])[1] + [1, 0]]
renint = dsn.route(grid=r23, mn=_mn)

# ENB
_mn = [r23.mn(in1.pins["G"])[1] + [-1, 0], r23.mn(ip1.pins["G"])[1] + [-1, 0]]
renb0, venb0 = dsn.route(grid=r23, mn=_mn, via_tag=[False, True])

# Internal
_mn = [r23.mn(ip0.pins["D"])[0], r23.mn(ip1.pins["S"])[0]]
rintp0 = dsn.route(grid=r23, mn=_mn)
_mn = [r23.mn(in0.pins["D"])[0], r23.mn(in1.pins["S"])[0]]
rintn0 = dsn.route(grid=r23, mn=_mn)

# VSS
rvss0 = dsn.route(grid=r12, mn=[r12.mn(in0.pins["RAIL"])[0], r12.mn(in1.pins["RAIL"])[1]])

# VDD
rvdd0 = dsn.route(grid=r12, mn=[r12.mn(ip0.pins["RAIL"])[0], r12.mn(ip1.pins["RAIL"])[1]])

# 6. Create pins.
pin0 = dsn.pin(name="I", grid=r23, mn=r23.mn.bbox(rin0))
pen0 = dsn.pin(name="EN", grid=r23, mn=r23.mn.bbox(ren0))
penb0 = dsn.pin(name="ENB", grid=r23, mn=r23.mn.bbox(renb0))
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
    filename="tinv_1x.png",
)
# skill export
skill_str = laygo2.interface.skill.export(lib, filename=libname + "_" + cellname + ".il", cellname=None, scale=1e-3)
# Filename example: logic_generated_tinv_2x.il

# 8. Export to a template database file.
nat_temp = dsn.export_to_template()
laygo2.interface.yaml.export_template(nat_temp, filename=libname + "_templates.yaml", mode="append")
# Filename example: logic_generated_templates.yaml
