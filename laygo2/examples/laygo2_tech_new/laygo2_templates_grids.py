import laygo2.tech.tech_templates
import laygo2.tech.tech_grids
from laygo2.object.template.tile import TileMOSTemplate, TileTapTemplate

tech_fname = './laygo2_tech/laygo2_tech.yaml'

def load_udf_templates(tlib, glib, libname):
    """ load udf templates 
    """

    placement_pattern = [ "gbndl", "bndl", "dmyl",  "core", "dmyr", "bndr", "gbndr" ]
    transform_pattern = dict( gbndl = "R0", dmyl = "R0", bndl  = "R0", core  = "R0",
                                            dmyr = "MY", bndr  = "R0", gbndr = "R0" )
    routing_map       = dict( G = 3, S = 1, D = 2, G_extension0_x = [None,None], S_extension0_m = [None, None], D_extension0_m = [None, None])
    
    nmos_ulvt = dict(
        core  = 'nmos4_fast_center_nf2',
        dmyr  = 'nmos4_fast_dmy_nf2',
        dmyl  = 'nmos4_fast_dmy_nf2',
        bndr  = 'nmos4_fast_boundary',
        bndl  = 'nmos4_fast_boundary',
        gbndr = 'nmos4_fast_boundary',
        gbndl = 'nmos4_fast_boundary',

        grid  = "routing_12_mos",
    )

    pmos_ulvt = dict(
        core  = 'pmos4_fast_center_nf2',
        dmyr  = 'pmos4_fast_dmy_nf2',
        dmyl  = 'pmos4_fast_dmy_nf2',
        bndr  = 'pmos4_fast_boundary',
        bndl  = 'pmos4_fast_boundary',
        gbndr = 'pmos4_fast_boundary',
        gbndl = 'pmos4_fast_boundary',

        grid  = "routing_12_mos",
        )

    ptap_ulvt = dict(
        core  = 'ntap_fast_center_nf2_v2',
        dmyr  = 'nmos4_fast_dmy_nf2',
        dmyl  = 'nmos4_fast_dmy_nf2',
        bndr  = 'ntap_fast_boundary',
        bndl  = 'ntap_fast_boundary',
        gbndr = 'ntap_fast_boundary',
        gbndl = 'ntap_fast_boundary',

        grid  = "routing_12_mos",
    )

    ntap_ulvt = dict(
        core  = 'ptap_fast_center_nf2_v2',
        dmyr  = 'pmos4_fast_dmy_nf2',
        dmyl  = 'pmos4_fast_dmy_nf2',
        bndr  = 'ptap_fast_boundary',
        bndl  = 'ptap_fast_boundary',
        gbndr = 'ptap_fast_boundary',
        gbndl = 'ptap_fast_boundary',

        grid  = "routing_12_mos",
    )
   
    gen_list = [["nmos", nmos_ulvt], ["pmos", pmos_ulvt]]
    for name, placement_map in gen_list:
        grid_name  = placement_map["grid"]

        temp = TileMOSTemplate( tlib, glib, grid_name, routing_map, placement_map, placement_pattern, transform_pattern, name)
        tlib.append(temp)
    
    gen_list = [["ptap", ntap_ulvt], ["ntap", ptap_ulvt] ]
    for name, placement_map in gen_list:
        grid_name  = placement_map["grid"]
        
        temp       = TileTapTemplate( tlib, glib, grid_name, routing_map, placement_map, placement_pattern, transform_pattern, name)
        tlib.append(temp)

    return tlib

def load_templates():
    """Load template to a template library object"""
    tlib = laygo2.tech.tech_templates.load_templates(tech_fname, load_udf_templates )
    return tlib


def load_grids(templates):
    """Load template to a template library object"""
    tlib = laygo2.tech.tech_grids.load_grids(templates = templates, tech_fname = tech_fname)

    return tlib



