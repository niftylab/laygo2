import laygo2.object.database
import laygo2.object.grid

import numpy as np
import yaml
import pprint

# Grid library for the advanced example technology (advtech).

# Technology parameters
if __name__ == '__main__':
    tech_fname = './technology_example.yaml'
else:
    tech_fname = '../test_tech/technology_example.yaml'
with open(tech_fname, 'r') as stream:
    try:
        tech_params = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


# Grid library
def load_grids(templates):
    """
    Load grids to a grid library object.

    Parameters
    ----------
    templates: laygo2.object.database.TemplateLibrary
        The template library object that contains via templates.
    """
    glib = laygo2.object.database.GridLibrary(name='advtech_grids')
    for gname, grid in tech_params['grids'].items():
        gv = laygo2.object.grid.OneDimGrid(name=gname + '_v', scope=grid['vertical']['scope'],
                                           elements=grid['vertical']['elements'])
        gh = laygo2.object.grid.OneDimGrid(name=gname + '_h', scope=grid['horizontal']['scope'],
                                           elements=grid['horizontal']['elements'])
        if grid['type'] == 'placement':  # placement grid
            g = laygo2.object.grid.PlacementGrid(name=gname, vgrid=gv, hgrid=gh)
            glib.append(g)
        elif grid['type'] == 'routing':  # routing grid
            vwidth = laygo2.object.grid.CircularMapping(elements=grid['vertical']['width'])
            hwidth = laygo2.object.grid.CircularMapping(elements=grid['horizontal']['width'])
            vextension = laygo2.object.grid.CircularMapping(elements=grid['vertical']['extension'])
            hextension = laygo2.object.grid.CircularMapping(elements=grid['horizontal']['extension'])
            vlayer = laygo2.object.grid.CircularMapping(elements=grid['vertical']['layer'], dtype=object)
            hlayer = laygo2.object.grid.CircularMapping(elements=grid['horizontal']['layer'], dtype=object)
            pin_vlayer = laygo2.object.grid.CircularMapping(elements=grid['vertical']['pin_layer'], dtype=object)
            pin_hlayer = laygo2.object.grid.CircularMapping(elements=grid['horizontal']['pin_layer'], dtype=object)
            xcolor = list()
            if 'xcolor' in grid['vertical'].keys():
                if grid['vertical']['xcolor'] == 'not MPT':
                    xcolor = ['not MPT']
                else:
                    xcolor.append(grid['vertical']['xcolor'])
            else:
                xcolor = ['not MPT']

            ycolor = list()
            if 'ycolor' in grid['horizontal'].keys():
                if grid['horizontal']['ycolor'] == 'not MPT':
                    ycolor = ['not MPT']
                else:
                    ycolor.append(grid['horizontal']['ycolor'])
            else:
                ycolor = ['not MPT']
            primary_grid = grid['primary_grid']
            # Create the via map defined by the yaml file.
            vmap_original = grid['via']['map']  # viamap defined in the yaml file.
            vmap_mapped = list()  # map template objects to the via map.
            for vmap_org_row in vmap_original:
                vmap_mapped_row = []
                for vmap_org_elem in vmap_org_row:
                    vmap_mapped_row.append(templates[vmap_org_elem])
                vmap_mapped.append(vmap_mapped_row)
            viamap = laygo2.object.grid.CircularMappingArray(elements=vmap_mapped, dtype=object)
            g = laygo2.object.grid.RoutingGrid(name=gname, vgrid=gv, hgrid=gh,
                                               vwidth=vwidth, hwidth=hwidth,
                                               vextension=vextension, hextension=hextension,
                                               vlayer=vlayer, hlayer=hlayer,
                                               pin_vlayer=pin_vlayer, pin_hlayer=pin_hlayer,
                                               viamap=viamap, primary_grid=primary_grid,
                                               xcolor=xcolor, ycolor=ycolor
                                               )
            glib.append(g)
    return glib

