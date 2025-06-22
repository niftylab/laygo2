"""
This module implements interfaces with external cad environments.
"""
from . import bag, mpl, yaml 
from ..object.database import Library

# functions added for compatibilities with laygo3 
def export(
    # Common parameters
    db: "laygo2.object.database.Library" or "laygo2.object.database.Design",
    cellname: str = None,
    target = None,
    tech: "laygo2.object.technology.BaseTechnology" = None,
    filename: str = None,
    scale = None,
    # BAG parameters
    reset_library = None,
    tech_library = None,
    # MPL and Bokeh parameters
    colormap: dict = None,
    order = None,
    xlim: list = None,
    ylim: list = None,
    show: bool = None,
    annotate_grid = None,
    export_to_webserver: bool = False,
):
    """
    Export the database to the target environment specified by target or laygo2_tech.

    This function exports the design database to various target environments such as bag or matplotlib(mpl).
    The target environment can be specified directly or inferred from the technology parameters.

    Parameters
    ----------
    db : laygo2.object.database.Library
        The design database to be exported.
    cellname : str, optional
        The name of the cell to be exported. If None, the entire database is exported.
    target : str, optional
        The target environment for export (such as bag or mpl). 
        If None, the default target from the technology parameters is used.
    tech : object, optional
        The technology object containing technology-specific parameters.
    filename : str, optional
        The name of the file to save the exported data. If None, a default name is used.
    scale : float, optional
        The scale factor for the exported data.
    reset_library : bool, optional
        BAG-specific parameter. If True, the library is reset before export.
    tech_library : str, optional
        BAG-specific parameter. The name of the technology library.
    colormap : dict, optional
        MPL-specific parameter. A dictionary defining the colormap for the export.
    order : list, optional
        MPL-specific parameter. The order in which to export the elements.
    xlim : list, optional
        MPL-specific parameter. The x-axis limits for the export.
    ylim : list, optional
        MPL-specific parameter. The y-axis limits for the export.
    show : bool, optional
        MPL-specific parameter. If True, the exported data is displayed.
    annotate_grid : list of laygo2.object.physical.Grid, optional
        MPL-specific parameter. A list of grids to annotate in the export.
    export_to_webserver : bool, optional
        If True, the exported data is exported to the webserver (for bokeh only).

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the target environment is not supported.

    Notes
    -----
    The function determines the target environment based on the provided parameters and exports
    the design database accordingly. If no target is specified, the default target from the
    technology parameters is used.

    Example
    -------
    >>> import laygo2
    >>> # Create a design database.
    >>> db = laygo2.Library(name='mylib')
    >>> # Export the database to GDS.
    >>> laygo2.export(db, cellname='mycell', target='gdspy', filename='mycell.gds', scale=0.001)
    """

    if db.__class__.__name__ == "Design":  # single design export
        _db = Library(name=db.libname)
        _db.append(db)
        db = _db

    tech_params = getattr(tech, "tech_params", None)
    if tech_params is None:
        raise Exception("Valid technology object is 'not' imported to template_exporter")
#    tech_params = tech.tech_params
    # if tech is not None:
    #     tech_params = tech.tech_params
    # else:
    #     tech_params = {}
    target = tech_params.get('export',{}).get('target','bag') # tech_params['export']['target']
    # if target is None:  # use the default tech
    #     if 'export' in tech_params:
    #         target = tech_params.get('export',{}).get('target','bag') # tech_params['export']['target']
    #     else:
    #         target = 'bag'  # use bag export if nothing is specified for target.

    # export functions
    if target == 'bag':  # bag export
        params_bag = tech_params.get('export',{}).get('bag',{})
        if scale is None:
            scale = params_bag.get('scale', 1) # tech_params['export']['bag']['scale']
        if reset_library is None:
            reset_library = params_bag.get('reset_library', False) # tech_params['export']['bag']['reset_library']
        if tech_library is None:
            tech_library = params_bag.get('reset_library', 'GPDK') # tech_params['export']['bag']['tech_library']
        return bag.export(
            db=db,
            filename=filename,
            cellname=cellname,
            scale=scale,
            reset_library=reset_library,
            tech_library=tech_library
        )
    elif (target == 'mpl') or (target == 'bokeh'):  # mpl export
        params_mpl = tech_params.get('export',{}).get('mpl',{})
        if colormap is None:
            colormap = params_mpl.get('colormap', None) # tech_params['export']['mpl']['colormap']
        if order is None:
            order = params_mpl.get('order', None) # tech_params['export']['mpl']['order']
        if show is None:
            show = params_mpl.get('show', False)
            # if 'show' in tech_params['export']['mpl']:
            #     show = tech_params['export']['mpl']['show']
            # else:
            #     show = False
        
        if filename is not None:
            if filename.endswith('.png'):
                filename = filename[:-4]
            filename = filename + '.png'

        return mpl.export(
            db=db,
            colormap=colormap,
            order=order,
            xlim=xlim,
            ylim=ylim,
            show=show,
            filename=filename,
            annotate_grid=annotate_grid
        )
    else:
        raise Exception("output file name is not specified")

        

def export_template(
        template:"laygo2.object.template.Template", 
        tech: "laygo2.object.technology.BaseTechnology" = None,
        filename:str=None, 
        mode:str='append',
        export_mask:bool=None,
        export_prelvs_info:bool=None,
        export_internal_shapes:bool=None,
        ):
    """Export a template to a yaml file.

    Parameters
    ----------
    template: laygo2.object.template.Template or laygo2.object.database.Design
        The template object to be exported. If laygo2.Design is given, it is converted to a template and exported.
    tech: laygo2.object.technology.BaseTechnology
        The technology object that the laygo can refer to for exporting templates.
    filename: str, optional
        The name of the yaml file.
    mode: str, optional
        If 'append', it adds the template entry without erasing the original file.
    export_mask: bool, optional
        If true, mask information is exported.
    export_prelvs_info: bool, optional
        If true, information for prelvs is exported.
    export_internal_shapes: bool, optional
        If true, internal shape parameters are exported. 
    """
    tech_params = getattr(tech, "tech_params", {})
    
    yaml.export_template(template=template, filename=filename, mode=mode)
