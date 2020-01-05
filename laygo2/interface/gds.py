#!/usr/bin/python
########################################################################################################################
#
# Copyright (c) 2014, Regents of the University of California
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
#   disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
#    following disclaimer in the documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################################################################

"""
This module implements interfaces with gds files.
Implemented by Eric Jan.

"""

__author__ = "Eric Jan"
__maintainer__ = "Jaeduk Han"
__status__ = "Prototype"

# TODO: Implement import functions (similar to load in python-gdsii)

import logging
import pprint

import numpy as np

import laygo2.object
from laygo2.interface.gds_helper import *


class Library:
    def __init__(self, version, name, physicalUnit, logicalUnit):
        """
        Initialize Library object

        Parameters
        ----------
        version : int
            GDSII file version. 5 is used for v5.
        name : str
            library name.
        physicalUnit : float
            physical resolution of the data.
        logicalUnit : float
            logical resolution of the data.
        """
        self.version = version
        self.name = name
        self.units = [logicalUnit, physicalUnit]
        self.structures = dict()
        assert physicalUnit > 0 and logicalUnit > 0

    def export(self, stream):
        """
        Export to stream

        Parameters
        ----------
        stream : stream
            file stream to be written
        """
        stream.write(pack_data("HEADER", self.version))
        stream.write(pack_bgn("BGNLIB"))
        stream.write(pack_data("LIBNAME", self.name))
        stream.write(pack_data("UNITS", self.units))
        for strname, struct in self.structures.items():
            struct.export(stream)
        stream.write(pack_no_data("ENDLIB"))

    def add_structure(self, name):
        """
        Add a structure object to library

        Parameters
        ----------
        name : str
            name of structure

        Returns
        -------
        laygo2.GDSIO.Structure
            created structure object
        """
        s=Structure(name)
        self.structures[name]=s
        return s

    def add_boundary(self, strname, layer, dataType, points):
        """
        Add a boundary object to specified structure

        Parameters
        ----------
        strname : str
            structure name to insert the boundary object
        layer : int
            layer name of the boundary object
        dataType : int
            layer purpose of the boundary object
        points : 2xn integer array list
            point array of the boundary object
            ex) [[x0, y0], [x1, x1], ..., [xn-1, yn-1], [x0, y0]]

        Returns
        -------
        laygo2.GDSIO.Boundary
            created boundary object
        """
        return self.structures[strname].add_boundary(layer, dataType, points)

    def add_path(self, strname, layer, dataType, points, width, pathtype=4, bgnextn=0, endextn=0):
        """
        Add a boundary object to specified structure

        Parameters
        ----------
        layer : int
            layer name
        dataType : int
            layer purpose
        points : list
            layer coordinates
        width : int
            the width of the path
        pathtype : int, optional
            the type of path. 0 for flushing endpoints, 1 for round-ended paths, 2 for half-square ended paths,
            4 for custom square-end extensions (set by bgnextn and endextn)

        Examples
        --------
        add_path('test', 50, 0, [[1000, 1000], [1000, 0], [0, 0], [0, 1000], [1000, 1000]], 10)

        Returns
        -------
        laygo2.GDSIO.Path
            generated path object
        """
        return self.structures[strname].add_path(layer, dataType, points, width, pathtype, bgnextn, endextn)

    def add_instance(self, strname, cellname, xy, transform='R0'):
        """
        Add an instance object to specified structure

        Parameters
        ----------
        strname : str
            structure name to insert the instance
        cellname : str
            instance cellname
        xy : [int, int]
            instance cooridnate
        transform : str
            transform parameter

        Returns
        -------
        laygo2.GDSIO.Instance
            created instance object
        """
        return self.structures[strname].add_instance(cellname, xy, transform)

    def add_instance_array(self, strname, cellname, n_col, n_row, xy, transform='R0'):
        """
        Add an instance array to specified structure

        Parameters
        ----------
        strname : str
            structure name to insert the instance
        cellname : str
            instance cellname
        n_col : int
            number of columns
        n_row : int
            number of rows
        xy : [int, int]
            instance coordinate
        transform : str
            transform parameter

        Returns
        -------
        laygo2.GDSIO.InstanceArray
            instance array object
        """
        return self.structures[strname].add_instance_array(cellname, n_col, n_row, xy, transform)

    def add_text(self, strname, layer, textType, xy, string, textHeight=100):
        """
        Add a text object to specified structure

        Parameters
        ----------
        strname : str
            structure name to insert the text object
        layer : int
            layer name of the text object
        textType : int
            layer purpose of the text object
        xy : [int, int]
            text coordinate
        string : str
            text string
        textHeight : int
            text height

        Returns
        -------
        laygo2.GDSIO.Text
            text object
        """
        return self.structures[strname].add_text(layer, textType, xy, string, textHeight)


class Structure(list):

    def __init__(self, name):
        """
        initialize Structure object

        Parameters
        ----------
        name : str
            structure name
        """
        list.__init__(self)
        self.name = name
        self.elements = []

    def export(self, stream):
        """
        Export to stream

        Parameters
        ----------
        stream : stream
            file stream to be written
        """
        stream.write(pack_bgn("BGNSTR"))
        stream.write(pack_data("STRNAME", self.name))
        for element in self.elements:
            element.export(stream)
        stream.write(pack_no_data("ENDSTR"))

    def add_boundary(self, layer, dataType, points):
        """
        Add a boundary object to structure

        Parameters
        ----------
        layer : int
            layer name
        dataType : int
            layer purpose
        points : list
            layer coordinates

        Examples
        --------
        add_boundary('test', 50, 0, [[1000, 1000], [1000, 0], [0, 0], [0, 1000], [1000, 1000]])

        Returns
        -------
        laygo2.GDSIO.Boundary
            generated boundary object
        """
        elem = Boundary(layer, dataType, points)
        self.elements.append(elem)
        return elem

    def add_path(self, layer, dataType, points, width, pathtype=4, bgnextn=0, endextn=0):
        """
        Add a boundary object to structure

        Parameters
        ----------
        layer : int
            layer name
        dataType : int
            layer purpose
        points : list
            layer coordinates
        width : int
            the width of the path
        pathtype : int, optional
            the type of path. 0 for flushing endpoints, 1 for round-ended paths, 2 for half-square ended paths,
            4 for custom square-end extensions (set by bgnextn and endextn)

        Examples
        --------
        add_path('test', 50, 0, [[1000, 1000], [1000, 0], [0, 0], [0, 1000], [1000, 1000]], 10)

        Returns
        -------
        laygo2.GDSIO.Path
            generated path object
        """
        elem = Path(layer, dataType, points, width, pathtype, bgnextn, endextn)
        self.elements.append(elem)
        return elem

    def add_instance(self, cellname, xy, transform='R0'):
        """
        Add an instance object to structure

        Parameters
        ----------
        cellname : str
            cell name
        xy : [int, int]
            xy coordinate
        transform : str
            transform parameter

        Returns
        -------
        laygo2.GDSIO.Instance
            generated instance object
        """
        elem = Instance(cellname, xy, transform)
        self.elements.append(elem)
        return elem

    def add_instance_array(self, cellname, n_col, n_row, xy, transform='R0'):
        """
        Add an instance array object to structure

        Parameters
        ----------
        cellname : str
            cell name
        n_col : int
            number of columns
        n_row : int
            number of rows
        xy : [int, int]
            xy coordinate
        transform : str
            transform parameter

        Examples
        --------
        new_lib.add_instance_array('test2', 'test', 2, 3, [[3000, 3000], [3000 + 2 * 2000, 3000], [3000, 3000 + 3 * 3000]])

        Returns
        -------
        laygo2.GDSIO.InstanceArray
            generated instance array object
        """
        elem = InstanceArray(cellname, n_col, n_row, xy, transform)
        self.elements.append(elem)
        return elem

    def add_text(self, layer, textType, xy, string, textHeight=100):
        """
        Add a text object to structure

        Parameters
        ----------
        layer : int
            layer name
        textType : int
            layer purpose
        xy : list
            xy coordinate
        string : str
            text string
        textHeight : int
            text height

        Returns
        -------
        laygo2.GDSIO.Text
            generated text object
        """
        elem = Text(layer, textType, xy, string, textHeight)
        self.elements.append(elem)
        return elem


class Element:
    """Base class for GDSIO objects"""
    possible_transform_parameters = {'R0': (None, None),
                                     'R90': (0, 90),
                                     'R180': (0, 180),
                                     'R270': (0, 270),
                                     'MX': (32768, 0),
                                     'MY': (32768, 180)
                                    }
    """dict: transform parameter dictionary"""

    def set_transform_parameters(self, transform):
        """
        initialize transform parameters

        Parameters
        ----------
        transform : str
            transform parameter,
            'R0' : default, no transform,
            'R90' : rotate by 90-degree,
            'R180' : rotate by 180-degree,
            'R270' : rotate by 270-degree,
            'MX' : mirror across X axis,
            'MY' : mirror across Y axis
        """
        if transform not in self.possible_transform_parameters:
            raise Exception("enter a viable transform parameter\npossible_transform_parameters = ['R0', 'R90', 'R180', 'R270', 'MX', 'MY']")
        self.strans, self.angle = self.possible_transform_parameters[transform]


class Boundary (Element):
    """Boundary object for GDSIO"""

    def __init__(self, layer, dataType, points):
        """
        initialize Boundary object

        Parameters
        ----------
        layer : int
            Layer id
        dataType : int
            Layer purpose
        points : list
            xy coordinates for Boundary object
        """
        if len(points) < 2:
            raise Exception("not enough points")
        if len(points) >= 2 and points[0] != points[len(points) - 1]:
            raise Exception("start and end points different")
        temp_xy = []
        for point in points:
            if len(point) != 2:
                raise Exception("error for point input: " + str(point))
            temp_xy += point
        self.layer = layer
        self.dataType = dataType
        self.xy = list(temp_xy)

    def export(self, stream):
        """
        Export to stream

        Parameters
        ----------
        stream : stream
            File stream to be written
        """
        stream.write(pack_no_data("BOUNDARY"))
        stream.write(pack_data("LAYER", self.layer))
        stream.write(pack_data("DATATYPE", self.dataType))
        stream.write(pack_data("XY", self.xy))
        stream.write(pack_no_data("ENDEL"))


class Path (Element):
    """Path object for GDSIO"""

    def __init__(self, layer, dataType, points, width, pathtype, bgnextn, endextn):
        """
        initialize Boundary object

        Parameters
        ----------
        layer : int
            Layer id
        dataType : int
            Layer purpose
        points : list
            xy coordinates for Path object
        width : int
            the width of the path
        pathtype : int, optional
            the type of path. 0 for flushing endpoints, 1 for round-ended paths, 2 for half-square ended paths,
            4 for custom square-end extensions (set by bgnextn and endextn)
        """
        if len(points) < 2:
            raise Exception("not enough points")
        temp_xy = []
        for point in points:
            if len(point) != 2:
                raise Exception("error for point input: " + str(point))
            temp_xy += point
        self.layer = layer
        self.dataType = dataType
        self.xy = list(temp_xy)
        self.width = width
        self.pathtype = pathtype
        self.bgnextn = bgnextn
        self.endextn = endextn

    def export(self, stream):
        """
        Export to stream

        Parameters
        ----------
        stream : stream
            File stream to be written
        """
        stream.write(pack_no_data("PATH"))
        stream.write(pack_data("LAYER", self.layer))
        stream.write(pack_data("DATATYPE", self.dataType))
        stream.write(pack_data("PATHTYPE", self.pathtype))
        stream.write(pack_data("WIDTH", self.width))
        stream.write(pack_data("BGNEXTN", self.bgnextn))
        stream.write(pack_data("ENDEXTN", self.endextn))
        stream.write(pack_data("XY", self.xy))
        stream.write(pack_no_data("ENDEL"))


class Instance (Element):
    """Instance object for GDSIO"""

    def __init__(self, sname, xy, transform='R0'):
        """
        initialize Instance object

        Parameters
        ----------
        sname : str
            Instance name
        xy : array
            xy coordinate of Instance Object
        transform : str
            transform parameter,
            'R0' : default, no transform,
            'R90' : rotate by 90-degree,
            'R180' : rotate by 180-degree,
            'R270' : rotate by 270-degree,
            'MX' : mirror across X axis,
            'MY' : mirror across Y axis
        """
        Element.__init__(self)
        self.sname = sname
        l = len(xy)
        if l > 1:
            raise Exception("too many points provided\ninstance should only be located at one point")
        elif l < 1:
            raise Exception("no point provided\ninstance should be located at a point")
        self.xy = list(xy[0])
        Element.set_transform_parameters(self, transform)

    def export(self, stream):
        """
        Export to stream

        Parameters
        ----------
        stream : stream
            File stream to be written
        """
        stream.write(pack_no_data("SREF"))
        stream.write(pack_data("SNAME", self.sname))
        pack_optional("STRANS", self.strans, stream)
        pack_optional("ANGLE", self.angle, stream)
        stream.write(pack_data("XY", self.xy))
        stream.write(pack_no_data("ENDEL"))


class InstanceArray (Element):
    """InstanceArray object for GDSIO"""

    def __init__(self, sname, n_col, n_row, xy, transform='R0'):
        """
        Initialize Instance Array object

        Parameters
        ----------
        sname : str
            InstanceArray name
        n_col: int
            Number of columns
        n_row : int
            Number of rows
        xy : array
            xy coordinates for InstanceArray Object,
            should be organized as: [(x0, y0), (x0+n_col*sp_col, y_0), (x_0, y0+n_row*sp_row)]
        transform : str
            Transform parameter,
            'R0' : default, no transform,
            'R90' : rotate by 90-degree,
            'R180' : rotate by 180-degree,
            'R270' : rotate by 270-degree,
            'MX' : mirror across X axis,
            'MY' : mirror across Y axis
        """
        l = len(xy)
        if l != 3:
            s = "\nxy: [(x0, y0), (x0+n_col*sp_col, y_0), (x_0, y0+n_row*sp_row)]"
            if l > 3:
                s = "too many points provided" + s
            else:
                s = "not enough points provided" + s
            raise Exception(s)
        self.sname = sname
        self.colrow = [n_col, n_row]
        temp_xy = []
        for point in xy:
            if len(point) != 2:
                raise Exception("error for point input: " + str(point))
            temp_xy += point
        self.xy = list(temp_xy)
        # self.xy = [num for pt in xy for num in pt]
        Element.set_transform_parameters(self, transform)

    def export(self, stream):
        """
        Export to stream

        Parameters
        ----------
        stream : stream
            File stream to be written
        """
        stream.write(pack_no_data("AREF"))
        stream.write(pack_data("SNAME", self.sname))
        pack_optional("STRANS", self.strans, stream)
        pack_optional("ANGLE", self.angle, stream)
        stream.write(pack_data("COLROW", self.colrow))
        stream.write(pack_data("XY", self.xy))
        stream.write(pack_no_data("ENDEL"))


class Text (Element):
    """Text object for GDSIO"""

    def __init__(self, layer, textType, xy, string, textHeight=100):
        """
        Initialize Text object

        Parameters
        ----------
        layer : int
            Layer id
        textType : int
            I'm not really sure what this is
        xy : array
            xy coordinates for Text Object
        string : str
            Text object display string
        """
        l = len(xy)
        if l > 1:
            raise Exception("too many points provided\ninstance should only be located at one point")
        elif l < 1:
            raise Exception("no point provided\ninstance should be located at a point")
        self.layer = layer
        self.textType = textType
        self.xy = xy[0]
        self.string = string
        self.strans = 0
        self.mag = textHeight

    def export(self, stream):
        """
        Export to stream

        Parameters
        ----------
        stream : stream
            File stream to be written
        """
        stream.write(pack_no_data("TEXT"))
        stream.write(pack_data("LAYER", self.layer))
        stream.write(pack_data("TEXTTYPE", self.textType))
        stream.write(pack_data("STRANS", self.strans))
        #stream.write(pack_data("ANGLE", self.angle))
        stream.write(pack_data("MAG", self.mag))
        stream.write(pack_data("XY", self.xy))
        stream.write(pack_data("STRING", self.string))
        stream.write(pack_no_data("ENDEL"))


#export functions
def export_from_laygo(db, filename, cellname=None, scale = 1e-9, layermapfile="default.layermap",
                      physical_unit=1e-9, logical_unit=0.001, pin_label_height=0.1,
                      pin_annotate_layer=['text', 'drawing'], text_height=0.1,
                      abstract_instances = False, abstract_instances_layer = ['prBoundary', 'drawing']):
    """
    Export specified laygo2 object(s) to a GDS file

    Parameters
    ----------
    db : laygo.object.database.Library
        a library object that designs to be exported.
    filename : str
        the name of the output file.
    cellname : list or str or None.
        the name of cells to be exported. If None, all cells in the libname are exported.
    scale : float
        the scaling factor that converts integer coordinates to actual ones (mostly 1e-6, to convert 1 to 1um).
    layermapfile : str
        the name of layermap file.
    physical_unit : float, optional
        GDS physical unit.
    logical_unit : float, optional
        GDS logical unit.
    pin_label_height : float, optional
        the height of pin label.
    pin_annotate_layer : [str, str], optional
        pin annotate layer name (used when pinname is different from netname).
    text_height : float, optional
        the height of text
    """
    scl = round(1/scale*physical_unit/logical_unit)
    # 1um in phy
    # 1um/1nm = 1000 in laygo2 if scale = 1e-9 (1nm)
    # 1000/1nm*1nm/0.001 = 1000000 in gds if physical_unit = 1e-9 (1nm) and logical_unit = 0.001
    layermap = load_layermap(layermapfile)  # load layermap information
    logging.debug('ExportGDS: Library:' + db.name)
    lib_export = Library(5, str.encode(db.name), physical_unit, logical_unit)

    cellname = db.keys() if cellname is None else cellname  # export all cells if cellname is not given.
    cellname = [cellname] if isinstance(cellname, str) else cellname # convert to a list for iteration.
    for sn in cellname:
        s = db[sn]
        logging.debug('ExportGDS: Structure:' + sn)
        s_export = lib_export.add_structure(sn)
        # export objects
        for objname, obj in s.items():
            _convert_laygo_object(objname=objname, obj=obj, scl=scl, layermap=layermap,
                                  lib_export=lib_export, sn=sn, pin_label_height=pin_label_height,
                                  pin_annotate_layer=pin_annotate_layer, text_height=text_height,
                                  abstract_instances=abstract_instances,
                                  abstract_instances_layer=abstract_instances_layer)
        with open(filename, 'wb') as stream:
            lib_export.export(stream)


def export(db, filename, cellname=None, scale = 1e-9, layermapfile="default.layermap",
           physical_unit=1e-9, logical_unit=0.001, pin_label_height=0.1,
           pin_annotate_layer=['text', 'drawing'], text_height=0.1,
           abstract_instances=False, abstract_instances_layer=['prBoundary', 'drawing']):
    """See laygo2.interface.gds.export_from_laygo for details."""
    export_from_laygo(db, filename, cellname, scale, layermapfile,
                      physical_unit, logical_unit, pin_label_height,
                      pin_annotate_layer, text_height,
                      abstract_instances, abstract_instances_layer)


def _convert_laygo_object(objname, obj, scl, layermap, lib_export, sn, pin_label_height=0.1,
                          pin_annotate_layer=['text', 'drawing'], text_height=0.1,
                          abstract_instances=False, abstract_instances_layer=['prBoundary', 'drawing']):
    """Convert laygo objects to gds objects.

    virtual_instance_type: str
        instance
        uniquified_instance
        flattened

    """
    # TODO: this code is not very readable. Refactor it.

    if obj.__class__ == laygo2.object.Rect:
        xy = obj.xy * scl
        hext = obj.hextension * scl  # extensions for routing wires.
        vext = obj.vextension * scl
        bx1, bx2 = sorted(xy[:, 0].tolist())  # really need to sort the coordinates?
        by1, by2 = sorted(xy[:, 1].tolist())
        ll = np.array([bx1, by1])  # lower-left
        ur = np.array([bx2, by2])  # upper-right
        _xy = np.vstack([ll,ur])
        c = [[round(_xy[0][0]-hext), round(_xy[0][1]-vext)], [round(_xy[0][0]-hext), round(_xy[1][1]+vext)],
             [round(_xy[1][0]+hext), round(_xy[1][1]+vext)], [round(_xy[1][0]+hext), round(_xy[0][1]-vext)],
             [round(_xy[0][0]-hext), round(_xy[0][1]-vext)]]  # build list
        l = layermap[obj.layer[0]][obj.layer[1]]
        lib_export.add_boundary(sn, l[0], l[1], c)
        logging.debug('ExportGDS: Rect:' + objname + ' layer:' + str(l) + ' xy:' + str(c))
    elif obj.__class__ == laygo2.object.Path:
        xy = obj.xy * scl
        width = obj.width * scl
        extn = obj.extension * scl
        l = layermap[obj.layer[0]][obj.layer[1]]
        lib_export.add_path(sn, l[0], l[1], xy.tolist(), width, pathtype=4, bgnextn=extn, endextn=extn)
        logging.debug('ExportGDS: Path:' + objname + ' layer:' + str(l) + ' xy:' + str(xy))
    elif obj.__class__ == laygo2.object.Pin:
        if obj.elements is None:
            _objelem = [obj]
        else:
            _objelem = obj.elements
        for idx, _obj in np.ndenumerate(_objelem):
            xy = _obj.xy * scl
            bx1, bx2 = sorted(xy[:,0].tolist())  # again, let's check this.
            by1, by2 = sorted(xy[:,1].tolist())
            ll = np.array([bx1, by1])  # lower-left
            ur = np.array([bx2, by2])  # upper-right
            _xy = np.vstack([ll,ur])
            c = [[round(_xy[0][0]), round(_xy[0][1])], [round(_xy[0][0]), round(_xy[1][1])],
                 [round(_xy[1][0]), round(_xy[1][1])], [round(_xy[1][0]), round(_xy[0][1])],
                 [round(_xy[0][0]), round(_xy[0][1])]]  # build list
            l = layermap[_obj.layer[0]][_obj.layer[1]]
            lib_export.add_boundary(sn, l[0], l[1], c)
            lib_export.add_text(sn, l[0], l[1], [[(_xy[0][0]+_xy[1][0])//2, (_xy[0][1]+_xy[1][1])//2]],
                                string=_obj.netname, textHeight=pin_label_height * scl)
            if not _obj.name == _obj.netname:  # if netname is different from pinname, create an annotate text
                if _obj.name is not None:
                    l_ann = layermap[pin_annotate_layer[0]][pin_annotate_layer[1]]
                    lib_export.add_text(sn, l_ann[0], l_ann[1],
                                        [[(_xy[0][0]+_xy[1][0])//2, (_xy[0][1]+_xy[1][1])//2]],
                                        string=_obj.name, textHeight=pin_label_height * scl)
            logging.debug('ExportGDS: Pin:' + objname + ' net:' + _obj.netname + ' layer:' + str(l) + ' xy:' + str(c))
    elif obj.__class__ == laygo2.object.physical.Text:
        xy = obj.xy * scl
        l = layermap[obj.layer[0]][obj.layer[1]]
        _xy = [round(_xy0) for _xy0 in xy]
        lib_export.add_text(sn, l[0], l[1], [_xy], string=obj.text, textHeight=round(text_height * scl))
        logging.debug('ExportGDS: Text:' + objname + ' text:' + obj.text + ' layer:' + str(l) + ' xy:' + str(_xy))
    elif obj.__class__ == laygo2.object.Instance:
        _convert_laygo_object_instance(lib_export, sn, objname, obj, scl, abstract_instances, abstract_instances_layer,
                                       layermap)
    elif obj.__class__ == laygo2.object.VirtualInstance:  # virtual instance
        virt_struc_name = sn + '_VirtualInst_' + objname
        s_virt = lib_export.add_structure(virt_struc_name)
        for en, e in obj.native_elements.items():
            _convert_laygo_object(objname=objname+'_'+en, obj=e, scl=scl, layermap=layermap, lib_export=lib_export,
                                  sn=virt_struc_name, pin_label_height=pin_label_height, pin_annotate_layer=pin_annotate_layer,
                                  text_height=text_height, abstract_instances=abstract_instances,
                                  abstract_instances_layer=abstract_instances_layer)
        xy = obj.xy * scl
        xyl = xy.tolist()
        if np.array_equal(obj.shape, np.array([1, 1])) or (obj.shape is None):  # single instance
            lib_export.add_instance(sn, virt_struc_name, [xyl], obj.transform)
            logging.debug('ExportGDS: VirtualInstance:' + objname + ' cellname:' + obj.cellname + ' xy:' + str(xy))
        else:  # mosaic
            xy_mosaic = [[round(xyl[0]), round(xyl[1])],
                         [round(xyl[0] + obj.shape[0] * (obj.spacing[0] * scl)), round(xyl[1])],
                         [round(xyl[0]), round(xyl[1] + obj.shape[1] * (obj.spacing[1] * scl))]]

            lib_export.add_instance_array(sn, virt_struc_name, obj.shape[0], obj.shape[1], xy_mosaic,
                                          obj.transform)
            logging.debug('ExportGDS: VirtualInstance:' + objname + ' cellname:' + obj.cellname + ' xy:' + str(xy_mosaic)
                 + ' shape:' + str(obj.shape.tolist()) + ' spacing:' + str(obj.spacing.tolist()))


def _convert_laygo_object_instance(lib_export, sn, objname, obj, scl, abstract_instances, abstract_instances_layer, layermap):
    """Internal function of the instance conversion."""
    if abstract_instances:  # export abstract
        _xy = obj.bbox * scl
        c = [[round(_xy[0][0]), round(_xy[0][1])], [round(_xy[0][0]), round(_xy[1][1])],
             [round(_xy[1][0]), round(_xy[1][1])], [round(_xy[1][0]), round(_xy[0][1])],
             [round(_xy[0][0]), round(_xy[0][1])]]  # build list
        l = layermap[abstract_instances_layer[0]][abstract_instances_layer[1]]
        lib_export.add_boundary(sn, l[0], l[1], c)
    else:
        xy = obj.xy * scl
        xyl = xy.tolist()
        if np.array_equal(obj.shape, np.array([1, 1])) or (obj.shape is None):  # single instance
            lib_export.add_instance(sn, obj.cellname, [xyl], obj.transform)
            logging.debug('ExportGDS: Instance:' + objname + ' cellname:' + obj.cellname + ' xy:' + str(xy))
        else:  # mosaic
            xy_mosaic = [[round(xyl[0]), round(xyl[1])],
                         [round(xyl[0] + obj.shape[0] * (obj.spacing[0] * scl)), round(xyl[1])],
                         [round(xyl[0]), round(xyl[1] + obj.shape[1] * (obj.spacing[1] * scl))]]

            lib_export.add_instance_array(sn, obj.cellname, obj.shape[0], obj.shape[1], xy_mosaic,
                                          obj.transform)
            logging.debug('ExportGDS: Instance:' + objname + ' cellname:' + obj.cellname + ' xy:' + str(xy_mosaic)
                          + ' shape:' + str(obj.shape.tolist()) + ' spacing:' + str(obj.spacing.tolist()))


# TODO: implement export_to_laygo function.

# test
if __name__ == '__main__':
    test_raw = True
    test_laygo = True
    # Test1 - creating a GDS file using raw access functions.
    if test_raw:
        # Create a new library
        new_lib = Library(5, b'MYLIB', 1e-9, 0.001)
        # Add a new structure to the new library
        struc = new_lib.add_structure('test')
        # Add a boundary object
        new_lib.add_boundary('test', 50, 0, [[0, 0], [0, 100000], [100000, 100000], [100000, 0], [0, 0]])
        # Add a path object
        new_lib.add_path('test', 50, 0, [[0, 0], [0, 100000], [100000, 100000], [100000, 0], [0, 0]], 50000, 4, 10000, 20000)
        # Add a new structure to the new library
        struc2 = new_lib.add_structure('test2')
        # Add an instance
        new_lib.add_instance('test2', 'test', [[0, 0]])
        # Add an array instance
        new_lib.add_instance_array('test2', 'test', 2, 3,
                                   [[300000, 300000], [300000 + 2 * 200000, 300000], [300000, 300000 + 3 * 300000]])
        # rotations
        # original Instance
        new_lib.add_instance('test2', 'test', [[0, -200000]])
        # rotate by 90
        new_lib.add_instance('test2', 'test', [[200000, -200000]], "R90")  # ANGLE 90, STRANS 0
        # rotate by 180
        new_lib.add_instance('test2', 'test', [[400000, -200000]], "R180")  # 180, 0
        # rotate by 270
        new_lib.add_instance('test2', 'test', [[600000, -200000]], "R270")  # 270, 0
        # mirror across x-axis
        new_lib.add_instance('test2', 'test', [[800000, -500000]], "MX")  # 0, 32768
        # mirror across y-axis
        new_lib.add_instance('test2', 'test', [[1000000, -500000]], "MY")  # 180, 32768
        # Add a text object
        new_lib.add_text('test2', 45, 0, [[0, 0]], 'mytext')

        # Export to a GDS file
        with open('GDS_raw_test1.gds', 'wb') as stream:
            new_lib.export(stream)

        # Import the GDS file back and display
        with open('GDS_raw_test1.gds', 'rb') as stream:
            pprint.pprint(readout(stream, scale=1))

    # Test2 - creating a GDS file from laygo2 object and export_from_laygo function.
    if test_laygo:
        import laygo2.object
        lib0 = laygo2.object.Library(name='MYLIB')
        dsn0 = laygo2.object.Design(name='test')
        rect0 = laygo2.object.Rect(name='R0', xy=[[0, 0], [100, 100]], layer=['M1', 'drawing'])
        dsn0.append(rect0)
        path0 = laygo2.object.Path(name='PT0', xy=[[0, 0], [0, 100], [100, 100], [100, 0], [0, 0]], width=50,
                                   extension=20, layer=['M1', 'drawing'])
        dsn0.append(path0)
        lib0.append(dsn0)
        dsn1 = laygo2.object.Design(name='test2')
        inst0 = laygo2.object.Instance(name='I0', xy=[0, 0], libname='MYLIB', cellname='test', transform='R0')
        dsn1.append(inst0)
        inst1 = laygo2.object.Instance(name='I1', xy=[300, 300], libname='MYLIB', cellname='test', shape=[2, 3],
                                       pitch=[200, 300], transform='R0')
        dsn1.append(inst1)
        inst2 = laygo2.object.Instance(name='I2', xy=[0, -200], libname='MYLIB', cellname='test', transform='R0')
        dsn1.append(inst2)
        inst3 = laygo2.object.Instance(name='I3', xy=[200, -200], libname='MYLIB', cellname='test', transform='R90')
        dsn1.append(inst3)
        inst4 = laygo2.object.Instance(name='I4', xy=[400, -200], libname='MYLIB', cellname='test', transform='R180')
        dsn1.append(inst4)
        inst5 = laygo2.object.Instance(name='I5', xy=[600, -200], libname='MYLIB', cellname='test', transform='R270')
        dsn1.append(inst5)
        inst6 = laygo2.object.Instance(name='I6', xy=[800, -500], libname='MYLIB', cellname='test', transform='MX')
        dsn1.append(inst6)
        inst7 = laygo2.object.Instance(name='I7', xy=[1000, -500], libname='MYLIB', cellname='test', transform='MY')
        dsn1.append(inst7)
        text0 = laygo2.object.Text(name='T0', xy=[0, 0], layer=['text', 'drawing'], text='mytext')
        dsn1.append(text0)
        inst8_pins = dict()
        inst8_pins['in'] = laygo2.object.Pin(xy=[[0, 0], [10, 10]], layer=['M1', 'drawing'], netname='in')
        inst8_pins['out'] = laygo2.object.Pin(xy=[[90, 90], [100, 100]], layer=['M1', 'drawing'], netname='out')
        inst8_native_elements = dict()
        inst8_native_elements['R0'] = laygo2.object.Rect(xy=[[0, 0], [10, 10]], layer=['M1', 'drawing'])
        inst8_native_elements['R1'] = laygo2.object.Rect(xy=[[90, 90], [100, 100]], layer=['M1', 'drawing'])
        inst8_native_elements['R2'] = laygo2.object.Rect(xy=[[0, 0], [100, 100]], layer=['prBoundary', 'drawing'])
        inst8 = laygo2.object.VirtualInstance(name='I8', xy=[500, 500], native_elements=inst8_native_elements,
                                              shape=[3, 2], pitch=[100, 100], unit_size=[100, 100], pins=inst8_pins,
                                              transform='R0')
        dsn1.append(inst8)
        lib0.append(dsn1)

        # Export to a GDS file
        export_from_laygo(lib0, filename='GDS_raw_test2.gds', cellname=None, scale = 1e-9,
                          layermapfile="gds_default.layermap", physical_unit=1e-9, logical_unit=0.001, pin_label_height=0.1,
                          pin_annotate_layer=['text', 'drawing'], text_height=0.1)

        # Import the GDS file back and display
        with open('GDS_raw_test2.gds', 'rb') as stream:
            pprint.pprint(readout(stream, scale=1e-9))


