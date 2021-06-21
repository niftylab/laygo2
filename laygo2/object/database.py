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
This module implements classes for various database objects.
"""

__author__ = "Jaeduk Han"
__maintainer__ = "Jaeduk Han"
__status__ = "Prototype"

import laygo2.object
import numpy as np


class BaseDatabase:
    """class that contains iterable contents."""

    name = None
    """str: the name of the object."""

    params = None
    """dict or None: a dictionary that contains parameters of the design. """

    elements = None
    """dict: the iterable elements of the design."""

    noname_index = 0
    """int: the suffix index of NoName objects (objects without its name)."""

    @property
    def keys(self):
        """Returns a list that contains keys of the elements of this object."""
        return self.elements.keys

    def items(self):
        """Matches to the items() function of its elements."""
        return self.elements.items()

    def __getitem__(self, pos):
        """Returns its sub-elements based on pos parameter."""
        return self.elements[pos]

    def __setitem__(self, key, item):
        item.name = key
        self.append(item)

    def append(self, item):
        if isinstance(item, list) or isinstance(item, np.ndarray):
            item_name_list = []
            item_list = []
            for i in item:
                _item_name, _item = self.append(i)
                item_name_list.append(_item_name)
                item_list.append(_item)
            return item_name_list, item_list
            #return [i[0] for i in item_list], [i[1] for i in item_list]
        else:
            item_name = item.name
            if item_name is None:  # NoName object. Put a name on it.
                while 'NoName_'+str(self.noname_index) in self.elements.keys():
                    self.noname_index += 1
                item_name = 'NoName_' + str(self.noname_index)
                self.noname_index += 1
            errstr = item_name + " cannot be added to " + self.name + ", as a child object with the same name exists."
            if item_name in self.elements.keys():
                raise KeyError(errstr)
            else:
                if item_name in self.elements.keys():
                    raise KeyError(errstr)
                else:
                    self.elements[item_name] = item
            return item_name, item

    def __iter__(self):
        """Iterator function. Directly mapped to its elements."""
        return self.elements.__iter__()

    def __str__(self):
        return self.summarize()

    def summarize(self):
        """Returns the summary of the object information."""
        return self.__repr__() + " " + \
               "name: " + self.name + ", " + \
               "params: " + str(self.params) + " \n" \
               "    elements: " + str(self.elements) + \
               ""

    def __init__(self, name, params=None, elements=None):
        """
        Constructor.

        Parameters
        ----------
        name : str
            The name of the object.
        """
        self.name = name
        self.params = params

        self.elements = dict()
        if elements is not None:
            for e in elements:
                self[e] = elements[e]


class Library(BaseDatabase):
    """This class implements layout libraries that contain designs as their child objects. """

    def get_libname(self):
        return self.name

    def set_libname(self, val):
        self.name = val

    libname = property(get_libname, set_libname)
    """str: library name"""
    
    def append(self, item):
        if isinstance(item, list) or isinstance(item, np.ndarray):
            item_name_list = []
            item_list = []
            for i in item:
                _item_name, _item = self.append(i)
                item_name_list.append(_item_name)
                item_list.append(_item)
            return item_name_list, item_list
        else:
            item_name, item = BaseDatabase.append(self, item)
            item.libname = self.name  # update library name
            return item_name, item

    def __init__(self, name, params=None, elements=None):
        """
        Constructor.

        Parameters
        ----------
        name : str
            The name of the library.
        """
        BaseDatabase.__init__(self, name=name, params=params, elements=elements)

    def summarize(self):
        """Returns the summary of the object information."""
        return BaseDatabase.summarize(self)


class TemplateLibrary(Library):
    """This class implements template libraries that contain templates as their child objects. """
    #vTODO: implement this.
    pass


class GridLibrary(Library):
    """This class implements layout libraries that contain grid objectss as their child objects. """
    # TODO: implement this.
    pass


class Design(BaseDatabase):
    """This class implements layout libraries that contain layout objects as their child objects. """

    @property
    def bbox(self):
        """ get design bbox which is union of instances.bbox  """
        libname  = self.libname
        cellname = self.cellname
        # Compute boundaries
        xy = [None, None]
        for n, i in self.instances.items():
            if xy[0] is None:
                xy[0] = i.bbox[0]   # bl
                xy[1] = i.bbox[1]   # tr
            else:
                xy[0][0] = min(xy[0][0], i.bbox[0, 0])
                xy[0][1] = min(xy[0][1], i.bbox[0, 1])
                xy[1][0] = max(xy[1][0], i.bbox[1, 0])
                xy[1][1] = max(xy[1][1], i.bbox[1, 1])
        for n, i in self.virtual_instances.items():
            if xy[0] is None:
                xy[0] = i.bbox[0]
                xy[1] = i.bbox[1]
            else:
                xy[0][0] = min(xy[0][0], i.bbox[0, 0])
                xy[0][1] = min(xy[0][1], i.bbox[0, 1])
                xy[1][0] = max(xy[1][0], i.bbox[1, 0])
                xy[1][1] = max(xy[1][1], i.bbox[1, 1])
        xy = np.array(xy)
        return (xy)
        pass


    def get_libname(self):
        return self._libname

    def set_libname(self, val):
        self._libname = val

    libname = property(get_libname, set_libname)
    """str: library name"""

    def get_cellname(self):
        return self.name

    def set_cellname(self, val):
        self.name = val

    cellname = property(get_cellname, set_cellname)
    """str: cell name"""
    rects = None
    paths = None
    pins = None
    texts = None
    instances = None
    virtual_instances = None

    def __iter__(self):
        """Iterator function. Directly mapped to its elements."""
        return self.elements.__iter__()

    def __init__(self, name, params=None, elements=None, libname=None):
        """
        Constructor.

        Parameters
        ----------
        name : str
            The name of the design.
        libname : str
            The library name of the design.
        """
        self.libname = libname
        self.rects = dict()
        self.paths = dict()
        self.pins = dict()
        self.texts = dict()
        self.instances = dict()
        self.virtual_instances = dict()
        BaseDatabase.__init__(self, name=name, params=params, elements=elements)

    def append(self, item):
        if isinstance(item, list) or isinstance(item, np.ndarray):
            return [self.append(i) for i in item]
        else:
            if item is None:
                return None, None  # don't do anything
            item_name, _item = BaseDatabase.append(self, item)
            if item.__class__ == laygo2.object.Rect:
                self.rects[item_name] = item
            elif item.__class__ == laygo2.object.Path:
                self.paths[item_name] = item
            elif item.__class__ == laygo2.object.Pin:
                self.pins[item_name] = item
            elif item.__class__ == laygo2.object.Text:
                self.texts[item_name] = item
            elif item.__class__ == laygo2.object.Instance:
                self.instances[item_name] = item
            elif item.__class__ == laygo2.object.VirtualInstance:
                self.virtual_instances[item_name] = item
            return item_name, item

    def summarize(self):
        """Returns the summary of the object information."""
        return \
            BaseDatabase.summarize(self) + " \n" + \
            "    libname:" + str(self.libname) + " \n" + \
            "    rects:" + str(self.rects) + " \n" + \
            "    paths:" + str(self.paths) + " \n" + \
            "    pins:" + str(self.pins) + " \n" + \
            "    texts:" + str(self.texts) + " \n" + \
            "    instances:" + str(self.instances) + "\n" + \
            "    virtual instances:" + str(self.virtual_instances) + \
            ""

    # Object creation and manipulation functions.
    def place(self, inst, grid, mn):
        """Places an instance on the specified coordinate mn, on this grid."""
        if isinstance(inst, ( laygo2.object.Instance, laygo2.object.VirtualInstance) ) :
            inst = grid.place(inst, mn)
            self.append(inst)
            return inst
        else:
            matrix = np.asarray( inst )
            size   = matrix.shape

            if len(size) == 2:
                m, n = size
            else:
                m, n = 1, size[0]
                matrix = [ matrix ]
            mn_ref = np.array(mn)

            for index in range(m):
                row = matrix[index]
                if index != 0 :
                    ns = 0
                    ms = index -1
                    while row[ns] == None:      # Right search
                        ns = ns + 1
                    while matrix[ms][ns] == None: # Down search
                        ms = ms - 1
                    mn_ref = grid.mn.top_left( matrix[ms][ns] )
                for element in row:
                    if isinstance( element, (laygo2.object.Instance, laygo2.object.VirtualInstance) ):
                        mn_bl    = grid.mn.bottom_left( element )
                        mn_comp  = mn_ref - mn_bl
                        inst_sub = grid.place( element, mn_comp)
                        self.append(inst_sub)
                        mn_ref = grid.mn.bottom_right( element )
                    else:
                        if element == None:
                            pass
                        elif isinstance( element, int):
                            mn_ref = mn_ref + [ element,0 ]

    def route(self, grid, mn, direction=None, via_tag=None):
        """Creates Path and Via objects over the abstract coordinates specified by mn, on this routing grid. """
        r = grid.route(mn=mn, direction=direction, via_tag=via_tag)
        self.append(r)
        return r

    def route_via_track(self, grid, mn, track):
        """Creates Path and Via objects over the abstract coordinates specified by mn, 
        on the track of specified routing grid. """
        r = grid.route_via_track(mn=mn, track=track)
        self.append(r)
        return r

    def via(self, grid, mn, params=None):
        """Creates a Via object over the abstract coordinates specified by mn, on this routing grid. """
        v = grid.via(mn=mn, params=params)
        self.append(v)
        return v

    def pin(self, name, grid, mn, direction=None, netname=None, params=None):
        """Creates a Pin object over the abstract coordinates specified by mn, on this routing grid. """
        p = grid.pin(name=name, mn=mn, direction=direction, netname=netname, params=params)
        self.append(p)
        return p

    # I/O functions
    def export_to_template(self, libname=None, cellname=None):
        """Convert this design to a native-instance template"""
        if libname is None:
            libname = self.libname
        if cellname is None:
            cellname = self.cellname

        xy   = self.bbox
        pins = self.pins
        return laygo2.object.template.NativeInstanceTemplate(libname=libname, cellname=cellname, bbox=xy, pins=pins)


    def get_matchedrects_by_layer(self, lpp ):
        """ return matched objects by [ "M*", "drawing or pin"] """
        rects  = self.rects
        insts  = self.instances
        vinsts = self.virtual_instances

        obj_check = []

        for rname, rect in rects.items():
            if np.array_equal( rect.layer, lpp):
                obj_check.append(rect)

        for iname, inst in insts.items():
            for pname , pin in inst.pins.items():
                if np.array_equal( pin.layer, lpp ):
                    obj_check.append(pin)

        for iname, vinst in vinsts.items():
            for name, inst in vinst.native_elements.items():
                if isinstance(inst, laygo2.object.physical.Rect):
                    if np.array_equal( inst.layer, lpp):
                        _xy   = vinst.get_element_position(vinst,inst)
                        ninst = laygo2.object.physical.Rect(
                            xy=_xy, layer = lpp, hextension = inst.hextension, vextension = inst.vextension
                            ,color = inst.color )
                        obj_check.append(ninst)   ## ninst is for sort, inst should be frozen for implement to layout
        return obj_check







if __name__ == '__main__':
    from laygo2.object.physical import *
    # Test
    lib = Library(name='mylib')
    dsn = Design(name='mycell')
    lib.append(dsn)
    rect0 = Rect(xy=[[0, 0], [100, 100]], layer=['M1', 'drawing'], name='R0', netname='net0', params={'maxI': 0.005})
    dsn.append(rect0)
    rect1 = Rect(xy=[[200, 0], [300, 100]], layer=['M1', 'drawing'], netname='net0', params={'maxI': 0.005})
    dsn.append(rect1)
    path0 = Path(xy=[[0, 0], [0, 100]], width=10, extension=5, layer=['M1', 'drawing'], netname='net0',
                 params={'maxI': 0.005})
    dsn.append(path0)
    pin0 = Pin(xy=[[0, 0], [100, 100]], layer=['M1', 'pin'], netname='n0', master=rect0, params={'direction': 'input'})
    dsn.append(pin0)
    #text0 = Text(xy=[0, 0], layer=['text', 'drawing'], text='test', params=None)
    #dsn.append(text0)
    inst0_pins = dict()
    inst0_pins['in'] = Pin(xy=[[0, 0], [10, 10]], layer=['M1', 'drawing'], netname='in')
    inst0_pins['out'] = Pin(xy=[[90, 90], [100, 100]], layer=['M1', 'drawing'], netname='out')
    inst0 = Instance(name='I0', xy=[100, 100], libname='mylib', cellname='mycell', shape=[3, 2], pitch=[100, 100],
                     unit_size=[100, 100], pins=inst0_pins, transform='R0')
    dsn.append(inst0)
    print(lib)
    print(dsn)