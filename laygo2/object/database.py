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
Module consisting of the classes implementing a hierarchical structure database that manages design and library.
"""

__author__ = "Jaeduk Han"
__maintainer__ = "Jaeduk Han"
__status__ = "Prototype"

import laygo2.object
import numpy as np


class BaseDatabase:
    """
    BaseDatabase is a base class implementing basic functions of various database objects such as library and design.

    Notes
    -----
    **(Korean)** BaseDatabase는 데이터베이스 객체들의 기본 기능을 구현하는 클래스.
    """

    name = None
    """str: BaseDatabase object name.

    Example
    --------
    >>> import laygo2
    >>> base = laygo2.object.database.BaseDatabase(name="mycell") 
    >>> base.name 
    "mycell"

    Notes
    -----
    **(Korean)** BaseDatabase 이름.
    """

    params = None
    """dict or None: Dictionary containing parameters of BaseDatabase object.

    Example
    --------
    >>> import laygo2
    >>> base = laygo2.object.database.BaseDatabase(name="mycell",
                   params={'ivdd': 0.001}) 
    >>> base.params 
    {'ivdd': 0.001}

    Notes
    -----
    **(Korean)** BaseDatabase의 속성.
    """

    elements = None
    """dict: Dictionary containing its element objects.

    Example
    --------
    >>> import laygo2
    >>> from laygo2.object.database import BaseDatabase
    >>> from laygo2.object.physical import Rect, Pin, Instance, Text
    >>> # Create a design.
    >>> dsn = BaseDatabase(name="mycell")
    >>> # Create layout objects.
    >>> r0 = Rect(xy=[[0, 0], [100, 100]], layer=["M1", "drawing"])
    >>> p0 = Pin(xy=[[0, 0], [50, 50]], layer=["M1", "pin"], name="P")
    >>> i0 = Instance(libname="tlib", cellname="t0", name="I0", xy=[0, 0])
    >>> t0 = Text(xy=[[50, 50], [100, 100]], layer=["text", "drawing"], text="T")
    >>> # Add layout objects to the design.
    >>> dsn.append(r0)
    >>> dsn.append(p0)
    >>> dsn.append(i0)
    >>> dsn.append(t0)
    >>> # 
    >>> # Display elements of the design.
    >>> print(dsn.elements) 
    {'NoName_0': <laygo2.object.physical.Rect object at 0x0000024C6C230F40>, 
    'P': <laygo2.object.physical.Pin object at 0x0000024C6C2EFF40>, 
    'I0': <laygo2.object.physical.Instance object at 0x0000024C6C2EFDC0>, 
    'NoName_1': <laygo2.object.physical.Text object at 0x0000024C6C2EF8B0>}

    Notes
    -----
    **(Korean)** BaseDatabase 객체의 구성 요소를 담고 있는 Dictionary.
    """

    noname_index = 0
    """
    int: A unique number used as the name of an unnamed object belonging to 
    the database.    

    Example
    --------
    >>> import laygo2
    >>> from laygo2.object.database import BaseDatabase
    >>> from laygo2.object.physical import Rect, Pin, Instance, Text
    >>> # Create a design
    >>> dsn = BaseDatabase(name="mycell")
    >>> # Create layout objects
    >>> r0 = Rect(xy=[[0, 0], [100, 100]], layer=["M1", "drawing"])
    >>> dsn.append(r0)
    >>> print(base.noname_index) 
    0 
    >>> r1 = Rect(xy=[[100, 100], [200, 200]], layer=["M1", "drawing"])
    >>> dsn.append(r1)
    >>> print(base.noname_index) 
    1

    Notes
    -----
    **(Korean)** BaseDatabase의 소속 객체들 중 이름이 정해지지 않은 객체의 
    이름을 정할 때 부여되는 고유 번호.
    """

    @property
    def keys(self):
        """ Keys of elements.

        Example
        --------
        >>> import laygo2
        >>> from laygo2.object.database import BaseDatabase
        >>> from laygo2.object.physical import Rect, Pin, Instance, Text
        >>> # Create a design
        >>> dsn = BaseDatabase(name="mycell")
        >>> # Create layout objects
        >>> r0 = Rect(xy=[[0, 0], [100, 100]], layer=["M1", "drawing"])
        >>> p0 = Pin(xy=[[0, 0], [50, 50]], layer=["M1", "pin"], name="P")
        >>> i0 = Instance(libname="tlib", cellname="t0", name="I0", xy=[0, 0])
        >>> t0 = Text(xy=[[50, 50], [100, 100]], layer=["text", "drawing"], text="T")
        >>> dsn.append(r0)
        >>> dsn.append(p0)
        >>> dsn.append(i0)
        >>> dsn.append(t0)
        >>> print(dsn.keys())
        dict_keys(['NoName_0', 'P', 'I0', 'NoName_1'])

        Notes
        -----
        **(Korean)** BaseDatabase 객체의 구성 요소를 담고 있는 Dictionary.
        """
        return self.elements.keys

    def items(self):
        """
        key/object pair of elements.

        Parameters
        ----------
        None

        Returns
        -------
        dict_items

        Example
        --------
        >>> import laygo2
        >>> from laygo2.object.database import BaseDatabase
        >>> from laygo2.object.physical import Rect, Pin, Instance, Text
        >>> # Create a design
        >>> dsn = BaseDatabase(name="mycell")
        >>> # Create layout objects
        >>> r0 = Rect(xy=[[0, 0], [100, 100]], layer=["M1", "drawing"])
        >>> p0 = Pin(xy=[[0, 0], [50, 50]], layer=["M1", "pin"], name="P")
        >>> i0 = Instance(libname="tlib", cellname="t0", name="I0", xy=[0, 0])
        >>> t0 = Text(xy=[[50, 50], [100, 100]], layer=["text", "drawing"], text="T")
        >>> dsn.append(r0)
        >>> dsn.append(p0)
        >>> dsn.append(i0)
        >>> dsn.append(t0)
        >>> print(dsn.items())
        dict_items([('NoName_0', <laygo2.object.physical.Rect object at 0x0000024C6C230F40>), 
                    ('P', <laygo2.object.physical.Pin object at 0x0000024C6C2EFF40>), 
                    ('I0', <laygo2.object.physical.Instance object at 0x0000024C6C2EFDC0>), 
                    ('NoName_1', <laygo2.object.physical.Text object at 0x0000024C6C2EF8B0>)])

        Notes
        -----
        **(Korean)** elements의 key/object 짝 출력.
        """
        return self.elements.items()

    def __getitem__(self, pos):
        """
        Return the object corresponding to the key.

        Parameters
        ----------
        key : str

        Returns
        -------
        laygo2.object.physical : corresponding object.

        Example
        --------
        >>> import laygo2
        >>> from laygo2.object.database import BaseDatabase
        >>> from laygo2.object.physical import Rect, Pin, Instance, Text
        >>> # Create a design
        >>> dsn = BaseDatabase(name="mycell")
        >>> # Create layout objects
        >>> r0 = Rect(xy=[[0, 0], [100, 100]], layer=["M1", "drawing"])
        >>> p0 = Pin(xy=[[0, 0], [50, 50]], layer=["M1", "pin"], name="P")
        >>> i0 = Instance(libname="tlib", cellname="t0", name="I0", xy=[0, 0])
        >>> t0 = Text(xy=[[50, 50], [100, 100]], layer=["text", "drawing"], text="T")
        >>> dsn.append(r0)
        >>> dsn.append(p0)
        >>> dsn.append(i0)
        >>> dsn.append(t0)
        >>> print(dsn["I0"])
        <laygo2.object.physical.Instance object at 0x0000024C6C2EFDC0> 
        name: I0,
        class: Instance,
        xy: [0, 0],
        params: None,
        size: [0, 0]
        shape: None
        pitch: [0, 0]   
        transform: R0
        pins: {}

        Notes
        -----
        **(Korean)** key에 해당하는 object 반환.
        """
        return self.elements[pos]

    def __setitem__(self, key, item):
        """
        Add key/object pair.

        Parameters
        ----------
        key : str

        Returns
        -------
        list

        Example
        --------
        >>> import laygo2
        >>> from laygo2.object.database import BaseDatabase
        >>> from laygo2.object.physical import Rect, Pin, Instance, Text
        >>> # Create a design
        >>> dsn = BaseDatabase(name="mycell")
        >>> # Create layout objects
        >>> r1 = Rect(xy=[[0, 0], [100, 100]], layer=["M1", "drawing"])
        >>> dsn.append(r1)
        >>> r2 = Rect(xy=[[0, 0], [100, 100]], layer=["M1", "drawing"])
        >>> dsn["R2"] = r2
        >>> print(dsn["R2"])
        <laygo2.object.physical.Rect object at 0x0000024C6C107C40> 
        name: R2,
        class: Rect,
        xy: [[0, 0], [100, 100]],
        params: None, , layer: ['M1', 'drawing'], netname: None

        Notes
        -----
        **(Korean)** 요소 추가 함수.
        """
        item.name = key
        self.append(item)

    def append(self, item):
        """(Simply) add a physical object to the BaseDatabase object.

        Parameters
        ----------
        item : laygo2.object.physical.PhysicalObject
            The physical object to be added. 

        Returns
        -------
        list : 
            A list containing the name of item and the item itself ([item.name, item]).

        Example
        --------
        >>> import laygo2
        >>> from laygo2.object.database import BaseDatabase
        >>> from laygo2.object.physical import Rect, Pin, Instance, Text
        >>> # Create a design
        >>> dsn = Design(name="mycell", libname="genlib")
        >>> # Create layout objects
        >>> r0 = Rect(xy=[[0, 0], [100, 100]], layer=["M1", "drawing"])
        >>> p0 = Pin(xy=[[0, 0], [50, 50]], layer=["M1", "pin"], name="P")
        >>> i0 = Instance(libname="tlib", cellname="t0", name="I0", xy=[0, 0])
        >>> t0 = Text(xy=[[50, 50], [100, 100]], layer=["text", "drawing"], text="T")
        >>> dsn.append(r0)
        >>> dsn.append(p0)
        >>> dsn.append(i0)
        >>> dsn.append(t0)
        >>> print(dsn)
        <laygo2.object.database.BaseDatabase object at 0x0000024C6C2EF010> 
            name: mycell, params: None 
            elements: {
                'NoName_0': <laygo2.object.physical.Rect object at 0x0000024C6C230F40>,
                'P': <laygo2.object.physical.Pin object at 0x0000024C6C2EFF40>, 
                'I0': <laygo2.object.physical.Instance object at 0x0000024C6C2EFDC0>, 
                'NoName_1': <laygo2.object.physical.Text object at 0x0000024C6C2EF8B0>}

        See Also
        --------
            laygo2.object.database.Library.append
            laygo2.object.database.Design.append
        """
        if isinstance(item, list) or isinstance(item, np.ndarray):
            item_name_list = []
            item_list = []
            for i in item:
                _item_name, _item = self.append(i)
                item_name_list.append(_item_name)
                item_list.append(_item)
            return item_name_list, item_list
            # return [i[0] for i in item_list], [i[1] for i in item_list]
        else:
            item_name = item.name
            if item_name is None:  # NoName object. Put a name on it.
                while "NoName_" + str(self.noname_index) in self.elements.keys():
                    self.noname_index += 1
                item_name = "NoName_" + str(self.noname_index)
                self.noname_index += 1
            errstr = (
                item_name
                + " cannot be added to "
                + self.name
                + ", as a child object with the same name exists."
            )
            if item_name in self.elements.keys():
                raise KeyError(errstr)
            else:
                if item_name in self.elements.keys():
                    raise KeyError(errstr)
                else:
                    self.elements[item_name] = item
            return item_name, item

    def __iter__(self):
        """Iterator function. Directly mapped to its elements.

        Example
        --------
        >>> import laygo2
        >>> from laygo2.object.database import BaseDatabase
        >>> from laygo2.object.physical import Rect, Pin, Instance, Text
        >>> # Create a design
        >>> dsn = BaseDatabase(name="mycell")
        >>> # Create layout objects
        >>> r0 = Rect(xy=[[0, 0], [100, 100]], layer=["M1", "drawing"])
        >>> p0 = Pin(xy=[[0, 0], [50, 50]], layer=["M1", "pin"], name="P")
        >>> i0 = Instance(libname="tlib", cellname="t0", name="I0", xy=[0, 0])
        >>> t0 = Text(xy=[[50, 50], [100, 100]], layer=["text", "drawing"], text="T")
        >>> dsn.append(r0)
        >>> dsn.append(p0)
        >>> dsn.append(i0)
        >>> for o in dsn.items():
        >>>     print(o)
        ('NoName_0', <laygo2.object.physical.Rect object at 0x0000024C6C230F40>)
        ('P', <laygo2.object.physical.Pin object at 0x0000024C6C2EFF40>)
        ('I0', <laygo2.object.physical.Instance object at 0x0000024C6C2EFDC0>)
        ('NoName_1', <laygo2.object.physical.Text object at 0x0000024C6C2EF8B0>)

        Notes
        -----
        **(Korean)** BaseDatabase의 Iterable 함수.
        """
        return self.elements.__iter__()

    def __str__(self):
        return self.summarize()

    def summarize(self):
        """Return the summary of the object information."""
        return (
            self.__repr__()
            + " "
            + "name: "
            + self.name
            + ", "
            + "params: "
            + str(self.params)
            + " \n"
            "    elements: " + str(self.elements) + ""
        )

    def __init__(self, name, params=None, elements=None):
        """
        BaseDatabase class constructor function.

        Parameters
        ----------
        name : str
            BaseDatabase object name.
        params : dict, optional
            parameters of BaseDatabase.
        elements : dict, optional
            dictionary having the elements of BaseDatabase.

        Returns
        -------
        laygo2.object.BaseDatabase

        Example
        --------
        >>> import laygo2
        >>> base = laygo2.object.database.BaseDatabase(name='mycell')
        >>> print(base)
        <laygo2.object.database.BaseDatabase object> 
        name: mycell, params: None elements: {}>

        Notes
        -----
        **(Korean)** BaseDatabase 클래스 생성자 함수.

        파라미터
            - name(str): BaseDatabase 객체의 이름
            - params(dict): BaseDatabase의 parameters [optional]
            - elements(dict): BaseDatabase의 elements를 갖고 있는 dict [optional]
        반환값
            - laygo2.object.BaseDatabase
        """
        self.name = name
        self.params = params

        self.elements = dict()
        if elements is not None:
            for e in elements:
                self.elements[e] = elements[e]


class Library(BaseDatabase):
    """
    Library class implements the library management function.

    Example
    --------
    >>> import laygo2
    >>> lib = laygo2.object.database.Library(name="mylib")
    >>> dsn0 = laygo2.object.database.Design(name="mycell0")
    >>> dsn1 = laygo2.object.database.Design(name="mycell1")
    >>> lib.append(dsn0)
    >>> lib.append(dsn1)
    >>> print(lib)
    <laygo2.object.database.Library object at 0x0000025F2D25B8B0> 
    name: mylib, params: None 
    elements: {
        'mycell0': <laygo2.object.database.Design object at 0x0000025F2D25B010>, 
        'mycell1': <laygo2.object.database.Design object at 0x0000025F2D25BF70>}

    See Also
    --------
    laygo2.object.databse.Design: Check for more comprehensive Example.

    Notes
    -----
    **(Korean)** Library 클래스는 라이브러리 관리 기능을 구현한다.

    """

    def get_libname(self):
        """getter function of libname property."""
        return self.name

    def set_libname(self, val):
        """setter function of libname property."""
        self.name = val

    libname = property(get_libname, set_libname)
    """str: The name of library.

    Example
    --------
    >>> import laygo2
    >>> lib = laygo2.object.database.Library(name='mylib') 
    >>> print(lib.name) 
    "mylib"

    Notes
    -----
    **(Korean)** Library 객체의 이름.
    """

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
        """Constructor function of Library class.

        Parameters
        ----------
        name : str
            Library object name.
        params : dict, optional
            Library parameters.
        elements : dict, optional
            Dictionary having the elements of Library.

        Returns
        -------
        laygo2.object.Library

        Example
        --------
        >>> import laygo2
        >>> lib = laygo2.object.database.Library(name='mylib')
        >>> print(lib)
        <laygo2.object.database.Library > name: mylib, params: None elements: {} >

        Notes
        -----
        **(Korean)** Library 클래스의 생성자 함수.
        파라미터
            - name(str): Library 객체의 이름
            - params(dict): Library의 parameters [optional]
            - elements(dict): Library의 elements를 갖고 있는 dict [optional]
        반환값
            - laygo2.object.Library
        """
        BaseDatabase.__init__(self, name=name, params=params, elements=elements)

    def summarize(self):
        """Return the summary of the object information."""
        return BaseDatabase.summarize(self)


class TemplateLibrary(Library):
    """This class implements template libraries that contain templates as their child objects."""

    # TODO: implement this.
    pass


class GridLibrary(Library):
    """This class implements layout libraries that contain grid objectss as their child objects."""

    # TODO: implement this.
    pass


class Design(BaseDatabase):
    """
    Design class implements the design management function.

    Example
    --------

    A physical (non-abstract) grid example:

    >>> import laygo2
    >>> from laygo2.object.database import Design
    >>> from laygo2.object.physical import Rect, Pin, Instance, Text
    >>> # Create a design.
    >>> dsn = Design(name="mycell", libname="genlib")
    >>> # Create layout objects.
    >>> r0 = Rect(xy=[[0, 0], [100, 100]], layer=["M1", "drawing"])
    >>> p0 = Pin(xy=[[0, 0], [50, 50]], layer=["M1", "pin"], name="P")
    >>> i0 = Instance(libname="tlib", cellname="t0", name="I0", xy=[0, 0])
    >>> t0 = Text(xy=[[50, 50], [100, 100]], layer=["text", "drawing"], text="T")
    >>> # Add the layout objects to the design object.
    >>> dsn.append(r0)
    >>> dsn.append(p0)
    >>> dsn.append(i0)
    >>> dsn.append(t0)
    >>> print(dsn)
    <laygo2.object.database.Design object at 0x0000024C6C2EF010> 
        name: mycell, params: None 
        elements: {
            'NoName_0': <laygo2.object.physical.Rect object at 0x0000024C6C230F40>,
            'P': <laygo2.object.physical.Pin object at 0x0000024C6C2EFF40>, 
            'I0': <laygo2.object.physical.Instance object at 0x0000024C6C2EFDC0>, 
            'NoName_1': <laygo2.object.physical.Text object at 0x0000024C6C2EF8B0>}
        libname:genlib
        rects:{
            'NoName_0': <laygo2.object.physical.Rect object at 0x0000024C6C230F40>}
        paths:{}
        pins:{
            'P': <laygo2.object.physical.Pin object at 0x0000024C6C2EFF40>}
        texts:{
            'NoName_1': <laygo2.object.physical.Text object at 0x0000024C6C2EF8B0>}
        instances:{
            'I0': <laygo2.object.physical.Instance object at 0x0000024C6C2EFDC0>}
        virtual instances:{}
    >>> # 
    >>> # Export to a NativeInstanceTemplate for reuse in higher levels.
    >>> nt0 = dsn.export_to_template()
    >>> nt0.dsn.export_to_template()
    >>> print(nt0)
        <laygo2.object.template.NativeInstanceTemplate object at 0x000001CB5A9CE380> 
        name: mycell, 
        class: NativeInstanceTemplate,
         bbox: [[0, 0], [0, 0]], 
         pins: {'P': <laygo2.object.physical.Pin object at 0x000001CB5A9CFF40>},
    >>> # 
    >>> # Export to a skill script.
    >>> lib = laygo2.object.database.Library(name="mylib")
    >>> lib.append(dsn)
    >>> scr = laygo2.interface.skill.export(lib, filename="myscript.il")
    >>> print(scr)
    ; (definitions of laygo2 skill functions)
    ; exporting mylib__mycell
    cv = _laygo2_open_layout("mylib" "mycell" "layout")
    _laygo2_generate_rect(cv, list( "M1" "drawing" ), list( list( 0.0000  0.0000  ) list( 0.1000  0.1000  ) ), "None")
    _laygo2_generate_pin(cv, "P", list( "M1" "pin" ), list( list( 0.0000  0.0000  ) list( 0.0500  0.0500  ) ) )
    _laygo2_generate_instance(cv, "I0", "tlib", "t0", "layout", list( 0.0000  0.0000  ), "R0", 1, 1, 0, 0, nil, nil)
    _laygo2_save_and_close_layout(cv)
    
    An abstract grid example:

    >>> import laygo2
    >>> from laygo2.object.grid import CircularMapping as CM
    >>> from laygo2.object.grid import CircularMappingArray as CMA
    >>> from laygo2.object.grid import OneDimGrid, PlacementGrid, RoutingGrid
    >>> from laygo2.object.template import NativeInstanceTemplate
    >>> from laygo2.object.database import Design
    >>> from laygo2.object.physical import Instance
    >>> # Placement grid construction (not needed if laygo2_tech is set up).
    >>> gx  = OneDimGrid(name="gx", scope=[0, 20], elements=[0])
    >>> gy  = OneDimGrid(name="gy", scope=[0, 100], elements=[0])
    >>> gp  = PlacementGrid(name="test", vgrid=gx, hgrid=gy)
    >>> # Routing grid construction (not needed if laygo2_tech is set up).
    >>> gv = OneDimGrid(name="gv", scope=[0, 50], elements=[0])
    >>> gh = OneDimGrid(name="gv", scope=[0, 100], elements=[0, 40, 60])
    >>> wv = CM([10])           # vertical (xgrid) width
    >>> wh = CM([20, 10, 10])   # horizontal (ygrid) width
    >>> ev = CM([10])           # vertical (xgrid) extension
    >>> eh = CM([10, 10, 10])   # horizontal (ygrid) extension
    >>> e0v = CM([15])          # vert. extension (for zero-length wires)
    >>> e0h = CM([15, 15, 15])  # hori. extension (for zero-length wires)
    >>> lv = CM([['M1', 'drawing']], dtype=object)  # layer information
    >>> lh = CM([['M2', 'drawing']]*3, dtype=object) 
    >>> plv = CM([['M1', 'pin']], dtype=object) # pin layers
    >>> plh = CM([['M2', 'pin']]*3, dtype=object)
    >>> xcolor = CM(['not MPT'], dtype=object)  # 'not MPT' for non-multipatterning 
    >>> ycolor = CM(['not MPT']*3, dtype=object) 
    >>> primary_grid = 'horizontal'
    >>> tvia = NativeInstanceTemplate(libname='tlib', cellname='via0')  # via 
    >>> viamap = CMA(elements=[[tvia, tvia, tvia]], dtype=object)
    >>> gr = laygo2.object.grid.RoutingGrid(name='mygrid', vgrid=gv, hgrid=gh,
                                            vwidth=wv, hwidth=wh,
                                            vextension=ev, hextension=eh,
                                            vlayer=lv, hlayer=lh,
                                            pin_vlayer=plv, pin_hlayer=plh,
                                            viamap=viamap, primary_grid=primary_grid,
                                            xcolor=xcolor, ycolor=ycolor,
                                            vextension0=e0v, hextension0=e0h)
    >>> # Create a design
    >>> dsn = Design(name="mycell", libname="genlib")
    >>> # Create an instance
    >>> i0 = Instance(libname="tlib", cellname="t0", name="I0", xy=[0, 0])
    >>> print(inst0.xy)
    [100, 100]
    >>> # Place the instance
    >>> dsn.place(inst=i0, grid=gp, mn=[10,10])
    >>> # Routing on grid 
    >>> mn_list = [[0, -2], [0, 1], [2, 1], [5,1] ]
    >>> route = dsn.route(grid=gr, mn=mn_list, 
                          via_tag=[True, None, True, True])
    >>> #
    >>> # Display generated design.
    >>> print(dsn)
    <laygo2.object.database.Design object at 0x000001C71AE3A110>
        ...
    >>> # 
    >>> # Export to a NativeInstanceTemplate for reuse in higher levels.
    >>> nt0 = dsn.export_to_template()
    >>> nt0.dsn.export_to_template()
    >>> print(nt0)
        ...
    >>> # 
    >>> # Export to a skill script.
    >>> lib = laygo2.object.database.Library(name="mylib")
    >>> lib.append(dsn)
    >>> scr = laygo2.interface.skill.export(lib, filename="myscript.il")
    >>> print(scr)
        ...
    
    An abstract template/grid example with technology setup (laygo2_tech):

    >>> import laygo2
    >>> import laygo2.interface
    >>> import laygo2_tech_quick_start as tech  # target tech's laygo2_tech
    >>> from laygo2.object.database import Design
    >>> templates = tech.load_templates()
    >>> mytemplate = templates['nmos']
    >>> grids = tech.load_grids(templates=templates)
    >>> gp = grids['placement_basic']
    >>> gr = grids['routing_23_cmos']
    >>> # Create a design
    >>> dsn = Design(name="mycell", libname="genlib")
    >>> # Create an instance
    >>> i0 = tnmos.generate(name='MN0', params={'nf': 4})
    >>> # Place the instance
    >>> dsn.place(inst=i0, grid=gp, mn=[10,10])
    >>> # Routing on grid 
    >>> mn_list = [[0, -2], [0, 1], [2, 1], [5,1] ]
    >>> route = dsn.route(grid=gr, mn=mn_list, 
                          via_tag=[True, None, True, True])
    >>> #
    >>> # Display generated design.
    >>> print(dsn)
    <laygo2.object.database.Design object at 0x000001C71AE3A110>
        ...
    >>> # 
    >>> # Export to a NativeInstanceTemplate for reuse in higher levels.
    >>> nt0 = dsn.export_to_template()
    >>> nt0.dsn.export_to_template()
    >>> print(nt0)
        ...
    >>> # 
    >>> # Export to a skill script.
    >>> lib = laygo2.object.database.Library(name="mylib")
    >>> lib.append(dsn)
    >>> scr = laygo2.interface.skill.export(lib, filename="myscript.il")
    >>> print(scr)
        ...
    

    Notes
    -----
    **(Korean)**
    Design 클래스는 디자인 관리 기능을 구현한다.
    """

    @property
    def bbox(self):
        """get the design bbox which is union of self.instances.bbox."""
        libname = self.libname
        cellname = self.cellname
        # Compute boundaries
        xy = [None, None]
        for n, i in self.instances.items():
            if xy[0] is None:
                xy[0] = i.bbox[0]  # bl
                xy[1] = i.bbox[1]  # tr
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
        return xy

    def get_libname(self):
        return self._libname

    def set_libname(self, val):
        self._libname = val

    libname = property(get_libname, set_libname)
    """str: Library name of Design object.

    Example
    --------
    >>> import laygo2
    >>> dsn = laygo2.object.database.Design(name="dsn", libname="testlib") 
    >>> print(dsn.libname) 
    “testlib”

    Notes
    -----
    **(Korean)** Design 객체의 라이브러리 이름.
    """

    def get_cellname(self):
        return self.name

    def set_cellname(self, val):
        self.name = val

    cellname = property(get_cellname, set_cellname)
    """str: Cell name of Design object.

    Example
    --------
    >>> import laygo2
    >>> dsn = laygo2.object.database.Design(name="dsn", libname="testlib") 
    >>> print(dsn.cellname) 
    “dsn”

    Notes
    -----
    **(Korean)** Design 객체의 셀 이름.
    """

    rects = None
    """dict: Dictionary containing Rectangle object affiliated with the 
    Design object.

    Example
    --------
    >>> import laygo2
    >>> from laygo2.object.database import Design
    >>> from laygo2.object.physical import Rect, Pin, Instance, Text
    >>> dsn = Design(name="dsn", libname="testlib") 
    >>> r0 = Rect(xy=[[0, 0], [100, 100]], layer=["M1", "drawing"])
    >>> dsn.append(r0) 
    >>> print(dsn.rects) 
    {'R0': <laygo2.object.physical.Rect object>}

    Notes
    -----
    **(Korean)** Design 객체에 소속된 Rect 객체들을 갖고 있는 dictionary.
    """
    
    paths = None

    pins = None
    """dict: Dictionary having the collection of Pin objects affiliated 
    with the Design object.

    Example
    --------
    >>> import laygo2
    >>> from laygo2.object.database import Design
    >>> from laygo2.object.physical import Rect, Pin, Instance, Text
    >>> dsn = Design(name="dsn", libname="testlib") 
    >>> p0 = Pin(xy=[[0, 0], [50, 50]], layer=["M1", "pin"], name="P")
    >>> dsn.append(p0) 
    >>> print(dsn.pins) 
    {'NoName_0': <laygo2.object.physical.Pin object>}
    
    Notes
    -----
    **(Korean)** Design 객체에 소속된 Pin 객체들을 갖고 있는 dictionary.
    """

    texts = None
    """dict: Dictionary containing Text objects affiliated with Design object.

    Example
    --------
    >>> import laygo2
    >>> from laygo2.object.database import Design
    >>> from laygo2.object.physical import Rect, Pin, Instance, Text
    >>> dsn = Design(name="dsn", libname="testlib") 
    >>> t0 = Text(xy=[[50, 50], [100, 100]], layer=["text", "drawing"], text="T")
    >>> dsn.append(t0) 
    >>> print(dsn.texts) 
    {'NoName_1': <laygo2.object.physical.Text object>}
    
    Notes
    -----
    **(Korean)** Design 객체에 소속된 Text 객체들을 갖고 있는 dictionary.
    """

    instances = None
    """dict: Dictionary containing Instance objects affiliated with Design object.

    Example
    --------
    >>> import laygo2
    >>> from laygo2.object.database import Design
    >>> from laygo2.object.physical import Rect, Pin, Instance, Text
    >>> dsn = Design(name="dsn", libname="testlib") 
    >>> i0 = Instance(libname="tlib", cellname="t0", name="I0", xy=[0, 0])
    >>> dsn.append(i0) 
    >>> print(dsn.instances) 
    {'I0': <laygo2.object.physical.Instance object>}
    
    Notes
    -----
    **(Korean)** Design 객체에 소속된 Instance 객체들을 갖고 있는 dictionary.
    """

    virtual_instances = None
    """dict: Dictionary containing VirtualInstance objects affiliated with 
    Design object.

    See Also
    --------
    instances

    Notes
    -----
    **(Korean)** Design 객체에 소속된 VirtualInstance 객체들을 갖고 있는 dictionary.
    """

    def __iter__(self):
        """Iterator function. Directly mapped to its elements.

        Example
        --------
        >>> import laygo2
        >>> from laygo2.object.database import Design
        >>> from laygo2.object.physical import Rect, Pin, Instance, Text
        >>> # Create a design
        >>> dsn = Design(name="mycell", libname="genlib")
        >>> # Create layout objects
        >>> r0 = Rect(xy=[[0, 0], [100, 100]], layer=["M1", "drawing"])
        >>> p0 = Pin(xy=[[0, 0], [50, 50]], layer=["M1", "pin"], name="P")
        >>> i0 = Instance(libname="tlib", cellname="t0", name="I0", xy=[0, 0])
        >>> t0 = Text(xy=[[50, 50], [100, 100]], layer=["text", "drawing"], text="T")
        >>> dsn.append(r0)
        >>> dsn.append(p0)
        >>> dsn.append(i0)
        >>> for o in dsn.items():
        >>>     print(o)
        ('NoName_0', <laygo2.object.physical.Rect object at 0x0000024C6C230F40>)
        ('P', <laygo2.object.physical.Pin object at 0x0000024C6C2EFF40>)
        ('I0', <laygo2.object.physical.Instance object at 0x0000024C6C2EFDC0>)
        ('NoName_1', <laygo2.object.physical.Text object at 0x0000024C6C2EF8B0>)
        """
        return self.elements.__iter__()

    def __init__(self, name, params=None, elements=None, libname=None):
        """
        Constructor function of Design class.

        Parameters
        ----------
        name : str
            Design object name.
        params : dict, optional
            Design object parameters.
        elements : dict, optional
            Design object elements.

        Returns
        -------
        laygo2.object.BaseDatabase

        Example
        --------
        >>> import laygo2
        >>> dsn = laygo2.object.database.Design(name='dsn', libname="testlib")
        >>> print(dsn)
        <laygo2.object.database.Design object>  name: dsn, params: None
            elements: {}
            libname:testlib
            rects:{}
            paths:{}
            pins:{}
            texts:{}
            instances:{}
            virtual instances:{}

        Notes
        -----
        **(Korean)** Design 클래스의 생성자 함수.

        파라미터
            - name(str): Design 객체의 이름
            - params(dict): Design 객체의 parameters [optional]
            - elements(dict): Design 객체의 elements [optional]
        반환값
            - laygo2.object.BaseDatabase
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
        """(Simply) add a physical object to the design.

        Parameters
        ----------
        item : laygo2.object.physical.PhysicalObject
            The physical object to be added. 

        Returns
        -------
        list : 
            A list containing the name of item and the item itself ([item.name, item]).

        Example
        --------
        >>> import laygo2
        >>> from laygo2.object.database import Design
        >>> from laygo2.object.physical import Rect, Pin, Instance, Text
        >>> # Create a design
        >>> dsn = Design(name="mycell", libname="genlib")
        >>> # Create layout objects
        >>> r0 = Rect(xy=[[0, 0], [100, 100]], layer=["M1", "drawing"])
        >>> p0 = Pin(xy=[[0, 0], [50, 50]], layer=["M1", "pin"], name="P")
        >>> i0 = Instance(libname="tlib", cellname="t0", name="I0", xy=[0, 0])
        >>> t0 = Text(xy=[[50, 50], [100, 100]], layer=["text", "drawing"], text="T")
        >>> dsn.append(r0)
        >>> dsn.append(p0)
        >>> dsn.append(i0)
        >>> dsn.append(t0)
        >>> print(dsn)
        <laygo2.object.database.Design object at 0x0000024C6C2EF010> 
            name: mycell, params: None 
            elements: {
                'NoName_0': <laygo2.object.physical.Rect object at 0x0000024C6C230F40>,
                'P': <laygo2.object.physical.Pin object at 0x0000024C6C2EFF40>, 
                'I0': <laygo2.object.physical.Instance object at 0x0000024C6C2EFDC0>, 
                'NoName_1': <laygo2.object.physical.Text object at 0x0000024C6C2EF8B0>}
            libname:genlib
            rects:{
                'NoName_0': <laygo2.object.physical.Rect object at 0x0000024C6C230F40>}
            paths:{}
            pins:{
                'P': <laygo2.object.physical.Pin object at 0x0000024C6C2EFF40>}
            texts:{
                'NoName_1': <laygo2.object.physical.Text object at 0x0000024C6C2EF8B0>}
            instances:{
                'I0': <laygo2.object.physical.Instance object at 0x0000024C6C2EFDC0>}
            virtual instances:{}

        See Also
        --------
            laygo2.object.database.Design.place : Place a (virtual) instance 
            on a grid and append to the design.
            laygo2.object.database.Design.route : Route on a grid and append 
            to the design.
            laygo2.object.database.Design.route_via_track : Route on a track 
            on a grid and append.
            laygo2.object.database.Design.pin : Place a pin on a grid and 
            append to the design.
        """
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
        """Return the summary of the object information."""
        return (
            BaseDatabase.summarize(self)
            + " \n"
            + "    libname:"
            + str(self.libname)
            + " \n"
            + "    rects:"
            + str(self.rects)
            + " \n"
            + "    paths:"
            + str(self.paths)
            + " \n"
            + "    pins:"
            + str(self.pins)
            + " \n"
            + "    texts:"
            + str(self.texts)
            + " \n"
            + "    instances:"
            + str(self.instances)
            + "\n"
            + "    virtual instances:"
            + str(self.virtual_instances)
            + ""
        )

    # Object creation and manipulation functions.
    def place(self, inst, grid, mn):
        """
        Place the instance on the abstract coordinate **mn**, on the abstract grid.

        Parameters
        ----------
        inst : laygo2.object.physical.Instance or laygo2.object.physical.VirtualInstance
            The instance to be placed.
        grid : laygo2.object.grid.PlacementGrid
            The placement grid where the instance is placed on.
        mn : numpy.ndarray or list
            The abstract coordinate value [m, n] to place the instance.

        Returns
        -------
        laygo2.object.physical.Instance or laygo2.object.physical.VirtualInstance :
            The placed instance.

        Example
        --------
        >>> import laygo2
        >>> from laygo2.object.grid import OneDimGrid, PlacementGrid
        >>> from laygo2.object.database import Design
        >>> from laygo2.object.physical import Instance
        >>> # Create a grid (not needed if laygo2_tech is set up).
        >>> gx  = OneDimGrid(name="gx", scope=[0, 20], elements=[0])
        >>> gy  = OneDimGrid(name="gy", scope=[0, 100], elements=[0])
        >>> g   = PlacementGrid(name="test", vgrid=gx, hgrid=gy)
        >>> # Create a design
        >>> dsn = Design(name="mycell", libname="genlib")
        >>> # Create an instance
        >>> i0 = Instance(libname="tlib", cellname="t0", name="I0", xy=[0, 0])
        >>> print(inst0.xy)
        [100, 100]
        >>> # Place the instance
        >>> dsn.place(inst=i0, grid=g, mn=[10,10])
        >>> # Print parameters of the placed instance.
        >>> print(i0.xy)
        [200, 1000]
        >>> print(dsn)
        <laygo2.object.database.Design object at 0x000002803D4C0F40> 
            name: mycell 
            params: None
            elements: 
                {'I0': <laygo2.object.physical.Instance object at 
                        0x000002803D57F010>}
            libname:genlib
            rects:{}
            paths:{}
            pins:{}
            texts:{}
            instances:
                {'I0': <laygo2.object.physical.Instance object at 0x000002803D57F010>}
            virtual instances:{}

        See Also
        --------
        laygo2.object.grid.PlacementGrid.place : place a (virtual) instance 
            on the grid.

        Notes
        -----
        **(Korean)** 인스턴스를 grid위 추상 좌표 mn에 배치하는 함수.

        파라미터
            - inst(laygo2.physical.instance): 배치할 인스턴스.
            - mn(numpy.ndarray or list): 인스턴스를 배치할 추상좌표.

        반환값
            - laygo2.physical.instance: 좌표가 수정된 인스턴스.
        """
        if isinstance(inst, (laygo2.object.Instance, laygo2.object.VirtualInstance)):
            inst = grid.place(inst, mn)
            self.append(inst)
            return inst
        else:
            matrix = np.asarray(inst)
            size = matrix.shape

            if len(size) == 2:
                m, n = size
            else:
                m, n = 1, size[0]
                matrix = [matrix]
            mn_ref = np.array(mn)

            for index in range(m):
                row = matrix[index]
                if index != 0:
                    ns = 0
                    ms = index - 1
                    while row[ns] == None:  # Right search
                        ns = ns + 1
                    while matrix[ms][ns] == None:  # Down search
                        ms = ms - 1
                    mn_ref = grid.mn.top_left(matrix[ms][ns])
                for element in row:
                    if isinstance(
                        element, (laygo2.object.Instance, laygo2.object.VirtualInstance)
                    ):
                        mn_bl = grid.mn.bottom_left(element)
                        mn_comp = mn_ref - mn_bl
                        inst_sub = grid.place(element, mn_comp)
                        self.append(inst_sub)
                        mn_ref = grid.mn.bottom_right(element)
                    else:
                        if element == None:
                            pass
                        elif isinstance(element, int):
                            mn_ref = mn_ref + [element, 0]

    def route(self, grid, mn, direction=None, via_tag=None):
        """
        Create wire object(s) for routing, on the abstract coordinate **mn**.

        Parameters
        ----------
        grid : laygo2.object.grid.RoutingGrid
            The placement grid where the wire is placed on.
        mn : list(numpy.ndarray)
            The list containing two or more mn coordinates to be connected.
        direction : str, optional.
            None or “vertical” or "horizontal". The direction of the routing 
            object.
        via_tag : list(Boolean), optional.
            The list containing switches deciding whether to place via at 
            the edges.

        Returns
        -------
        laygo2.object.physical.Rect or list :
            The generated routing object(s).
            Check the example code in laygo2.object.grid.RoutingGrid.route 
            for details.

        Example
        --------
        >>> import laygo2
        >>> from laygo2.object.grid import CircularMapping as CM
        >>> from laygo2.object.grid import CircularMappingArray as CMA
        >>> from laygo2.object.grid import OneDimGrid, RoutingGrid
        >>> from laygo2.object.template import NativeInstanceTemplate
        >>> from laygo2.object.database import Design
        >>> from laygo2.object.physical import Instance
        >>> # Routing grid construction (not needed if laygo2_tech is set up).
        >>> gv = OneDimGrid(name="gv", scope=[0, 50], elements=[0])
        >>> gh = OneDimGrid(name="gv", scope=[0, 100], elements=[0, 40, 60])
        >>> wv = CM([10])           # vertical (xgrid) width
        >>> wh = CM([20, 10, 10])   # horizontal (ygrid) width
        >>> ev = CM([10])           # vertical (xgrid) extension
        >>> eh = CM([10, 10, 10])   # horizontal (ygrid) extension
        >>> e0v = CM([15])          # vert. extension (for zero-length wires)
        >>> e0h = CM([15, 15, 15])  # hori. extension (for zero-length wires)
        >>> lv = CM([['M1', 'drawing']], dtype=object)  # layer information
        >>> lh = CM([['M2', 'drawing']]*3, dtype=object) 
        >>> plv = CM([['M1', 'pin']], dtype=object) # pin layers
        >>> plh = CM([['M2', 'pin']]*3, dtype=object)
        >>> xcolor = CM(['not MPT'], dtype=object)  # 'not MPT' for non-multipatterning 
        >>> ycolor = CM(['not MPT']*3, dtype=object) 
        >>> primary_grid = 'horizontal'
        >>> tvia = NativeInstanceTemplate(libname='tlib', cellname='via0')  # via 
        >>> viamap = CMA(elements=[[tvia, tvia, tvia]], dtype=object)
        >>> g = laygo2.object.grid.RoutingGrid(name='mygrid', vgrid=gv, hgrid=gh,
                                               vwidth=wv, hwidth=wh,
                                               vextension=ev, hextension=eh,
                                               vlayer=lv, hlayer=lh,
                                               pin_vlayer=plv, pin_hlayer=plh,
                                               viamap=viamap, primary_grid=primary_grid,
                                               xcolor=xcolor, ycolor=ycolor,
                                               vextension0=e0v, hextension0=e0h)
        >>> # Create a design
        >>> dsn = Design(name="mycell", libname="genlib")
        >>> # Routing on grid 
        >>> mn_list = [[0, -2], [0, 1], [2, 1], [5,1] ]
        >>> route = dsn.route(grid=g, mn=mn_list, 
                              via_tag=[True, None, True, True])
        >>> # Display generated design.
        >>> print(dsn)
        <laygo2.object.database.Design object at 0x000001C71AE3A110> 
            name: mycell, params: None
            elements: {
            'NoName_0': <laygo2.object.physical.Instance object at 0x000001C71AE3BA90>, 
            'NoName_1': <laygo2.object.physical.Rect object at 0x000001C71AE3B820>, 
            'NoName_2': <laygo2.object.physical.Rect object at 0x000001C71AE3ABF0>, 
            'NoName_3': <laygo2.object.physical.Instance object at 0x000001C71AE3A140>, 
            'NoName_4': <laygo2.object.physical.Rect object at 0x000001C71AE39DB0>, 
            'NoName_5': <laygo2.object.physical.Instance object at 0x000001C71AE3AB60>}      
            libname:genlib
            rects: {  # wires
            'NoName_1': <laygo2.object.physical.Rect object at 0x000001C71AE3B820>, 
            'NoName_2': <laygo2.object.physical.Rect object at 0x000001C71AE3ABF0>, 
            'NoName_4': <laygo2.object.physical.Rect object at 0x000001C71AE39DB0>}      
            paths:{}
            pins:{}
            texts:{}
            instances:{  # vias
            'NoName_0': <laygo2.object.physical.Instance object at 0x000001C71AE3BA90>, 
            'NoName_3': <laygo2.object.physical.Instance object at 0x000001C71AE3A140>, 
            'NoName_5': <laygo2.object.physical.Instance object at 0x000001C71AE3AB60>}
            virtual instances:{}

        .. image:: ../assets/img/object_grid_RoutingGrid_route.png
           :height: 250

        See Also
        --------
        laygo2.object.grid.RoutingGrid.route : route wire(s) on the grid.

        Notes
        -----
        **(Korean)** 추상 좌표 위에 라우팅을 수행 하는 함수.

        파라미터
            - mn(list(numpy.ndarray)): 배선을 수행할 2개 이상의 mn 좌표를 담고 있는 list.
            - direction(str): None or “vertical”; path의 방향을 결정 (수평 or 수직) [optional].
            - via_tag(list(Boolean)): Path에 via를 형성 할지를 결정하는 switch들을 담고 있는 list [optional].
        
        반환값
            - list: 생성된 routing object들을 담고 있는 list.
        """
        r = grid.route(mn=mn, direction=direction, via_tag=via_tag)
        self.append(r)
        return r

    def via(self, grid, mn, params=None):
        """
        Create Via object(s) on abstract grid.

        Parameters
        ----------
        mn : list(numpy.ndarray)
            Abstract coordinate(s) that specify location(s) to insert via(s).

        Returns
        -------
        list(physical.PhysicalObject):
            The list containing the generated via objects.

        Example
        --------
        >>> import laygo2
        >>> from laygo2.object.grid import CircularMapping as CM
        >>> from laygo2.object.grid import CircularMappingArray as CMA
        >>> from laygo2.object.grid import OneDimGrid, RoutingGrid
        >>> from laygo2.object.template import NativeInstanceTemplate
        >>> from laygo2.object.database import Design
        >>> from laygo2.object.physical import Instance
        >>> # Routing grid construction (not needed if laygo2_tech is set up).
        >>> gv = OneDimGrid(name="gv", scope=[0, 50], elements=[0])
        >>> gh = OneDimGrid(name="gv", scope=[0, 100], elements=[0, 40, 60])
        >>> wv = CM([10])           # vertical (xgrid) width
        >>> wh = CM([20, 10, 10])   # horizontal (ygrid) width
        >>> ev = CM([10])           # vertical (xgrid) extension
        >>> eh = CM([10, 10, 10])   # horizontal (ygrid) extension
        >>> e0v = CM([15])          # vert. extension (for zero-length wires)
        >>> e0h = CM([15, 15, 15])  # hori. extension (for zero-length wires)
        >>> lv = CM([['M1', 'drawing']], dtype=object)  # layer information
        >>> lh = CM([['M2', 'drawing']]*3, dtype=object) 
        >>> plv = CM([['M1', 'pin']], dtype=object) # pin layers
        >>> plh = CM([['M2', 'pin']]*3, dtype=object)
        >>> xcolor = CM(['not MPT'], dtype=object)  # 'not MPT' for non-multipatterning 
        >>> ycolor = CM(['not MPT']*3, dtype=object) 
        >>> primary_grid = 'horizontal'
        >>> tvia = NativeInstanceTemplate(libname='tlib', cellname='via0') 
        >>> viamap = CMA(elements=[[tvia, tvia, tvia]], dtype=object)
        >>> g = laygo2.object.grid.RoutingGrid(name='mygrid', vgrid=gv, hgrid=gh,
                                               vwidth=wv, hwidth=wh,
                                               vextension=ev, hextension=eh,
                                               vlayer=lv, hlayer=lh,
                                               pin_vlayer=plv, pin_hlayer=plh,
                                               viamap=viamap, primary_grid=primary_grid,
                                               xcolor=xcolor, ycolor=ycolor,
                                               vextension0=e0v, hextension0=e0h)
        >>> # Create a design
        >>> dsn = Design(name="mycell", libname="genlib")
        >>> # Via on grid 
        >>> mn_list = [[0, -2], [1, 0], [2, 5]]
        >>> via = dsn.via(grid=g, mn=mn_list)
        >>> # Display generated design.
        >>> print(dsn)
        <laygo2.object.database.Design object at 0x0000015A77C6A110> 
        name: mycell, params: None,
        elements: {
        'NoName_0': <laygo2.object.physical.Instance object at 0x0000015A77C6AC20>, 
        'NoName_1': <laygo2.object.physical.Instance object at 0x0000015A77C6AD10>, 
        'NoName_2': <laygo2.object.physical.Instance object at 0x0000015A77C6AD40>}
        libname:genlib
        rects:{}
        paths:{}
        pins:{}
        texts:{}
        instances:{
        'NoName_0': <laygo2.object.physical.Instance object at 0x0000015A77C6AC20>, 
        'NoName_1': <laygo2.object.physical.Instance object at 0x0000015A77C6AD10>, 
        'NoName_2': <laygo2.object.physical.Instance object at 0x0000015A77C6AD40>}
        virtual instances:{}

        .. image:: ../assets/img/object_grid_RoutingGrid_via.png
           :height: 250

        See Also
        --------
        laygo2.object.grid.RoutingGrid.via

        Notes
        -----
        **(Korean)** via 생성함수.
        파라미터
            - mn(list(numpy.ndarray)): via를 생성할 mn좌표. 복수 개 입력 가능.
        반환값
            - list(physical.PhysicalObject)): 생성된 via object들을 담고 있는 list.
        """
        v = grid.via(mn=mn, params=params)
        self.append(v)
        return v

    def route_via_track(self, grid, mn, track, via_tag=[None, True]):
        """
        Perform routing on the specified track with accessing wires to mn.

        Parameters
        ----------
        grid : laygo2.object.grid.RoutingGrid
            The placement grid where the wire is placed on.
        mn : list(numpy.ndarray)
            list containing coordinates of the points being connected through a track
        track : numpy.ndarray
            list containing coordinate values and direction of a track.
            Vertical tracks have [v, None] format, while horizontal tracks have [None, v] format
            (v is the coordinates of the track).
        via_tag : list(Boolean), optional.
            The list containing switches deciding whether to place via at 
            the edges of individual stubs.

        Returns
        -------
        list:
            The list containing the generated routing objects;
            The last object corresponds to the routing object on the track.

        Example
        --------
        >>> import laygo2
        >>> from laygo2.object.grid import CircularMapping as CM
        >>> from laygo2.object.grid import CircularMappingArray as CMA
        >>> from laygo2.object.grid import OneDimGrid, RoutingGrid
        >>> from laygo2.object.template import NativeInstanceTemplate
        >>> from laygo2.object.database import Design
        >>> from laygo2.object.physical import Instance
        >>> # Routing grid construction (not needed if laygo2_tech is set up).
        >>> gv = OneDimGrid(name="gv", scope=[0, 50], elements=[0])
        >>> gh = OneDimGrid(name="gv", scope=[0, 100], elements=[0, 40, 60])
        >>> wv = CM([10])           # vertical (xgrid) width
        >>> wh = CM([20, 10, 10])   # horizontal (ygrid) width
        >>> ev = CM([10])           # vertical (xgrid) extension
        >>> eh = CM([10, 10, 10])   # horizontal (ygrid) extension
        >>> e0v = CM([15])          # vert. extension (for zero-length wires)
        >>> e0h = CM([15, 15, 15])  # hori. extension (for zero-length wires)
        >>> lv = CM([['M1', 'drawing']], dtype=object)  # layer information
        >>> lh = CM([['M2', 'drawing']]*3, dtype=object) 
        >>> plv = CM([['M1', 'pin']], dtype=object) # pin layers
        >>> plh = CM([['M2', 'pin']]*3, dtype=object)
        >>> xcolor = CM(['not MPT'], dtype=object)  # 'not MPT' for non-multipatterning 
        >>> ycolor = CM(['not MPT']*3, dtype=object) 
        >>> primary_grid = 'horizontal'
        >>> tvia = NativeInstanceTemplate(libname='tlib', cellname='via0')  # via 
        >>> viamap = CMA(elements=[[tvia, tvia, tvia]], dtype=object)
        >>> g = laygo2.object.grid.RoutingGrid(name='mygrid', vgrid=gv, hgrid=gh,
                                               vwidth=wv, hwidth=wh,
                                               vextension=ev, hextension=eh,
                                               vlayer=lv, hlayer=lh,
                                               pin_vlayer=plv, pin_hlayer=plh,
                                               viamap=viamap, primary_grid=primary_grid,
                                               xcolor=xcolor, ycolor=ycolor,
                                               vextension0=e0v, hextension0=e0h)
        >>> # Create a design
        >>> dsn = Design(name="mycell", libname="genlib")
        >>> # Do routing
        >>> mn_list = [[0, -2], [1, 0], [2, 5], [3, 4], [4, 5], [5, 5]]
        >>> track = dsn.route_via_track(grid=g, mn=mn_list, track=[None,0])
        >>> # Display design
        >>> print(dsn)
            <laygo2.object.database.Design object at 0x0000015A77C6BA60> 
            name: mycell, params: None
            elements: {
            'NoName_0': <laygo2.object.physical.Rect object at 0x0000015A77C6B790>, 
            'NoName_1': <laygo2.object.physical.Instance object at 0x0000015A77C6B820>, 
            'NoName_2': <laygo2.object.physical.Instance object at 0x0000015A77C6B7C0>, 
            'NoName_3': <laygo2.object.physical.Rect object at 0x0000015A77C6B760>, 
            'NoName_4': <laygo2.object.physical.Instance object at 0x0000015A77C6A2F0>, 
            'NoName_5': <laygo2.object.physical.Rect object at 0x0000015A77C6BA90>}
            libname:genlib
            rects:{
            'NoName_0': <laygo2.object.physical.Rect object at 0x0000015A77C6B790>, 
            'NoName_3': <laygo2.object.physical.Rect object at 0x0000015A77C6B760>, 
            'NoName_5': <laygo2.object.physical.Rect object at 0x0000015A77C6BA90>}
            paths:{}
            pins:{}
            texts:{}
            instances:{
            'NoName_1': <laygo2.object.physical.Instance object at 0x0000015A77C6B820>, 
            'NoName_2': <laygo2.object.physical.Instance object at 0x0000015A77C6B7C0>, 
            'NoName_4': <laygo2.object.physical.Instance object at 0x0000015A77C6A2F0>}
            virtual instances:{}
        >>> print(track[-1]) 
            <laygo2.object.physical.Rect object at 0x0000015A77C6BA90> 
            name: None,
            class: Rect,
            xy: [[0, 0], [100, 0]],
            params: None, , layer: ['M2' 'drawing'], netname: None 

        .. image:: ../assets/img/object_grid_RoutingGrid_route_via_track.png
           :height: 250

        See Also
        --------
        laygo2.object.grid.RoutingGrid.route_via_track

        Notes
        -----
        **(Korean)** wire 라우팅 함수, track을 기준점으로 routing을 진행한다.

        파라미터
            - track(numpy.ndarray): track의 좌표값과 방향을 담고 있는 list. 
            수직 트랙일 경우 [v, None], 수평 트랙일 경우 [None, v]의 형태를 
            가지고 있다 (v는 track의 좌표값).
            - mn(list(numpy.ndarray)): track을 통해 연결될 지점들의 좌표를 
            담고 있는 list.
        반환값
            - list: 생성된 routing object들을 담고 있는 list. 
            마지막 object가 track위의 routing object에 해당.
        """
        r = grid.route_via_track(mn=mn, track=track, via_tag=via_tag)
        self.append(r)
        return r

    def pin(self, name, grid, mn, direction=None, netname=None, params=None):
        """
        Create a Pin object over the abstract coordinates specified by mn,
        on the specified routing grid.

        Parameters
        ----------
        name : str
            Pin name.
        mn : numpy.ndarray
            Abstract coordinates for generating Pin.
        direction : str, optional.
            Direction.
        netname : str, optional.
            Net name of Pin.
        params : dict, optional
            Pin attributes.

        Returns
        -------
        laygo2.physical.Pin: The generated pin object.

        Example
        --------
        >>> import laygo2
        >>> from laygo2.object.grid import CircularMapping as CM
        >>> from laygo2.object.grid import CircularMappingArray as CMA
        >>> from laygo2.object.grid import OneDimGrid, RoutingGrid
        >>> from laygo2.object.template import NativeInstanceTemplate
        >>> from laygo2.object.database import Design
        >>> from laygo2.object.physical import Instance
        >>> # Routing grid construction (not needed if laygo2_tech is set up).
        >>> gv = OneDimGrid(name="gv", scope=[0, 50], elements=[0])
        >>> gh = OneDimGrid(name="gv", scope=[0, 100], elements=[0, 40, 60])
        >>> wv = CM([10])           # vertical (xgrid) width
        >>> wh = CM([20, 10, 10])   # horizontal (ygrid) width
        >>> ev = CM([10])           # vertical (xgrid) extension
        >>> eh = CM([10, 10, 10])   # horizontal (ygrid) extension
        >>> e0v = CM([15])          # vert. extension (for zero-length wires)
        >>> e0h = CM([15, 15, 15])  # hori. extension (for zero-length wires)
        >>> lv = CM([['M1', 'drawing']], dtype=object)  # layer information
        >>> lh = CM([['M2', 'drawing']]*3, dtype=object) 
        >>> plv = CM([['M1', 'pin']], dtype=object) # pin layers
        >>> plh = CM([['M2', 'pin']]*3, dtype=object)
        >>> xcolor = CM(['not MPT'], dtype=object)  # 'not MPT' for non-multipatterning 
        >>> ycolor = CM(['not MPT']*3, dtype=object) 
        >>> primary_grid = 'horizontal'
        >>> tvia = NativeInstanceTemplate(libname='tlib', cellname='via0')  # via 
        >>> viamap = CMA(elements=[[tvia, tvia, tvia]], dtype=object)
        >>> g = laygo2.object.grid.RoutingGrid(name='mygrid', vgrid=gv, hgrid=gh,
                                               vwidth=wv, hwidth=wh,
                                               vextension=ev, hextension=eh,
                                               vlayer=lv, hlayer=lh,
                                               pin_vlayer=plv, pin_hlayer=plh,
                                               viamap=viamap, primary_grid=primary_grid,
                                               xcolor=xcolor, ycolor=ycolor,
                                               vextension0=e0v, hextension0=e0h)
        >>> # Create a design
        >>> dsn = Design(name="mycell", libname="genlib")
        >>> mn = [[0, 0], [10, 10]]
        >>> pin = dsn.pin(name="pin", grid=g, mn=mn)
        >>> print(pin)
        <laygo2.object.physical.Pin object at 0x0000028DABE3AB90> 
            name: pin,
            class: Pin,
            xy: [[0, -10], [500, 350]],
            params: None, , layer: ['M2' 'pin'], netname: pin, shape: None, 
            master: None
        >>> print(dsn)
        <laygo2.object.database.Design object at 0x0000028DABE3A110> name: mycell, params: None
            elements: {'pin': <laygo2.object.physical.Pin object at 
            0x0000028DABE3AB90>}
            libname:genlib
            rects:{}
            paths:{}
            pins:{'pin': <laygo2.object.physical.Pin object at 0x0000028DABE3AB90>}
            texts:{}
            instances:{}
            virtual instances:{}

        Notes
        -----
        **(Korean)**
        pin 생성함수.
        
        파라미터
            - name(str): Pin 이름.
            - mn(numpy.ndarray): Pin을 생성할 abstract 좌표.
            - direction(str): 방향 [optional].
            - netname(str): Pin의 net이름 [optional].
            - params(dict): Pin 속성 [optional].
        반환값
            - laygo2.physical.Pin: Pin object.

        """
        p = grid.pin(
            name=name, mn=mn, direction=direction, netname=netname, params=params
        )
        self.append(p)
        return p

    # I/O functions
    def export_to_template(self, libname=None, cellname=None):
        """
        Generate a NativeInstanceTemplate object corresponding to Design object.

        Parameters
        ----------
        libname: str
            The library name.
        cellname: str
            The cell name.

        Returns
        -------
        laygo2.NativeInstanceTemplate: The generated template object.

        Example
        --------
        >>> import laygo2
        >>> from laygo2.object.database import Design
        >>> from laygo2.object.physical import Rect, Pin, Instance, Text
        >>> # Create a design
        >>> dsn = Design(name="mycell", libname="genlib")
        >>> # Create layout objects
        >>> r0 = Rect(xy=[[0, 0], [100, 100]], layer=["M1", "drawing"])
        >>> p0 = Pin(xy=[[0, 0], [50, 50]], layer=["M1", "pin"], name="P")
        >>> i0 = Instance(libname="tlib", cellname="t0", name="I0", xy=[0, 0])
        >>> t0 = Text(xy=[[50, 50], [100, 100]], layer=["text", "drawing"], text="T")
        >>> dsn.append(r0)
        >>> dsn.append(p0)
        >>> dsn.append(i0)
        >>> dsn.append(t0)
        >>> # Export the design to a template.
        >>> nt0 = dsn.export_to_template()
        >>> print(nt0)
        <laygo2.object.template.NativeInstanceTemplate object at XXXX> 
            name: mycell, class: NativeInstanceTemplate, 
            bbox: [[0, 0], [0, 0]], 
            pins: {'P': <laygo2.object.physical.Pin object at YYYY>},
        >>> # Save the template into a yaml file.
        >>> laygo2.interface.yaml.export_template(nt0, filename='mytemp.yaml')

        See Also
        --------
            laygo2.interface.yaml.export_template : Export a template to 
            a yaml file.

        Notes
        -----
        **(Korean)** Design 객체에 해당하는 NativeInstanceTemplate 객체 생성.

        반환값
            - laygo2.NativeInstanceTemplate
        """
        if libname is None:
            libname = self.libname
        if cellname is None:
            cellname = self.cellname

        xy = self.bbox
        pins = self.pins
        return laygo2.object.template.NativeInstanceTemplate(
            libname=libname, cellname=cellname, bbox=xy, pins=pins
        )

    def get_matched_rects_by_layer(self, layer):
        """
        Return a list containing physical objects matched with the layer input in Design object.

        Parameters
        ----------
        layer : list
            The layer information. Format is [name, purpose].

        Returns
        -------
        list: The list containing the matched Physical objects.

        Example
        --------
        >>> dsn    = laygo2.object.Design(name='dsn', libname="testlib")
        >>> rect0  = laygo2.object.Rect(xy=[[0, 0], [100, 100]], layer=[‘M1’, ‘drawing’]……)
        >>> pin0   = laygo2.object.Pin(xy=[[0, 0], [100, 100]], layer=[‘M1’, ‘pin’]……)
        >>> inst0  = laygo2.object.Instance(name=‘I0’, xy=[100, 100]……)
        >>> vinst0_pins[‘in’]  = laygo2.object.physical.Pin(xy=[[0, 0], [10, 10]], layer=[‘M1’,’drawing’]……)
        >>> vinst0_pins[‘out’] = laygo2.object.physical.Pin(xy=[[90, 90], [100, 100]], layer=[‘M1’, drawing’] ……)
        >>> vinst0 = laygo2.object.physical.VirtualInstance(name=‘VI0’, ……)
        >>> text0  = laygo2.object.physical.Text(xy=[[ 0, 0], [100,100 ]], layer=[‘text’, ‘drawing’]……)
        >>> dsn.append(rect0)
        >>> dsn.append(pin0)
        >>> dsn.append(inst0)
        >>> dsn.append(vinst0)
        >>> dsn.append(text0)
        >>> print( dsn.get_matchedrects_by_layer( [“M1”, “drawing”] )
        [<laygo2.object.physical.Rect object>,
         <laygo2.object.physical.Pin object>,
         <laygo2.object.physical.Pin object>,
         <laygo2.object.physical.Rect object>]

        Notes
        -----
        **(Korean)**
        주어진 layer와 일치되는 Physical object 갖는 list 반환.
        파라미터
        layer purpose pair(list): 레이어 정보
        반환값
        list: 매치되는 Physical object를 담고 있는 list
        참조
        없음
        """
        rects = self.rects
        insts = self.instances
        vinsts = self.virtual_instances

        obj_check = []

        for rname, rect in rects.items():
            if np.array_equal(rect.layer, layer):
                obj_check.append(rect)

        for iname, inst in insts.items():
            for pname, pin in inst.pins.items():
                if np.array_equal(pin.layer, layer):
                    obj_check.append(pin)

        for iname, vinst in vinsts.items():
            for name, inst in vinst.native_elements.items():
                if isinstance(inst, laygo2.object.physical.Rect):
                    if np.array_equal(inst.layer, layer):
                        _xy = vinst.get_element_position(inst)
                        ninst = laygo2.object.physical.Rect(
                            xy=_xy,
                            layer=layer,
                            hextension=inst.hextension,
                            vextension=inst.vextension,
                            color=inst.color,
                        )
                        obj_check.append(
                            ninst
                        )  ## ninst is for sort, inst should be frozen for implement to layout
        return obj_check


if __name__ == "__main__":
    from laygo2.object.physical import *

    # Test
    lib = Library(name="mylib")
    dsn = Design(name="mycell")
    lib.append(dsn)
    rect0 = Rect(
        xy=[[0, 0], [100, 100]],
        layer=["M1", "drawing"],
        name="R0",
        netname="net0",
        params={"maxI": 0.005},
    )
    dsn.append(rect0)
    rect1 = Rect(
        xy=[[200, 0], [300, 100]],
        layer=["M1", "drawing"],
        netname="net0",
        params={"maxI": 0.005},
    )
    dsn.append(rect1)
    path0 = Path(
        xy=[[0, 0], [0, 100]],
        width=10,
        extension=5,
        layer=["M1", "drawing"],
        netname="net0",
        params={"maxI": 0.005},
    )
    dsn.append(path0)
    pin0 = Pin(
        xy=[[0, 0], [100, 100]],
        layer=["M1", "pin"],
        netname="n0",
        master=rect0,
        params={"direction": "input"},
    )
    dsn.append(pin0)
    # text0 = Text(xy=[0, 0], layer=['text', 'drawing'], text='test', params=None)
    # dsn.append(text0)
    inst0_pins = dict()
    inst0_pins["in"] = Pin(xy=[[0, 0], [10, 10]], layer=["M1", "drawing"], netname="in")
    inst0_pins["out"] = Pin(
        xy=[[90, 90], [100, 100]], layer=["M1", "drawing"], netname="out"
    )
    inst0 = Instance(
        name="I0",
        xy=[100, 100],
        libname="mylib",
        cellname="mycell",
        shape=[3, 2],
        pitch=[100, 100],
        unit_size=[100, 100],
        pins=inst0_pins,
        transform="R0",
    )
    dsn.append(inst0)
    print(lib)
    print(dsn)
