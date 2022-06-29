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

    Attributes
    ----------
    name : str
    params : dict
    noname_index : int
    elements : dict
    keys : list

    Methods
    -------
    items()
    __getitem__()
    __setitem__()
    __iter__()
    __init__()

    Notes
    -----
    Reference in Korean:
    BaseDatabase는 데이터베이스 객체들의 기본 기능을 구현하는 클래스.
    """

    name = None
    """attribute
    str: BaseDatabase object name.

    

    Examples
    --------
    >>> base = laygo2.object.BaseDatabase(name='mycell’) 
    >>> base.name 
    “mycell”

    Notes
    -----
    Reference in Korean:
    str: BaseDatabase 이름.
    """

    params = None
    """attribute
    dict or None: Dictionary containing parameters of BaseDatabase object.

    

    Examples
    --------
    >>> base = laygo2.object.BaseDatabase(name='mycell’) 
    >>> base.params 
    None

    Notes
    -----
    Reference in Korean:
    dict or None: BaseDatabase의 속성.
    """

    elements = None
    """attribute
    dict: Dictionary having objects.

    

    Examples
    --------
    >>> base = laygo2.object.BaseDatabase(name='mycell’) 
    >>> rect0 = laygo2.object.Rect(name='R0’ ……) 
    >>> rect1 = laygo2.object.Rect(xy=……) 
    >>> pin0 = laygo2.object.Pin(xy=……) 
    >>> inst0 = laygo2.object.Instance(name='I0', xy=[100, 100] ……) 
    >>> base.append(rect0) 
    >>> base.append(rect1) 
    >>> base.append(pin0) 
    >>> base.append(inst0)
    >>> print(base.elements()) 
    {'R0': <laygo2.object.physical.Rect object>, 'NoName_0': <laygo2.object.physical.Rect object>, 'NoName_1': <laygo2.object.physical.Pin object>, 'I0': <laygo2.object.physical.Instance object>}

    Notes
    -----
    Reference in Korean:
    BaseDatabase 객체의 구성 요소를 담고 있는 Dictionary.
    """

    noname_index = 0
    """attribute
    int: A unique number used as the name of an unnamed object belonging to the database.

    

    Examples
    --------
    >>> base = laygo2.object.BaseDatabase(name='mycell’) 
    >>> rect0 = laygo2.object.Rect(name='R0’ ……) 
    >>> base.append(rect0) 
    >>> print(base.noname_index) 
    0 
    >>> rect1 = laygo2.object.Rect(xy=……) 
    >>> base.append(rect1)
    >>> print(base.noname_index) 
    1

    Notes
    -----
    Reference in Korean:
    int: BaseDatabase의 소속 객체들 중 이름이 정해지지 않은 객체의 이름을 사용할 때 부여되는 고유 번호.
    """

    @property
    def keys(self):
        """attribute
        list: Keys of elements.

        

        Examples
        --------
        >>> base = laygo2.object.BaseDatabase(name='mycell’) 
        >>> rect0 = laygo2.object.Rect(name='R0’ ……) 
        >>> rect1 = laygo2.object.Rect(xy=……) 
        >>> pin0 = laygo2.object.Pin(xy=……) 
        >>> inst0 = laygo2.object.Instance(name='I0', xy=[100, 100] ……) 
        >>> base.append(rect0) 
        >>> base.append(rect1) 
        >>> base.append(pin0) 
        >>> base.append(inst0)
        >>> print(base.keys()) 
        ['R0', 'NoName_0', 'NoName_1', 'I0']

        Notes
        -----
        Reference in Korean:
        list: BaseDatabase 객체의 구성 요소를 담고 있는 Dictionary.
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

        

        Examples
        --------
        >>> base = laygo2.object.BaseDatabase(name='mycell’) 
        >>> rect0 = laygo2.object.Rect(name='R0’ ……) 
        >>> rect1 = laygo2.object.Rect(xy=……) 
        >>> pin0 = laygo2.object.Pin(xy=……) 
        >>> inst0 = laygo2.object.Instance(name='I0', xy=[100, 100] ……) 
        >>> base.append(rect0) 
        >>> base.append(rect1) 
        >>> base.append(pin0)
        >>> base.append(inst0) 
        >>> print(base.items()) 
        dict_items([('R0', <laygo2.object.physical.Rect object>),
                    ('NoName_0', <laygo2.object.physical.Rect object>),
                    ('NoName_1', <laygo2.object.physical.Pin object>),
                    ('I0', <laygo2.object.physical.Instance object>)])
        
        Notes
        -----
        Reference in Korean:
        elements의 key/object 짝 출력.
        파라미터
        없음
        반환값
        dict_items
        참조
        없음
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
        element

        

        Examples
        --------
        >>> base = laygo2.object.BaseDatabase(name='mycell’) 
        >>> rect0= laygo2.object.Rect(name='R0’ ……) 
        >>> rect1= laygo2.object.Rect(xy=……) 
        >>> pin0 = laygo2.object.Pin(xy=……) 
        >>> inst0 = laygo2.object.Instance(name='I0', xy=[100, 100] ……) 
        >>> base.append(rect0) 
        >>> base.append(rect1) 
        >>> base.append(pin0) 
        >>> base.append(inst0)
        >>> print(base[“R0”]) 
        <laygo2.object.physical.Rect object> name: R0, class: Rect  ……

        Notes
        -----
        Reference in Korean:
        key에 해당하는 object 반환.
        파라미터
        key(str)
        반환값
        element
        참조
        없음
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

        

        Examples
        --------
        >>> base = laygo2.object.BaseDatabase(name='mycell’) 
        >>> rect0 = laygo2.object.Rect(name='R0’ ……) 
        >>> rect1 = laygo2.object.Rect(xy=……) 
        >>> pin0 = laygo2.object.Pin(xy=……) 
        >>> inst0 = laygo2.object.Instance(name='I0', xy=[100, 100] ……) 
        >>> base.append(rect0) 
        >>> base.append(rect1) 
        >>> base.append(pin0) 
        >>> base.append(inst0) 
        >>> rect2 = laygo2.object.Rect(name=‘R2’ ……)
        >>> base[“R2”]=rect2

        Notes
        -----
        Reference in Korean:
        요소 추가 함수.
        파라미터
        key(str)
        반환값
        list
        참조
        없음
        """
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
        """
        Iterator object of BaseDatabaseParameters.

        Parameters
        ----------
        None

        Returns
        -------
        dict_keyiterator

        

        Examples
        --------
        >>> base = laygo2.object.BaseDatabase(name='mycell’) 
        >>> rect0= laygo2.object.Rect(name='R0’ ……) 
        >>> rect1= laygo2.object.Rect(xy=……) 
        >>> pin0 = laygo2.object.Pin(xy=……) 
        >>> inst0 = laygo2.object.Instance(name='I0', xy=[100, 100] ……) 
        >>> base.append(rect0) 
        >>> base.append(rect1) 
        >>> base.append(pin0) 
        >>> base.append(inst0)
        >>> for obj in base: print(obj) 
        R0 
        NoName_0 
        NoName_1 
        I0

        Notes
        -----
        Reference in Korean:
        BaseDatabase의 Iterable 객체 반환.
        파라미터
        없음
        반환값
        dict_keyiterator
        참조
        없음
        """
        return self.elements.__iter__()

    def __str__(self):
        return self.summarize()

    def summarize(self):
        """Return the summary of the object information."""
        return self.__repr__() + " " + \
               "name: " + self.name + ", " + \
               "params: " + str(self.params) + " \n" \
               "    elements: " + str(self.elements) + \
               ""

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

        

        Examples
        --------
        >>> base = laygo2.object.BaseDatabase(name='mycell’) 
        >>> print(base) 
        <laygo2.object.database.BaseDatabase object> name: mycell, params: None elements: {}>

        Notes
        -----
        Reference in Korean:
        BaseDatabase 클래스 생성자 함수.
        파라미터
        name(str): BaseDatabase 객체의 이름
        params(dict): BaseDatabase의 parameters [optional]
        elements(dict): BaseDatabase의 elements를 갖고 있는 dict [optional]
        반환값
        laygo2.object.BaseDatabase
        참조
        없음
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

    Attributes
    ----------
    name : str
    params : dict
    noname_index : int
    keys : list
    libname : str

    Methods
    -------
    items()
    __getitem__()
    __setitem__()
    __iter__()
    __init__()

    Notes
    -----
    Reference in Korean:
    Library 클래스는 라이브러리 관리 기능을 구현한다.
    """

    def get_libname(self):
        return self.name

    def set_libname(self, val):
        self.name = val

    libname = property(get_libname, set_libname)
    """attribute
    str: Library object name.

    

    Examples
    --------
    >>> lib = laygo2.object.Library(name='mylib’) 
    >>> print(lib.name) 
    “mylib”

    Notes
    -----
    Reference in Korean:
    str: Library 객체의 이름.
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
        """
        Constructor function of Library class.

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

        

        Examples
        --------
        >>> lib = laygo2.object.Library(name='mylib’) 
        >>> print(lib) 
        <laygo2.object.database.Library > name: mylib, params: None elements: {} >

        Notes
        -----
        Reference in Korean:
        Library 클래스의 생성자 함수.
        파라미터
        name(str): Library 객체의 이름
        params(dict): Library의 parameters [optional]
        elements(dict): Library의 elements를 갖고 있는 dict [optional]
        반환값
        laygo2.object.Library
        참조
        없음
        """
        BaseDatabase.__init__(self, name=name, params=params, elements=elements)

    def summarize(self):
        """Return the summary of the object information."""
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
    """
    Design class implements the design management function.

    Attributes
    ----------
    name : str
    params : dict
    noname_index : int
    keys : list
    libname : str
    cellname : str
    rects : dict
    paths : dict
    pins : dict
    texts : dict
    instances : dict
    virtual_instances : dict

    Methods
    -------
    items()
    __getitem__()
    __setitem__()
    __iter__()
    __init__()
    place()
    route()
    route_via_track()
    via()
    pin()
    get_matchedrects_by_layer()
    export_to_template()

    Notes
    -----
    Reference in Korean:
    Design 클래스는 디자인 관리 기능을 구현한다.
    """

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
    """attribute
    str: Library name of Design object.

    

    Examples
    --------
    >>> dsn = laygo2.object.Design(name='dsn', libname="testlib") 
    >>> rect0  = laygo2.object.Rect(xy=[[0, 0], [100, 100]], layer=['M1', 'drawing’]……) 
    >>> pin0   = laygo2.object.Pin(xy=[[0, 0], [100, 100]], layer=['M1', 'pin’]……) 
    >>> inst0  = laygo2.object.Instance(name='I0', xy=[100, 100]……) 
    >>> vinst0 = laygo2.object.physical.VirtualInstance(name='VI0’, ……) 
    >>> text0  = laygo2.object.physical.Text(xy=[[ 0, 0], [100,100 ]], layer=['text', 'drawing’]……) 
    >>> dsn.append(rect0) 
    >>> dsn.append(pin0) 
    >>> dsn.append(inst0) 
    >>> dsn.append(vinst0) 
    >>> dsn.append(text0)
    >>> print(dsn.libname) 
    “testlib”

    Notes
    -----
    Reference in Korean:
    str: Design 객체의 라이브러리 이름.
    """

    def get_cellname(self):
        return self.name

    def set_cellname(self, val):
        self.name = val

    cellname = property(get_cellname, set_cellname)
    """attribute
    str: Cell name of Design object.

    

    Examples
    --------
    >>> dsn = laygo2.object.Design(name='dsn', libname="testlib") 
    >>> rect0  = laygo2.object.Rect(xy=[[0, 0], [100, 100]], layer=['M1', 'drawing’]……) 
    >>> pin0   = laygo2.object.Pin(xy=[[0, 0], [100, 100]], layer=['M1', 'pin’]……) 
    >>> inst0  = laygo2.object.Instance(name='I0', xy=[100, 100]……) 
    >>> vinst0 = laygo2.object.physical.VirtualInstance(name='VI0’, ……) 
    >>> text0  = laygo2.object.physical.Text(xy=[[ 0, 0], [100,100 ]], layer=['text', 'drawing’]……) 
    >>> dsn.append(rect0) 
    >>> dsn.append(pin0) 
    >>> dsn.append(inst0) 
    >>> dsn.append(vinst0) 
    >>> dsn.append(text0)
    >>> print(dsn.cellname) 
    “dsn”

    Notes
    -----
    Reference in Korean:
    str: Design 객체의 셀 이름.
    """
    rects = None
    """attribute
    dict: Dictionary containing Rectangle object affiliated with Design object.

    

    Examples
    --------
    >>> dsn = laygo2.object.Design(name='dsn', libname="testlib") 
    >>> rect0  = laygo2.object.Rect(xy=[[0, 0], [100, 100]], layer=['M1', 'drawing’]……) 
    >>> pin0   = laygo2.object.Pin(xy=[[0, 0], [100, 100]], layer=['M1', 'pin’]……) 
    >>> inst0  = laygo2.object.Instance(name='I0', xy=[100, 100]……) 
    >>> vinst0 = laygo2.object.physical.VirtualInstance(name='VI0’, ……) 
    >>> text0  = laygo2.object.physical.Text(xy=[[ 0, 0], [100,100 ]], layer=['text', 'drawing’]……) 
    >>> dsn.append(rect0) 
    >>> dsn.append(pin0) 
    >>> dsn.append(inst0) 
    >>> dsn.append(vinst0) 
    >>> dsn.append(text0)
    >>> print(dsn.rects) 
    {'R0': <laygo2.object.physical.Rect object>}

    Notes
    -----
    Reference in Korean:
    dict: Design 객체에 소속된 Rect 객체들을 갖고 있는 dictionary.
    """
    paths = None
    pins = None
    """attribute
    dict: Dictionary having the collection of Pin objects affiliated with Design object.

    

    Examples
    --------
    >>> dsn = laygo2.object.Design(name='dsn', libname="testlib") 
    >>> rect0  = laygo2.object.Rect(xy=[[0, 0], [100, 100]], layer=['M1', 'drawing’]……) 
    >>> pin0   = laygo2.object.Pin(xy=[[0, 0], [100, 100]], layer=['M1', 'pin’]……) 
    >>> inst0  = laygo2.object.Instance(name='I0', xy=[100, 100]……) 
    >>> vinst0 = laygo2.object.physical.VirtualInstance(name='VI0’, ……) 
    >>> text0  = laygo2.object.physical.Text(xy=[[ 0, 0], [100,100 ]], layer=['text', 'drawing’]……) 
    >>> dsn.append(rect0) 
    >>> dsn.append(pin0) 
    >>> dsn.append(inst0) 
    >>> dsn.append(vinst0) 
    >>> dsn.append(text0)
    >>> print(dsn.pins) 
    {'NoName_0': <laygo2.object.physical.Pin object>}
    
    Notes
    -----
    Reference in Korean:
    dict: Design 객체에 소속된 Pin 객체들을 갖고 있는 dictionary.
    """
    texts = None
    """attribute
    dict: Dictionary containing Text objects affiliated with Design object.

    

    Examples
    --------
    >>> dsn = laygo2.object.Design(name='dsn', libname="testlib") 
    >>> rect0  = laygo2.object.Rect(xy=[[0, 0], [100, 100]], layer=['M1', 'drawing’]……) 
    >>> pin0   = laygo2.object.Pin(xy=[[0, 0], [100, 100]], layer=['M1', 'pin’]……) 
    >>> inst0  = laygo2.object.Instance(name='I0', xy=[100, 100]……) 
    >>> vinst0 = laygo2.object.physical.VirtualInstance(name='VI0’, ……) 
    >>> text0  = laygo2.object.physical.Text(xy=[[ 0, 0], [100,100 ]], layer=['text', 'drawing’]……) 
    >>> dsn.append(rect0) 
    >>> dsn.append(pin0) 
    >>> dsn.append(inst0) 
    >>> dsn.append(vinst0) 
    >>> dsn.append(text0)
    >>> print(dsn.texts) 
    {'NoName_1': <laygo2.object.physical.Text object>}
    
    Notes
    -----
    Reference in Korean:
    dict: Design 객체에 소속된 Text 객체들을 갖고 있는 dictionary.
    """
    instances = None
    """attribute
    dict: Dictionary containing Instance objects affiliated with Design object.

    

    Examples
    --------
    >>> dsn = laygo2.object.Design(name='dsn', libname="testlib") 
    >>> rect0  = laygo2.object.Rect(xy=[[0, 0], [100, 100]], layer=['M1', 'drawing’]……) 
    >>> pin0   = laygo2.object.Pin(xy=[[0, 0], [100, 100]], layer=['M1', 'pin’]……) 
    >>> inst0  = laygo2.object.Instance(name='I0', xy=[100, 100]……) 
    >>> vinst0 = laygo2.object.physical.VirtualInstance(name='VI0’, ……) 
    >>> text0  = laygo2.object.physical.Text(xy=[[ 0, 0], [100,100 ]], layer=['text', 'drawing’]……) 
    >>> dsn.append(rect0) 
    >>> dsn.append(pin0) 
    >>> dsn.append(inst0) 
    >>> dsn.append(vinst0) 
    >>> dsn.append(text0)
    >>> print(dsn.instances) 
    {'I0': <laygo2.object.physical.Instance object>}
    
    Notes
    -----
    Reference in Korean:
    dict: Design 객체에 소속된 Instance 객체들을 갖고 있는 dictionary.
    """
    virtual_instances = None
    """attribute
    dict: Dictionary containing VirtualInstance objects affiliated with Design object.

    

    Examples
    --------
    >>> dsn = laygo2.object.Design(name='dsn', libname="testlib") 
    >>> rect0  = laygo2.object.Rect(xy=[[0, 0], [100, 100]], layer=['M1', 'drawing’]……) 
    >>> pin0   = laygo2.object.Pin(xy=[[0, 0], [100, 100]], layer=['M1', 'pin’]……) 
    >>> inst0  = laygo2.object.Instance(name='I0', xy=[100, 100]……) 
    >>> vinst0 = laygo2.object.physical.VirtualInstance(name='VI0’, ……) 
    >>> text0  = laygo2.object.physical.Text(xy=[[ 0, 0], [100,100 ]], layer=['text', 'drawing’]……) 
    >>> dsn.append(rect0) 
    >>> dsn.append(pin0) 
    >>> dsn.append(inst0) 
    >>> dsn.append(vinst0) 
    >>> dsn.append(text0)
    >>> print(dsn.virtual_instances) 
    virtual_instnaces {'VI0': <laygo2.object.physical.VirtualInstance object>}

    Notes
    -----
    Reference in Korean:
    dict: Design 객체에 소속된 VirtualInstance 객체들을 갖고 있는 dictionary.
    """

    def __iter__(self):
        """Iterator function. Directly mapped to its elements."""
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

        

        Examples
        --------
        >>> dsn = laygo2.object.Design(name='dsn', libname="testlib") 
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
        Reference in Korean:
        Design 클래스의 생성자 함수.
        파라미터
        name(str): Design 객체의 이름
        params(dict): Design 객체의 parameters [optional]
        elements(dict): Design 객체의 elements [optional]
        반환값
        laygo2.object.BaseDatabase
        참조
        없음
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
        """Return the summary of the object information."""
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
        """Place an instance on the specified coordinate mn, on this grid."""
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
        """Create Path and Via objects over the abstract coordinates specified by mn, on this routing grid. """
        r = grid.route(mn=mn, direction=direction, via_tag=via_tag)
        self.append(r)
        return r

    def route_via_track(self, grid, mn, track, via_tag=[None, True]):
        """Create Path and Via objects over the abstract coordinates specified by mn, 
        on the track of specified routing grid. """
        r = grid.route_via_track(mn=mn, track=track, via_tag=via_tag)
        self.append(r)
        return r

    def via(self, grid, mn, params=None):
        """Create a Via object over the abstract coordinates specified by mn, on this routing grid. """
        v = grid.via(mn=mn, params=params)
        self.append(v)
        return v

    def pin(self, name, grid, mn, direction=None, netname=None, params=None):
        """Create a Pin object over the abstract coordinates specified by mn, on this routing grid. """
        p = grid.pin(name=name, mn=mn, direction=direction, netname=netname, params=params)
        self.append(p)
        return p

    # I/O functions
    def export_to_template(self, libname=None, cellname=None):
        """
        Generate NativeInstanceTemplate object corresponding to Design object.

        Parameters
        ----------
        None

        Returns
        -------
        laygo2.NativeInstanceTemplate

        

        Examples
        --------
        >>> dsn    = laygo2.object.Design(name='dsn', libname="testlib") 
        >>> rect0  = laygo2.object.Rect(xy=[[0, 0], [100, 100]], layer=['M1', 'drawing’]……) 
        >>> pin0   = laygo2.object.Pin(xy=[[0, 0], [100, 100]], layer=['M1', 'pin’]……) 
        >>> inst0  = laygo2.object.Instance(name='I0', xy=[100, 100]……) 
        >>> vinst0 = laygo2.object.physical.VirtualInstance(name='VI0’, ……) 
        >>> text0  = laygo2.object.physical.Text(xy=[[ 0, 0], [100,100 ]], layer=['text', 'drawing’]……) 
        >>> dsn.append(rect0) 
        >>> dsn.append(pin0)
        >>> dsn.append(inst0) 
        >>> dsn.append(vinst0) 
        >>> dsn.append(text0) 
        >>> print(dsn.export_to_template()) 
        <laygo2.object.template.NativeInstanceTemplate object> name: dsn,
         class: NativeInstanceTemplate,
         bbox: [[100, 100], [800, 700]],
         pins: {'NoName_0': <laygo2.object.physical.Pin object>}

        Notes
        -----
        Reference in Korean:
        Design 객체에 해당하는 NativeInstanceTemplate 객체 생성.
        파라미터
        없음
        반환값
        laygo2.NativeInstanceTemplate
        참조
        없음
        """
        if libname is None:
            libname = self.libname
        if cellname is None:
            cellname = self.cellname

        xy   = self.bbox
        pins = self.pins
        return laygo2.object.template.NativeInstanceTemplate(libname=libname, cellname=cellname, bbox=xy, pins=pins)

    def get_matchedrects_by_layer(self, lpp ):
        """
        Return a list containing physical objects matched with the layer input in Design object.

        Parameters
        ----------
        layer purpose pair : list
            layer information.

        Returns
        -------
        list: list containing the matched Physical object.

        

        Examples
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
        Reference in Korean:
        주어진 layer와 일치되는 Physical object 갖는 list 반환.
        파라미터
        layer purpose pair(list): 레이어 정보
        반환값
        list: 매치되는 Physical object를 담고 있는 list
        참조
        없음
        """
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
                        _xy   = vinst.get_element_position(inst)
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