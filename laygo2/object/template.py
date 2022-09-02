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
This module implements classes for various layout template objects.
"""

__author__ = "Jaeduk Han"
__maintainer__ = "Jaeduk Han"
__status__ = "Prototype"

from abc import *
import numpy as np
import laygo2.object


class Template(metaclass=ABCMeta):
    """
    The base class that defines the functions and attributes of the template.

    Attributes
    ----------
    name : str

    Methods
    -------
    __init__() 
    width()
    size() 
    bbox() 
    pins() 
    generate() 

    Notes
    -----
    Reference in Korean:
    Template의 기본동작과 속성을 정의한 기본 클래스.
    """

    name = None
    """str: Template name."""

    def __init__(self, name=None):
        """
        Constructor function of Template class.

        Parameters
        ----------
        name : str or None, optional.
            The name of this template.
        """
        self.name = name

    def __str__(self):
        """Return a string corresponding to this object's information."""
        return self.summarize()

    def summarize(self):
        """Return the summary of the template information."""
        return self.__repr__() + " " \
                                 "name: " + self.name + ", " + \
                                 "class: " + self.__class__.__name__ + ", " + \
                                 ""

    def height(self, params=None):
        """int: Return the height of a template."""
        return abs(self.bbox(params=params)[0, 1] - self.bbox(params=params)[1, 1])

    def width(self, params=None):
        """int: Return the width of a template."""
        return abs(self.bbox(params=params)[0, 0] - self.bbox(params=params)[1, 0])

    def size(self, params=None):
        """int: Return the size of a template."""
        return np.array([self.width(params=params), self.height(params=params)])

    @abstractmethod
    def bbox(self, params=None):
        """numpy.ndarray: (Abstract method) Return the bounding box of a template."""
        pass

    @abstractmethod
    def pins(self, params=None):
        """dict: (Abstract method) Return dict having the collection of pins of a template."""
        pass

    @abstractmethod
    def generate(self, name=None, shape=None, pitch=None, transform='R0', params=None):
        """instance: (Abstract method) Return the instance generated from a template."""
        pass


class NativeInstanceTemplate(Template):
    """
    NativeInstanceTemplate class implements the template that generate Instance.

    Attributes
    ----------
    name : str
    libname : str
    cellname : str

    Methods
    -------
    __init__() 
    height() 
    width() 
    size() 
    bbox() 
    pins() 
    generate() 
    export_to_dict() 

    Notes
    -----
    Reference in Korean:
    NativeInstanceTemplate 클래스는 기본 Instance를 반환하는 템플릿을 구현한다.
    """
    libname = None
    """attribute
    str: Library name of NativeInstanceTemplate object.

    

    Examples
    --------
    >>> nat_temp_pins = dict() >>> nat_temp_pins['in'] = laygo2.object.Pin(xy=[[0, 0], [10, 10]], layer=['M1', 'drawing’], netname='in’)
    >>> nat_temp_pins['out'] = laygo2.object.Pin(xy=[[90, 90], [100, 100]], layer=['M1', 'drawing’], netname='out’)
    >>> nat_temp = laygo2.object.NativeInstanceTemplate(libname='mylib', cellname='mynattemplate’, bbox=[[0, 0], [100, 100]], pins=nat_temp_pins)
    >>> nat_temp.libname 
    “mylib”

    Notes
    -----
    Related Images:
    https://github.com/niftylab/laygo2/tree/master/docs_workspace/assets/img/object_template_NativeInstanceTemplate_libname.png

    Reference in Korean:
    str: NativeInstanceTemplate 객체의 library 이름.
    """

    cellname = None
    """attribute
    str: Cellname of NativeInstanceTemplate object.

    

    Examples
    --------
    >>> nat_temp_pins = dict() >>> nat_temp_pins['in']  = laygo2.object.Pin(xy=[[0, 0], [10, 10]], layer=['M1', 'drawing’], netname='in’)
    >>> nat_temp_pins['out'] = laygo2.object.Pin(xy=[[90, 90], [100, 100]], layer=['M1', 'drawing’], netname='out’)
    >>> nat_temp = laygo2.object.NativeInstanceTemplate(libname='mylib', cellname='mynattemplate’, bbox=[[0, 0], [100, 100]], pins=nat_temp_pins)
    >>> nat_temp.cellname 
    “mynattemplate”

    Notes
    -----
    Related Images:
    https://github.com/niftylab/laygo2/tree/master/docs_workspace/assets/img/object_template_NativeInstanceTemplate_cellname.png

    Reference in Korean:
    str: NativeInstanceTemplate 객체의 cellname.
    """

    _bbox = np.array([[0, 0], [0, 0]])
    
    _pins = None

    def __init__(self, libname, cellname, bbox=np.array([[0, 0], [0, 0]]), pins=None):
        """
        Constructor function of NativeInstanceTemplate class.

        Parameters
        ----------
        libname : str
            library name.
        cellname : str
            cell name.
        bbox : numpy.ndarray
            bbox.
        pins : dict
            dictionary having the pin object.

        Returns
        -------
        NativeInstanceTemplate

        

        Examples
        --------
        >>> nat_temp_pins = dict() 
        >>> nat_temp_pins['in']  = laygo2.object.Pin(xy=[[0, 0], [10, 10]], layer=['M1', 'drawing’], netname='in’)
        >>> nat_temp_pins['out'] = laygo2.object.Pin(xy=[[90, 90], [100, 100]], layer=['M1', 'drawing’], netname='out’)
        >>> nat_temp = laygo2.object.NativeInstanceTemplate(libname='mylib', cellname='mynattemplate’, bbox=[[0, 0], [100, 100]], pins=nat_temp_pins)
        <laygo2.object.template.NativeInstanceTemplate object>

        Notes
        -----
        Related Images:
        https://github.com/niftylab/laygo2/tree/master/docs_workspace/assets/img/object_template_NativeInstanceTemplate_init.png

        Reference in Korean:
        NativeInstanceTemplate 클래스의 생성자함수.
        파라미터
        libname(str): library 이름
        cellname(str): cell 이름
        bbox(numpy.ndarray): bbox
        pins(dict): pin 객체를 갖고있는 dictionary
        반환값
        laygo2.NativeInstanceTemplate
        참조
        없음
        """
        self.libname = libname
        self.cellname = cellname
        self._bbox = None if bbox is None else np.asarray(bbox)
        self._pins = pins
        Template.__init__(self, name=cellname)

    def summarize(self):
        """Return the summary of the template information."""
        return self.__repr__() + " " \
                                 "name: " + self.name + ", " + \
               "class: " + self.__class__.__name__ + ", " + \
               "bbox: " + str(self.bbox().tolist()) + ", " + \
               "pins: " + str(self.pins()) + ", " + \
               ""

    # Core template functions
    def bbox(self, params=None):
        """
        bbox of NativeInstanceTemplate object.

        Parameters
        ----------
        None

        Returns
        -------
        numpy.ndarray

        

        Examples
        --------
        >>> nat_temp_pins = dict() 
        >>> nat_temp_pins['in']  = laygo2.object.Pin(xy=[[0, 0], [10, 10]], layer=['M1', 'drawing’], netname='in’)
        >>> nat_temp_pins['out'] = laygo2.object.Pin(xy=[[90, 90], [100, 100]], layer=['M1', 'drawing’], netname='out’)
        >>> nat_temp = laygo2.object.NativeInstanceTemplate(libname='mylib', cellname='mynattemplate’, bbox=[[0, 0], [100, 100]], pins=nat_temp_pins)
        >>> nat_temp.bbox() 
        [[0,0], [100, 100]]

        Notes
        -----
        Related Images:
        https://github.com/niftylab/laygo2/tree/master/docs_workspace/assets/img/object_template_NativeInstanceTemplate_bbox.png

        Reference in Korean:
        NativeInstanceTemplate 객체의 bbox.
        파라미터
        없음
        반환값
        numpy.ndarray
        참조
        없음
        """
        return self._bbox

    def pins(self, params=None):
        """
        Return pin dictionary of NativeInstanceTemplate object.

        Parameters
        ----------
        None

        Returns
        -------
        dict

        

        Examples
        --------
        >>> nat_temp_pins = dict() 
        >>> nat_temp_pins['in']  = laygo2.object.Pin(xy=[[0, 0], [10, 10]], layer=['M1', 'drawing’], netname='in’)
        >>> nat_temp_pins['out'] = laygo2.object.Pin(xy=[[90, 90], [100, 100]], layer=['M1', 'drawing’], netname='out’)
        >>> nat_temp = laygo2.object.NativeInstanceTemplate(libname='mylib', cellname='mynattemplate’, bbox=[[0, 0], [100, 100]], pins=nat_temp_pins)
        >>> nat_temp.pins() 
        {'in': <laygo2.object.physical.Pin object>, 'out': <laygo2.object.physical.Pin object>}

        Notes
        -----
        Related Images:
        https://github.com/niftylab/laygo2/tree/master/docs_workspace/assets/img/object_template_ParameterizedInstanceTemplate_pins.png

        Reference in Korean:
        NativeInstanceTemplate 객체의 pin dictionary 반환.
        파라미터
        없음
        반환값
        dict
        참조
        없음
        """
        return self._pins

    def generate(self, name=None, shape=None, pitch=None, transform='R0', params=None):
        """
        Generate Instance object.

        Parameters
        ----------
        name : str 
            name of the instance to be generated.
        shape : numpy.ndarray, optional.
            shape of the object to be generated.
        pitch : numpy.ndarray, optional.
            pitch of the object to be generated.
        params : dict, optional.
            dictionary having the object attributes.
        transform : str
            transformation attribute of the object to be generated.
        
        Returns
        -------
        (laygo2.object.physical.Instance) generated Instance object

        See Also
        --------
        Class Instance

        Examples
        --------
        >>> nat_temp_pins = dict()
        >>> nat_temp_pins['in'] = laygo2.object.Pin(xy=[[0, 0], [10, 10]], layer=['M1', 'drawing'], netname='in')
        >>> nat_temp_pins['out'] = laygo2.object.Pin(xy=[[90, 90], [100, 100]], layer=['M1', 'drawing'], netname='out')
        >>> nat_temp = laygo2.object.NativeInstanceTemplate(libname='mylib', cellname='mynattemplate', bbox=[[0, 0], [100, 100]], pins=nat_temp_pins)
        >>> nat_temp.generate(name="I1")
        <laygo2.object.physical.Instance object>
        >>> nat_temp.generate(name="I2")
        <laygo2.object.physical.Instance object>

        Notes
        -----
        Related Images:
        https://github.com/niftylab/laygo2/tree/master/docs_workspace/assets/img/object_template_ParameterizedInstanceTemplate_generate.png

        Reference in Korean:
        Instance 객체 생성.
        파라미터
        name(str): 생성할 인스턴스의 이름
        shape(numpy.ndarray): 생성할 객체의 shape [ optional ]
        pitch(numpy.ndarray): 생성할 객체간의 간격 [ optional ]
        params(dict) : 객체의 속성을 갖는 Dictionary [ optional ]
        transform(str): 생성할 객체의 변환 속성 [ optional ]
        반환값
        laygo2.Instance: 생성된 개체
        참조
        Class Instance
        """
        return laygo2.object.physical.Instance(libname=self.libname, cellname=self.cellname, xy=np.array([0, 0]),
                                               shape=shape, pitch=pitch, unit_size=self.size(params), pins=self.pins(params),
                                               transform=transform, name=name, params=params)

    # I/O functions
    def export_to_dict(self):
        """
        Return Dictionary containing the information of NativeInstanceTemplate.

        Parameters
        ----------
        None

        Returns
        -------
        dict

        See Also
        --------
        Class Instance

        Examples
        --------
        >>> nat_temp_pins = dict()
        >>> nat_temp_pins['in'] = laygo2.object.Pin(xy=[[0, 0], [10, 10]], layer=['M1', 'drawing'], netname='in')
        >>> nat_temp_pins['out'] = laygo2.object.Pin(xy=[[90, 90], [100, 100]], layer=['M1', 'drawing'], netname='out')
        >>> nat_temp = laygo2.object.NativeInstanceTemplate(libname='mylib', cellname='mynattemplate', bbox=[[0, 0], [100, 100]], pins=nat_temp_pins)
        >>> nat_temp.export_to_dict()
        {'libname': 'mylib', 'cellname': 'mynattemplate', 'bbox': [[0, 0], [100, 100]], 'pins': {'in': {'xy': [[0, 0], [10, 10]], 'layer': ['M1', 'drawing'], 'name': None, 'netname': 'in'}, 'out': {'xy': [[90, 9 0], [100, 100]], 'layer': ['M1', 'drawing'], 'name': None, 'netname': 'out'}}}

        Notes
        -----
        Related Images:
        https://github.com/niftylab/laygo2/tree/master/docs_workspace/assets/img/object_template_NativeInstanceTemplate_export_to_dict.png

        Reference in Korean:
        NativeInstanceTemplate 의 정보가 담긴 Dictonary 반환.
        파라미터
        없음
        반환값
        dict
        참조
        Class Instance
        """
        db = dict()
        db['libname'] = self.libname
        db['cellname'] = self.cellname
        db['bbox'] = self.bbox().tolist()
        db['pins'] = dict()
        for pn, p in self.pins().items():
            db['pins'][pn] = p.export_to_dict()
        return db


class ParameterizedInstanceTemplate(Template):
    """
    ParameterizedInstanceTemplate class implements the template that generate ParameterizedInstnace.

    Attributes
    ----------
    name : str
    libname : str
    cellname : str

    Methods
    -------
    __init__() 
    height() 
    width() 
    size() 
    bbox() 
    pins() 
    generate()

    Notes
    -----
    Reference in Korean:
    ParameterizedInstanceTemplate 클래스는 Parameterized Instance를 반환하는 템플릿을 구현한다.
    """

    libname = None
    """str: Libname of the instance being generated."""

    cellname = None
    """str: Cellname of the instance being generated."""

    _bbox = None

    _pins = None

    def __init__(self, libname, cellname, bbox_func=None, pins_func=None):
        """
        Generate ParameterizedInstanceTemplate object.

        Parameters
        ----------
        libname : str
            library name.
        cellname : str 
            The cell name of the template. 
        bbox_func : callable
            bbox.
        pins_func : callable
            dictionary having the pin object.
        
        Returns
        -------
        laygo2.NativeInstanceTemplate

        

        Examples
        --------
        >>> def pcell_bbox_func(params): 
            …… 
        >>> def pcell_pins_func(params): 
            ……
        >>> pcell_temp = laygo2.object.ParameterizedInstanceTemplate(libname='mylib', cellname='mynattemplate’, bbox_func= pcell_bbox_func, pins_func= pcell_pins_func)
        <laygo2.object.template.ParameterizedInstanceTemplate object>
        
        Notes
        -----
        Related Images:
        https://github.com/niftylab/laygo2/tree/master/docs_workspace/assets/img/object_template_ParameterizedInstanceTemplate_init.png

        Reference in Korean:
        ParameterizedInstanceTemplate 클래스의 생성자함수.
        파라미터
        libname(str): library 이름 
        cellname(str): cell 이름 
        bbox_func(callable): bbox 
        pins_func(callable): pin 객체를 갖고있는 dictionary
        반환값
        laygo2.NativeInstanceTemplate
        참조
        없음
        """
        self.libname = libname
        self.cellname = cellname
        self._bbox = bbox_func
        self._pins = pins_func
        Template.__init__(self, name=cellname)

    # Core template functions
    def bbox(self, params=None):
        """
        bbox of ParameterizedInstanceTemplate object.

        Parameters
        ----------
        None

        Returns
        -------
        numpy.ndarray

        

        Examples
        --------
        >>> def pcell_bbox_func(params):
                if params==None: 
                    params={“W”:1}
                return np.array([[0, 0], [100 , 100* params['W']]])
        >>> def pcell_pins_func(params): 
            ……
        >>> pcell_temp = laygo2.object.ParameterizedInstanceTemplate(libname='mylib', cellname='mynattemplate’, bbox_func= pcell_bbox_func, pins_func= pcell_pins_func)
        >>> pcell_temp.bbox 
        [[0,0], [100,100]]

        Notes
        -----
        Related Images:
        https://github.com/niftylab/laygo2/tree/master/docs_workspace/assets/img/object_template_ParameterizedInstanceTemplate_bbox.png

        Reference in Korean:
        ParameterizedInstanceTemplate 객체의 bbox.
        파라미터
        없음
        반환값
        numpy.ndarray
        참조
        없음
        """
        return self._bbox(params=params)

    def pins(self, params=None):
        """
        Return pin dictionary of ParameterizedInstanceTemplate object.

        Parameters
        ----------
        None

        Returns
        -------
        dict

        

        Examples
        --------
        >>> def pcell_bbox_func(params): 
            ……
        >>> def pcell_pins_func(params):
                if params==None:
                    params={"W":1} 
                    i = params['W']
                template_pins = dict()
                pin_in  = laygo2.object.Pin(xy =[ [ 0, 0], [100 , 0 ] ],      layer=['M1', 'pin'], netname='in')
                pin_out = laygo2.object.Pin(xy =[ [ 0, 100], [100 , 100* i]], layer=['M1', 'pin’], netname='out')
                template_pins['in' ] = pin_in
                template_pins['out'] = pin_out
                return template_pins
        >>> pcell_temp = laygo2.object.ParameterizedInstanceTemplate(libname='mylib', cellname='mynattemplate’, bbox_func= pcell_bbox_func, pins_func= pcell_pins_func)
        >>> pcell_temp.pins 
        {'in': <laygo2.object.physical.Pin object>, 'out': <laygo2.object.physical.Pin object>}

        Notes
        -----
        Related Images:
        https://github.com/niftylab/laygo2/tree/master/docs_workspace/assets/img/object_template_ParameterizedInstanceTemplate_pins.png

        Reference in Korean:
        ParameterizedInstanceTemplate 객체의 pin dictionary 반환.
        파라미터
        없음
        반환값
        dict
        참조
        없음
        """
        return self._pins(params=params)

    def generate(self, name=None, shape=None, pitch=None, transform='R0', params=None):
        """
        Generate ParameterizedInstance object.

        Parameters
        ----------
        name : str
            name of the instance to be generated.
        shape : numpy.ndarray, optional.
            shape of the object to be generated.
        pitch : numpy.ndarray, optional.
            pitch of the object to be generated.
        params : dict, optional.
            dictionary having the entity attributes.
        transform : str, optional.
            transformation attribute of the entity to be generated.
        
        Returns
        -------
        (laygo2.object.physical.Instance) generated Instance object

        See Also
        --------
        Class Instance

        Examples
        --------
        >>> def pcell_bbox_func(params):
            ……
        >>> def pins_bbox_func(params): 
            ……
        >>> pcell_temp = laygo2.object.ParameterizedInstanceTemplate(libname='mylib', cellname='mynattemplate’, bbox_func=pcell_bbox_func, pins_func=pcell_pins_func)
        >>> pcell_temp.generate(name=“I1”, params={“W”=2, “L”=1}) 
        <laygo2.object.physical.Instance object>
        >>> pcell_temp.generate(name=“I2”, params={“W”=2, “L”=1}) 
        <laygo2.object.physical.Instance object>

        Notes
        -----
        Related Images:
        https://github.com/niftylab/laygo2/tree/master/docs_workspace/assets/img/object_template_ParameterizedInstanceTemplate_generate.png

        Reference in Korean:
        ParameterizedInstance 객체 생성.
        파라미터
        name(str): 생성할 인스턴스의 이름
        shape(numpy.ndarray): 생성할 객체의 shape [ optional ]
        pitch(numpy.ndarray): 생성할 객체간의 간격 [ optional ]
        params(dict) : 개체의 속성을 갖는 Dictionary [ optional ]
        transform(str):  생성할 개체의 변환 속성 [ optional ]
        반환값
        laygo2.Instance: 생성된 객체
        참조
        Class Instance
        """
        #xy = xy + np.dot(self.xy(params)[0], tf.Mt(transform).T)
        return laygo2.object.physical.Instance(libname=self.libname, cellname=self.cellname, xy=np.array([0, 0]),
                                               shape=shape, pitch=pitch, unit_size=self.size(params),
                                               pins=self.pins(params), transform=transform, name=name, params=params)


class UserDefinedTemplate(Template):
    """
    UserDefinedTemplate class implements the template that generate VirtualInstance.

    Attributes
    ----------
    name : str

    Methods
    -------
    __init__() 
    height() 
    width() 
    size() 
    bbox() 
    pins() 
    generate() 

    Notes
    -----
    Reference in Korean:
    UserDefinedTemplate 클래스는 VirtualInstance를 반환하는 템플릿을 구현한다.
    """

    _bbox = None

    _pins = None

    _generate = None

    def __init__(self, bbox_func, pins_func, generate_func, name=None):
        """
        Constructor function of UserDefinedTemplate class.

        Parameters
        ----------
        bbox_func: callable
            method computing bbox.
        pins_func: callable
            method computing pins.
        generate_func: callable
            method generating VirtualInstance.
        name : str
            template name.

        Returns
        -------
        laygo2.UserDefinedTemplate

        

        Examples
        --------
        >>> def user_bbox_func(params):-> numpy.ndarray …  ## return bbox0 * multi 
            …… 
        >>> def user_pins_func(params):-> dict          …  ## pin0.bbox = pins0.bbox * multi 
            …… 
        >>> def user_generate_func(params): 
            …… 
        >>> user_temp = laygo2.object.UserDefinedTemplate(name='myusertemplate', bbox_func=user_bbox_func, pins_func=user_pins_func, generate_func=user_generate_func)
        <laygo2.object.template.UserDefinedTemplate object>

        Notes
        -----
        Related Images:
        https://github.com/niftylab/laygo2/tree/master/docs_workspace/assets/img/object_template_UserDefinedTemplate_init.png

        Reference in Korean:
        UserDefinedTemplate 클래스의 생성자함수.
        파라미터
        bbox_func(callable ): bbox를 연산해주는 메소드
        pins_func(callable ): pins를 연산해주는 메소드
        generate_func(callable ): VirtualInstance를 생성하는 메소드
        name(str): 템플릿 이름
        반환값
        laygo2.UserDefinedTemplate
        참조
        없음
        """
        self._bbox = bbox_func
        self._pins = pins_func
        self._generate = generate_func
        Template.__init__(self, name=name)

    # Core template functions
    def bbox(self, params=None):
        """
        Return bbox of UserDefinedTemplate object.

        Parameters
        ----------
        None

        Returns
        -------
        numpy.ndarray

        

        Examples
        --------
        >>> params={}; params[“multi”] = 10; bbox0 = [ [0,0],[100,100]]; pin0 = [ in, out ] 
        >>> def user_bbox_func(params):-> numpy.ndarray …  ## return bbox0 * multi
                if params==None: 
                    params={}
                    params['multi'] = 1
                return np.array([[0, 0], [100 * params['multi'], 100]])
        >>> def user_pins_func(params):-> dict          …  ## pin0.bbox = pins0.bbox * multi 
            …… 
        >>> def user_generate_func(params): 
            …… 
        >>> user_temp = laygo2.object.UserDefinedTemplate(name='myusertemplate', bbox_func=user_bbox_func, pins_func=user_pins_func, generate_func=user_generate_func)
        >>> user_temp.bbox() 
        [[0, 0], [100, 100]]

        Notes
        -----
        Related Images:
        https://github.com/niftylab/laygo2/tree/master/docs_workspace/assets/img/object_template_UserDefinedTemplate_bbox.png

        Reference in Korean:
        UserDefinedTemplate 객체의 bbox 반환.
        파라미터
        없음
        반환값
        numpy.ndarray
        참조
        없음
        """
        return self._bbox(params=params)

    def pins(self, params=None):
        """
        Pins of UserDefinedTemplate object.

        Parameters
        ----------
        None

        Returns
        -------
        dict

        

        Examples
        --------
        >>> params={}; params[“multi”] = 10; bbox0 = [ [0,0],[100,100]]; pin0 = [ in, out ] 
        >>> def user_bbox_func(params):-> numpy.ndarray …  ## return bbox0 * multi 
        >>> def user_pins_func(params):-> dict
                if params==None:
                    params={"multi":1}
                i = params['multi']
                template_pins = dict()
                pin_in = laygo2.object.Pin(xy =[ [ 0, 0], [100 * i, 0 ] ],   layer=['M1', 'pin'], netname='in') 
                pin_out = laygo2.object.Pin(xy=[ [ 0, 100], [100 * i, 100]], layer=['M1', 'pin'], netname='out') 
                template_pins['in' ] = pin_in 
                template_pins['out'] = pin_out 
                return template_pins
        >>> def user_generate_func(params):
            …… 
        >>> user_temp = laygo2.object.UserDefinedTemplate(name='myusertemplate', bbox_func=user_bbox_func, pins_func=user_pins_func, generate_func=user_generate_func)
        >>> user_temp.pins()
        {'in': <laygo2.object.physical.Pin>, 'out': <laygo2.object.physical.Pin object>}

        Notes
        -----
        Related Images:
        https://github.com/niftylab/laygo2/tree/master/docs_workspace/assets/img/object_template_UserDefinedTemplate_pins.png

        Reference in Korean:
        UserDefinedTemplate 객체의 pins.
        파라미터
        없음
        반환값
        dict
        참조
        없음
        """
        return self._pins(params=params)

    def generate(self, name=None, shape=None, pitch=None, transform='R0', params=None):
        """
        Generate a VirtualInstance object by calling generate_func() bound to the template.

        Parameters
        ----------
        name : str
            name of the instance to be generated.
        shape : numpy.ndarray, optional.
            shape of the object to be generated.
        pitch : numpy.ndarray, optional.
            pitch of the object to be generated.
        params : dict, optional.
            dictionary having the entity attributes.
        transform : str, optional.
            transformation attribute of the entity to be generated.

        Returns
        -------
        (laygo2.object.physical.VirtualInstance) generated VirtualInstance object.

        Examples
        --------
        >>> def user_bbox_func(params):-> numpy.ndarray …  ## return bbox0 * multi 
        >>> def user_pins_func(params):-> dict 
        >>> def user_generate_func(params={“multi”:1}):
                m = params['multi'] 
                shape = np.array([1, 1]); inst_pins = user_pins_func(params) ; inst_native_elements = dict() 
                inst_native_elements['left'] = laygo2.object.Rect(xy=[ [0, 0], [0,100]], layer=['M1’, 'drawing']) 
                ofst = np.array([100, 0])
                for i in range(m):
                    bl  = np.array([0,0]) 
                    tr  = np.array([100,100]) 
                    inst_native_elements['center'+str(i)] = laygo2.object.Rect(xy=[ i*ofst + bl , i*ofst + tr ]…
                inst_native_elements['right'] = laygo2.object.Rect(xy=[ m*ofst + [0, 0], m*ofst + [ 0 , 100]], …) 
                inst = VirtualInstance(name=name,xy=np.array([0, 0]),native_elements=inst_native_elements,……)
        >>> user_temp = laygo2.object.UserDefinedTemplate(name='myusertemplate', bbox_func=user_bbox_func, pins_func=user_pins_func, generate_func=user_generate_func) 
        >>> nat_temp.generate(name=“I1”, {“multi”=1}) 
        <laygo2.object.physical.VirtualInstance >
        >>> nat_temp.generate(name=“I2”, {“multi”=2}) 
        <laygo2.object.physical.VirtualInstance >

        Notes
        -----
        Related Images:
        https://github.com/niftylab/laygo2/tree/master/docs_workspace/assets/img/object_template_UserDefinedTemplate_generate.png

        Reference in Korean:
        VirtualInstance 객체 생성.
        파라미터
        없음
        반환값
        laygo2.VirtualInstance
        참조
        없음
        """
        return self._generate(name=name, shape=shape, pitch=pitch, transform=transform, params=params)


# Test
if __name__ == '__main__':
    test_native_template = True
    test_pcell_template = True
    test_user_template = True

    import laygo2.object

    if test_native_template:
        print("NativeInstanceTemplate test")
        # define pins
        nat_temp_pins = dict()
        nat_temp_pins['in'] = laygo2.object.Pin(xy=[[0, 0], [10, 10]], layer=['M1', 'drawing'], netname='in')
        nat_temp_pins['out'] = laygo2.object.Pin(xy=[[90, 90], [100, 100]], layer=['M1', 'drawing'], netname='out')
        # create a template
        nat_temp = NativeInstanceTemplate(libname='mylib', cellname='mynattemplate', bbox=[[0, 0], [100, 100]],
                                          pins=nat_temp_pins)
        # generate
        nat_inst = nat_temp.generate(name='mynatinst', shape=[2, 2], pitch=[100, 100], transform='R0')
        # display
        print(nat_temp)
        print(nat_inst)

    if test_pcell_template:
        print("ParameterizedInstanceTemplate test")

        # define the bbox computation function.
        def pcell_bbox_func(params):
            return np.array([[0, 0], [100 * params['mult'], 100]])

        # define the pin generation function.
        def pcell_pins_func(params):
            template_pins = dict()
            for i in range(params['mult']):
                template_pins['in' + str(i)] = laygo2.object.Pin(xy=[[i * 100 + 0, 0], [i * 100 + 10, 10]],
                                                                 layer=['M1', 'drawing'], netname='in' + str(i))
                template_pins['out' + str(i)] = laygo2.object.Pin(xy=[[i * 100 + 90, 90], [i * 100 + 90, 100]],
                                                                  layer=['M1', 'drawing'], netname='out' + str(i))
            return template_pins

        # create a template.
        pcell_temp = ParameterizedInstanceTemplate(libname='mylib', cellname='mypcelltemplate',
                                                   bbox_func=pcell_bbox_func, pins_func=pcell_pins_func)
        # generate based on the parameter assigned.
        pcell_inst_params = {'mult': 4}
        pcell_inst_size = pcell_temp.size(params=pcell_inst_params)
        pcell_inst = pcell_temp.generate(name='mypcellinst', shape=[2, 2], pitch=pcell_inst_size, transform='R0',
                                         params=pcell_inst_params)
        # display
        print(pcell_temp)
        print(pcell_inst)

    if test_user_template:
        print("UserDefinedTemplate test")

        # define the bbox computation function.
        def user_bbox_func(params):
            return np.array([[0, 0], [100 * params['mult'], 100]])

        # define the pin generation function.
        def user_pins_func(params):
            template_pins = dict()
            for i in range(params['mult']):
                template_pins['in' + str(i)] = laygo2.object.Pin(xy=[[i * 100 + 0, 0], [i * 100 + 10, 10]],
                                                                 layer=['M1', 'drawing'], netname='in' + str(i))
                template_pins['out' + str(i)] = laygo2.object.Pin(xy=[[i * 100 + 90, 90], [i * 100 + 90, 100]],
                                                                  layer=['M1', 'drawing'], netname='out' + str(i))
            return template_pins

        # define the instance generation function.
        def user_generate_func(name=None, shape=None, pitch=np.array([0, 0]), transform='R0', params=None):
            m = params['mult']
            shape = np.array([1, 1]) if shape is None else np.asarray(shape)

            inst_pins = user_pins_func(params)
            inst_native_elements = dict()
            for i in range(m):
                ofst = i * 100
                inst_native_elements['R0_' + str(i)] = laygo2.object.Rect(xy=[[ofst, 0], [ofst + 10, 10]],
                                                                          layer=['M1', 'drawing'])
                inst_native_elements['R1_' + str(i)] = laygo2.object.Rect(xy=[[ofst + 90, 90], [ofst + 100, 100]],
                                                                          layer=['M1', 'drawing'])
            inst_native_elements['R2'] = laygo2.object.Rect(xy=[[0, 0], [m * 100, 100]],
                                                            layer=['prBoundary', 'drawing'])
            inst = laygo2.object.VirtualInstance(name=name, libname='mylib', cellname='myvinst', xy=np.array([0, 0]),
                                                 native_elements=inst_native_elements, shape=shape,
                                                 pitch=pitch, unit_size=[m * 100, 100], pins=inst_pins,
                                                 transform=transform, params=params)
            return inst

        user_temp = UserDefinedTemplate(name='myusertemplate', bbox_func=user_bbox_func, pins_func=user_pins_func,
                                        generate_func=user_generate_func)
        user_inst = user_temp.generate(name='myuserinst', shape=[2, 1], params={'mult': 5})
        print(user_temp)
        print(user_inst)
        print(user_inst.bbox)
