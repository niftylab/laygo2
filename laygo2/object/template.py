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
    An abstract class of templates. Templates are defined by inheriting this class and implementing the following
    core functions in their specific ways.
    xy(params): returns the x and y coordinates of the template for the parameters given by params.
    pins(params): returns a dictionary that contains Pin objects of the template for the input parameters.
    generate(params): returns a generated instance for the parameters given by params.

    Three representative templates are implemented in this module, which cover most usage cases:
    1. NativeInstanceTemplate: generates a single, non-parameterized instance.
    2. ParameterizedInstanceTemplate: generates a single, parameterized instance (p-cell)
    3. UserDefinedInstanceTemplate: generates a virtual instance (composed of multiple objects) from its custom generate(params) function.

    Or users can just inherit this Template class and implement abstract functions to build a new template.

    """

    name = None
    """str: The name of this template."""

    def __init__(self, name=None):
        """
        Constructor.

        Parameters
        ----------
        name : str or None, optional.
            The name of this template.
        """
        self.name = name

    def __str__(self):
        """Returns a string corresponding to this object's information."""
        return self.summarize()

    def summarize(self):
        """Returns the summary of the template information."""
        return self.__repr__() + " " \
                                 "name: " + self.name + ", " + \
                                 "class: " + self.__class__.__name__ + ", " + \
                                 ""

    def height(self, params=None):
        """int: Returns the height of the template."""
        return abs(self.bbox(params=params)[0, 1] - self.bbox(params=params)[1, 1])

    def width(self, params=None):
        """int: returns the width of the template."""
        return abs(self.bbox(params=params)[0, 0] - self.bbox(params=params)[1, 0])

    def size(self, params=None):
        """numpy.array(dtype=int): returns the size of the template."""
        return np.array([self.width(params=params), self.height(params=params)])

    @abstractmethod
    def bbox(self, params=None):
        """
        Computes the xy-coordinates of the bounding box of this template, corresponding to params.

        Parameters
        ----------
        params : dict() or None, optional.
            The dictionary that contains the parameters of the bounding box computed.

        Returns
        -------
        numpy.ndarray(dtype=int) : A 2x2 integer array that contains the bounding box coordinates.
        """
        pass

    @abstractmethod
    def pins(self, params=None):
        """
        Returns the dictionary that contains the pins of this template, corresponding to params.

        Parameters
        ----------
        params : dict() or None, optional.
            The dictionary that contains the parameters of the pins.

        Returns
        -------
        Dict[laygo2.object.physical.Pin] : A dictionary that contains pins of this template, with their names as keys.
        """
        pass

    @abstractmethod
    def generate(self, name=None, shape=None, pitch=np.array([0, 0]), transform='R0', params=None):
        """
        Generates an instance from this template.

        Parameters
        ----------
        name : str or None, optional.
            The name of the instance to be generated.
        shape : numpy.ndarray(dtype=int) or List[int] or None, optional.
            The shape of the instance to be generated.
        pitch : numpy.ndarray(dtype=int) or List[int], optional.
            The pitch between sub-elements of the generated instance.
        transform : str, optional.
            The transform parameter of the instance to be generated.
        params : dict() or None, optional.
            The dictionary that contains the parameters of the instance to be generated.

        Returns
        -------
        laygo2.object.physical.Instance or laygo2.object.physical.VirtualInstance: the generated instance.
        """
        pass


class NativeInstanceTemplate(Template):
    """A basic template object that generates a vanilla instance."""
    libname = None
    """str: The library name of the template."""

    cellname = None
    """str: The cellname of the template."""

    _bbox = np.array([[0, 0], [0, 0]])
    """numpy.array(dtype=int): A 2x2 numpy array that specifies the bounding box of the template."""

    _pins = None
    """Dict[laygo2.object.Pin] or None: A dictionary that contains pin information."""

    def __init__(self, libname, cellname, bbox=np.array([[0, 0], [0, 0]]), pins=None):
        """
        Constructor.

        Parameters
        ----------
        libname : str
            The library name of the template.
        cellname : str
            The cell name of the template.
        bbox : List[int] or numpy.ndarray(dtype=int)
            The xy-coordinates of the template or the function that returns the xy-coordinates of the template.
        pins : Dict[laygo2.object.Pin]
            A dict that contains the pin information of the template or a function that returns the dict.
            The pin dictionary.
        """
        self.libname = libname
        self.cellname = cellname
        self._bbox = None if bbox is None else np.asarray(bbox)
        self._pins = pins
        Template.__init__(self, name=cellname)

    def summarize(self):
        """Returns the summary of the template information."""
        return self.__repr__() + " " \
                                 "name: " + self.name + ", " + \
               "class: " + self.__class__.__name__ + ", " + \
               "bbox: " + str(self.bbox().tolist()) + ", " + \
               "pins: " + str(self.pins()) + ", " + \
               ""

    # Core template functions
    def bbox(self, params=None):
        """
        Computes the xy-coordinates of the bounding box of this template, corresponding to params.
        See laygo2.object.template.Template.bbox() for details.
        """
        return self._bbox

    def pins(self, params=None):
        """
        Returns the dictionary that contains the pins of this template, corresponding to params.
        See laygo2.object.template.Template.pins() for details.
        """
        return self._pins

    def generate(self, name=None, shape=None, pitch=None, transform='R0', params=None):
        """
        Creates an instance from this template. See laygo2.object.template.Template.generate() for details.

        Parameters
        ----------
        name : str or None, optional.
            The name of the instance to be generated.
        shape : numpy.ndarray(dtype=int) or List[int] or None, optional.
            The shape of the instance to be generated.
        pitch : numpy.ndarray(dtype=int) or List[int], optional.
            The pitch between sub-elements of the generated instance.
        transform : str, optional.
            The transform parameter of the instance to be generated.
        params : dict() or None, optional.
            The dictionary that contains the parameters of the instance to be generated.

        Returns
        -------
        laygo2.object.physical.Instance or laygo2.object.physical.VirtualInstance: the generated instance.
        """
        return laygo2.object.physical.Instance(libname=self.libname, cellname=self.cellname, xy=np.array([0, 0]),
                                               shape=shape, pitch=pitch, unit_size=self.size(params), pins=self.pins(params),
                                               transform=transform, name=name, params=params)

    # I/O functions
    def export_to_dict(self):
        db = dict()
        db['libname'] = self.libname
        db['cellname'] = self.cellname
        db['bbox'] = self.bbox().tolist()
        db['pins'] = dict()
        for pn, p in self.pins().items():
            db['pins'][pn] = p.export_to_dict()
        return db


class ParameterizedInstanceTemplate(Template):
    """A parameterized-instance-based template that helps users to define the templates without implementing
    the instantiation functions."""

    libname = None
    """str: The library name of the template."""

    cellname = None
    """str: The cellname of the template."""

    _bbox = None
    """callable(params=dict()): Returns the x and y coordinates of the template. Should be replaced with a user-defined 
    function."""

    _pins = None
    """callable(params=dict()): Returns a dictionary that contains the pin information. Should be replaced with a 
    user-defined function."""

    def __init__(self, libname, cellname, bbox_func=None, pins_func=None):
        """
        Constructor.

        Parameters
        ----------
        libname : str
            The library name of the template.
        cellname : str 
            The cell name of the template. 
        bbox_func : numpy.ndarray(dtype=int) or callable(params=dict())
            The function that returns the xy-coordinates for the template boundary.
        #xy_offset : numpy.ndarray(dtype=int) or callable(params=dict())
        #    The offset of the generated instance's x and y coordinates from the template's x and y coordinates.
        #    This is used to specify the difference between the boundary of the template and the actual instance's origin.
        pins_func : dict or callable(params=dict)
            The function that returns a dict that contains the pin information of the template.
        """
        self.libname = libname
        self.cellname = cellname
        self._bbox = bbox_func
        self._pins = pins_func
        Template.__init__(self, name=cellname)

    # Core template functions
    def bbox(self, params=None):
        """
        Computes the xy-coordinates of the bounding box of this template, corresponding to params.
        See laygo2.object.template.Template.bbox() for details.
        """
        return self._bbox(params=params)

    def pins(self, params=None):
        """
        Returns the dictionary that contains the pins of this template, corresponding to params.
        See laygo2.object.template.Template.pins() for details.
        """
        return self._pins(params=params)

    def generate(self, name=None, shape=None, pitch=np.array([0, 0]), transform='R0', params=None):
        """
        Creates an instance from this template. See laygo2.object.template.Template.generate() for details.
        """
        #xy = xy + np.dot(self.xy(params)[0], tf.Mt(transform).T)
        return laygo2.object.physical.Instance(libname=self.libname, cellname=self.cellname, xy=np.array([0, 0]),
                                               shape=shape, pitch=pitch, unit_size=self.size(params),
                                               pins=self.pins(params), transform=transform, name=name, params=params)


class UserDefinedTemplate(Template):
    """A virtual-instance-based template that produce subelements by calling user-defined functions.
    """

    _bbox = None
    """callable(params=dict()): Returns the bounding box of the template. Should be replaced with a user-defined 
    function."""

    _pins = None
    """callable(params=dict()): Returns a dictionary that contains the pin information. Should be replaced with a 
    user-defined function."""

    _generate = None
    """callable(name, shape, pitch, transform, params): 
    Returns a generated instance based on the input arguments.
    
    Should be mapped to a used-defined function that follows the definition format below:
    
    def generate_function_name(name=None, shape=None, pitch=np.array([0, 0]), transform='R0', params=None):
        body_of_generate_function
        return generated_instance
    """

    def __init__(self, bbox_func, pins_func, generate_func, name=None):
        """
        Constructor.

        Parameters
        ----------
        bbox_func: callable
            A function that computes the bounding box coordinates of the template.
        pins_func: callable
            A function that produces a dictionary that contains pin information of the template.
        generate_func: callable
            A fucntion that generates a (virtual) instance from the template.
        name : str
            The name of the template.
        """
        self._bbox = bbox_func
        self._pins = pins_func
        self._generate = generate_func
        Template.__init__(self, name=name)

    # Core template functions
    def bbox(self, params=None):
        """
        Computes the xy-coordinates of the bounding box of this template, corresponding to params.
        See laygo2.object.template.Template.bbox() for details.
        """
        return self._bbox(params=params)

    def pins(self, params=None):
        """
        Returns the dictionary that contains the pins of this template, corresponding to params.
        See laygo2.object.template.Template.pins() for details.
        """
        return self._pins(params=params)

    def generate(self, name=None, shape=None, pitch=np.array([0, 0]), transform='R0', params=None):
        """
        Creates an instance from this template. See laygo2.object.template.Template.generate() for details.
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
