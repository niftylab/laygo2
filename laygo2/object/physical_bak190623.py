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
This module implements classes for various layout objects.
"""

__author__ = "Jaeduk Han"
__maintainer__ = "Jaeduk Han"
__status__ = "Prototype"

import numpy as np
import laygo2.util.transform as tf


class Pointer: #LayoutObject):
    """
    Pointer class. Pointers are attached to other LayoutObjects to indicate various positional information
    associated with the object for placing and routing purposes.

    See Also
    --------
    LayoutObject
    """

    name = None
    """str or None: the name of the object."""

    _xy = np.zeros(2, dtype=np.int)
    """numpy.ndarray(dtype=numpy.int): the internal variable of xy."""

    def get_xy(self):
        """numpy.ndarray(dtype=int): gets the xy."""
        return self._xy

    def set_xy(self, value):
        """numpy.ndarray(dtype=int): sets the xy."""
        self._xy = np.asarray(value, dtype=np.int)

    xy = property(get_xy, set_xy)
    """numpy.ndarray(detype=numpy.int): the xy-coordinates of the object."""

    direction = None
    """str: the direction of the pointer."""
    master = None
    """LayoutObject.Instance: the master instance that the point tag is attached."""
    params = None
    """dict or None: the parameters of the object. """


    def __init__(self, name, xy, direction=None, master=None, params=None):
        """
        Constructor.

        Parameters
        ----------
        name : str
            The name of the object.
        xy : numpy.ndarray(dtype=int)
            The xy-coordinates of the object. Its default format is [x, y].
        direction : str
            The direction of the pointer, which will be used for place functions.
        master : LayoutObject.LayoutObject or its derived classes
            The master instance handle.
        """
        self.name = name
        self.xy = xy
        self.params = params
        self.direction = direction
        self.master = master

    def __str__(self):
        return self.summarize()

    def summarize(self):
        """Returns the summary of the object information."""
        return self.__repr__() + " " \
                                 "name: " + self.name + ", " + \
                                 "class: " + self.__class__.__name__ + ", " + \
                                 "xy: " + str(self.xy.tolist()) + ", " + \
                                 "params: " + str(self.params) + ", " + \
                                 "direction: " + str(self.direction) + ", " + \
                                 "master: /% " + str(self.master) + " %/"


class LayoutObject:
    """
    Basic layout object.
    """

    name = None
    """str or None: the name of the object."""

    _xy = np.zeros(2, dtype=np.int)
    """numpy.ndarray(dtype=numpy.int): the internal variable of xy."""

    def get_xy(self):
        """numpy.ndarray(dtype=int): gets the xy."""
        return self._xy

    def set_xy(self, value):
        """numpy.ndarray(dtype=int): sets the xy."""
        self._xy = np.asarray(value, dtype=np.int)
        self.update_pointers()

    xy = property(get_xy, set_xy)
    """numpy.ndarray(detype=numpy.int): the xy-coordinates of the object."""

    @property
    def bbox(self):
        """numpy.ndarray(dtype=int): the bounding box of the object. Its default format is
        [[x_ll, y_ll], [x_ur, y_ur]]"""
        return np.sort(np.array([self.xy, self.xy]), axis=0)

    params = None
    """dict or None: the parameters of the object. """

    # Pointers
    pointers = dict()

    def __init__(self, name, xy, params=None):
        """
        Constructor.

        Parameters
        ----------
        name : str
            The name of the object.
        xy : numpy.ndarray([int, int])
            The xy-coordinates of the object.
        params : dict or None
            Additional parameters of the object.
        """
        self.name = name

        # initialize pointers
        self.pointers['left'] = Pointer(name='left', xy=[0, 0], direction='left', master=self)
        self.pointers['right'] = Pointer(name='right', xy=[0, 0], direction='right', master=self)
        self.pointers['bottom'] = Pointer(name='bottom', xy=[0, 0], direction='bottom', master=self)
        self.pointers['top'] = Pointer(name='top', xy=[0, 0], direction='top', master=self)
        self.pointers['bottom_left'] = Pointer(name='bottom_left', xy=[0, 0], direction='bottom_left', master=self)
        self.pointers['bottom_right'] = Pointer(name='bottom_right', xy=[0, 0], direction='bottom_right', master=self)
        self.pointers['top_left'] = Pointer(name='top_left', xy=[0, 0], direction='top_left', master=self)
        self.pointers['top_right'] = Pointer(name='top_right', xy=[0, 0], direction='top_right', master=self)
        self.left = self.pointers['left']
        self.right = self.pointers['right']
        self.bottom = self.pointers['bottom']
        self.top = self.pointers['top']
        self.bottom_left = self.pointers['bottom_left']
        self.bottom_right = self.pointers['bottom_right']
        self.top_left = self.pointers['top_left']
        self.top_right = self.pointers['top_right']

        self.params = params
        self.xy = xy

    def __str__(self):
        return self.summarize()

    def summarize(self):
        """Returns the summary of the object information."""
        return self.__repr__() + " " \
                                 "name: " + self.name + ", " + \
                                 "class: " + self.__class__.__name__ + ", " + \
                                 "xy: " + str(self.xy.tolist()) + ", " + \
                                 "params: " + str(self.params)

    def update_pointers(self):
        xy_left = np.diag(np.dot(np.array([[1, 0], [0.5, 0.5]]), self.bbox)).astype(np.int)
        xy_right = np.diag(np.dot(np.array([[0, 1], [0.5, 0.5]]), self.bbox)).astype(np.int)
        xy_bottom = np.diag(np.dot(np.array([[0.5, 0.5], [1, 0]]), self.bbox)).astype(np.int)
        xy_top = np.diag(np.dot(np.array([[0.5, 0.5], [0, 1]]), self.bbox)).astype(np.int)
        xy_bottom_left = np.diag(np.dot(np.array([[1, 0], [1, 0]]), self.bbox)).astype(np.int)
        xy_bottom_right = np.diag(np.dot(np.array([[0, 1], [1, 0]]), self.bbox)).astype(np.int)
        xy_top_left = np.diag(np.dot(np.array([[1, 0], [0, 1]]), self.bbox)).astype(np.int)
        xy_top_right = np.diag(np.dot(np.array([[0, 1], [0, 1]]), self.bbox)).astype(np.int)
        self.pointers['left'].xy = xy_left
        self.pointers['right'].xy = xy_right
        self.pointers['bottom'].xy = xy_bottom
        self.pointers['top'].xy = xy_top
        self.pointers['bottom_left'].xy = xy_bottom_left
        self.pointers['bottom_right'].xy = xy_bottom_right
        self.pointers['top_left'].xy = xy_top_left
        self.pointers['top_right'].xy = xy_top_right



class LayoutIterableObject(LayoutObject):
    """
    Layout object that contains iterable elements. The iteration feature is implemented through a numpy ndarray object,
    called elements, by mapping iterator-related functions. Indexing sub-elements follows the numpy convention, which
    provides easy multi-dimensional indexing and advanced slicing.

    See Also
    --------
    LayoutObject
    """

    elements = None
    """numpy.array(dtype=LayoutObject): the iterable elements."""

    def __getitem__(self, pos):
        """Returns its sub-elements based on pos parameter."""
        return self.elements[pos]

    def __setitem__(self, key, item):
        self.elements[key] = item

    def __iter__(self):
        """Iterator function. Directly mapped to its elements."""
        return self.elements.__iter__()

    def __next__(self):
        """Iterator function. Directly mapped to its elements."""
        return self.elements.__next__()

    def ndenumerate(self):
        """Enumerates over the element array. Calls np.ndenumerate() of its elements."""
        return np.ndenumerate(self.elements)

    def __init__(self, name, xy=None, params=None, elements=None):
        """
        Constructor.

        Parameters
        ----------
        name : str
            The name of the object.
        xy : numpy.ndarray(dtype=int)
            The xy-coordinates of the layout object. Its default format is [x, y].
        params : dict or None
            The parameters of the object.
        elements : numpy.ndarray(dtype=LayoutObject) or None
            The iterable elements of the object.
        """

        self.elements = elements
        LayoutObject.__init__(self, name=name, xy=xy, params=params)


class LayoutObjectArray(np.ndarray):
    """LayoutObject array class for containing multiple layout objects. Subclassing ndarray to utilize advance slicing
     functions."""
    name = None
    """str: the name of the object."""

    params = None
    """dict or None: parameters of the object. """

    def __new__(cls, input_array, name=None, xy=None, params=None):
        """
        Constructor for ndarray subclasses - check the NumPy manual for details.

        Parameters
        ----------
        input_array : np.ndarray
            An array of LayoutObject objects.
        name : str
            The name of the array.
        xy : numpy.ndarray(dtype=int)
            The xy-coordinate of the object. The format is [x0, y0].
        params : dict
            Additional parameters of the array.
        """
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj.name = name
        obj.xy = xy
        obj.params = params
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        """
        Array finalizing function for subclassing ndarray - check the NumPy manual for details
        """
        if obj is None: return
        # Transfer parameters
        self.name = getattr(obj, 'name', None)
        self.xy = getattr(obj, 'xy', None)
        self.params = getattr(obj, 'params', None)

    def summarize(self):
        """Summarizes object information."""
        return "  " + \
               "name:" + self.name + ", " + \
               "class:" + self.__class__.__name__ + ", " + \
               "shape:" + str(self.shape) + ", " + \
               "xy:" + str(self.xy) + ", " + \
               "params:" + str(self.params) + "\n"


class Rect(LayoutObject):
    """
    Rect object.

    See Also
    --------
    LayoutObject
    """

    layer = None
    """list(str): Rect layer. The format is [name, purpose]."""
    netname = None
    """str: the name of the net associated with the rect."""

    def get_xy0(self):
        """Gets the xy0."""
        return self.xy[0]

    def set_xy0(self, value):
        """Sets the xy0."""
        self.xy[0] = np.asarray(value)
        self.update_pointers()

    xy0 = property(get_xy0, set_xy0)
    """numpy.array(dtype=int): the xy-coordinate of the primary (mostly lower-left) corner of the rect."""

    def get_xy1(self):
        """gets the xy1."""
        return self.xy[1]

    def set_xy1(self, value):
        """Sets the xy1."""
        self.xy[1] = np.asarray(value)
        self.update_pointers()

    xy1 = property(get_xy1, set_xy1)
    """numpy.array(dtype=int): the xy-coordinate of the secondary (mostly upper-right) corner of the rect."""

    @property
    def height(self):
        """int: the height of the rect"""
        return abs(self.xy0[1] - self.xy1[1])

    @property
    def width(self):
        """int: the width of the rect"""
        return abs(self.xy0[0] - self.xy1[0])

    @property
    def size(self):
        """numpy.ndarray(dtype=int): the size of the rect."""
        return np.array([self.width, self.height])

    @property
    def bbox(self):
        """numpy.ndarray(dtype=int): the bounding box of the object. Its default format is
        [[x_ll, y_ll], [x_ur, y_ur]]"""
        return np.sort(np.array([self.xy0, self.xy1]), axis=0)

    pointers = dict()
    """dict(): pointer dictionary."""
    # frequently used pointers
    left = None
    right = None
    top = None
    bottom = None
    center = None
    bottom_left = None
    bottom_right = None
    top_left = None
    top_right = None

    def __init__(self, name, xy, layer, netname=None, params=None):
        """
        Constructor.

        Parameters
        ----------
        name : str
            The name of the object
        xy : numpy.ndarray(dtype=int)
            The xy-coordinates of the object. Its format is [[x0, y0], [x1, y1]].
        layer : list(str)
            The layer information of the object. Its format is [layer, purpose]
        netname : str, optional
            The name of net associated with the object.
        params : dict
            Additional parameters of the object.
        """
        LayoutObject.__init__(self, name, xy, params=params)

        self.layer = layer
        self.netname = netname

    def summarize(self):
        """Returns the summary of the object information."""
        return LayoutObject.summarize(self) + ", " + \
                                              "layer: " + str(self.layer) + ", " + \
                                              "netname: " + str(self.netname)


class Wire(Rect):
    """
    Wire object.

    See Also
    --------
    LayoutObject
    """
    width = None
    """int: the width of the wire"""
    extension = 0
    """int: the amount of extension from the both edges of the wire."""

    def __init__(self, name, xy, layer, width, extension=0, netname=None, params=None):
        self.width = width
        self.extension = extension

        Rect.__init__(self, name, xy, layer, netname=netname, params=params)

    def summarize(self):
        """Returns the summary of the object information."""
        return LayoutObject.summarize(self) + ", " + \
               "width: " + str(self.width) + ", " + \
               "extension: " + str(self.extension) + ", " + \
               "layer: " + str(self.layer) + ", " + \
               "netname: " + str(self.netname)


class Pin(LayoutIterableObject):
    """
    Pin object.

    See Also
    --------
    LayoutIterableObject
    """

    layer = None
    """list(str): the layer of the pin. Its format is [name, purpose]."""
    netname = None
    """str: the name of the net associated with the pin."""
    master = None
    """Instance: the master instance of the pin. Used for instance pins only."""
    elements = None
    """numpy.array(dtype=Pin): the array contains its sub-element pins."""

    def get_xy0(self):
        """numpy.ndarray(dtype=int): gets the primary corner of the instance."""
        return self.xy[0]

    def set_xy0(self, value):
        """numpy.ndarray(dtype=int): sets the primary corner of the instance."""
        self.xy[0] = np.asarray(value)
        self.update_pointers()

    xy0 = property(get_xy0, set_xy0)
    """numpy.array(dtype=int): The xy-coordinate of the primary corner of the pin."""

    def get_xy1(self):
        """numpy.ndarray(dtype=int): gets the secondary corner of the instance."""
        return self.xy[1]

    def set_xy1(self, value):
        """numpy.ndarray(dtype=int): sets the primary corner of the instance."""
        self.xy[1] = np.asarray(value)
        self.update_pointers()

    xy1 = property(get_xy1, set_xy1)
    """numpy.array(dtype=int): The xy-coordinate of the secondary corner of the rect."""

    @property
    def height(self):
        """int: the height of the rect."""
        return abs(self.xy0[1] - self.xy1[1])

    @property
    def width(self):
        """int: the width of the rect."""
        return abs(self.xy0[0] - self.xy1[0])

    @property
    def size(self):
        """numpy.ndarray(dtype=int): the size of the rect."""
        return np.array([self.width, self.height])

    @property
    def bbox(self):
        """numpy.ndarray(dtype=int): the bounding box of the object. Its default format is
        [[x_ll, y_ll], [x_ur, y_ur]]"""
        return np.sort(np.array([self.xy0, self.xy1]), axis=0)

    def __init__(self, name, xy, layer, netname=None, master=None, params=None, elements=None):
        """
        Constructor.

        Parameters
        ----------
        name : str
            The name of the object.
        xy : numpy.ndarray(dtype=int)
            The xy-coordinates of the object. Its default format is [[x0, y0], [x1, y1]].
        layer : list(str)
            Layer information. Its default format is [layer, purpose].
        netname : str, optional
            The name of net associated with the object. If None, name is used for the net name.
        master : Instance, optional
            Master instance handle.
        params : dict, optional
            Additional parameters of the object.
        """
        self.layer = layer
        if netname is None:
            netname = name
        self.netname = netname
        self.elements = elements
        self.master = master

        LayoutIterableObject.__init__(self, name, xy, params=params, elements=elements)

    def summarize(self):
        """Returns the summary of the object information."""
        return LayoutIterableObject.summarize(self) + ", " + \
                                                      "layer: " + str(self.layer) + ", " + \
                                                      "netname: " + str(self.netname) + ", " + \
                                                      "master: " + str(self.master)


class Text(LayoutObject):
    """
    Text object.

    See Also
    --------
    LayoutObject
    """
    layer = None
    """list(str): the layer information of the text. Its default format is [name, purpose]."""
    text = None
    """str: the text body."""

    def __init__(self, name, xy, layer, text, params=None):
        """
        Constructor.

        Parameters
        ----------
        name : str
            The name of the object.
        xy : numpy.ndarray(dtype=int)
            The xy-coordinates of the object. Its default format is [x, y].
        layer : list(str)
            The layer information of the text. Its default format is [name, purpose].
        text : str
            The text entry.
        """
        self.layer = layer
        self.text = text

        LayoutObject.__init__(self, name, xy, params=params)

    def summarize(self):
        """Returns the summary of the object information."""
        return LayoutObject.summarize(self) + ", " + \
                                              "layer: " + str(self.layer) + ", " + \
                                              "text: " + str(self.text)


class Instance(LayoutIterableObject):
    """
    Instance object, corresponding to a single/mosaic layout instance.

    See Also
    --------
    LayoutIterableObject
    """
    shape = None
    """np.array([int, int]) or None: the shape of the instance mosaic. None if the instance is non-mosaic."""
    _pitch = None
    """np.array([int, int]) or None: the internal variable for pitch."""
    _unit_size = None
    """np.array([int, int]) or None: the internal variable for unit_size."""
    transform = 'R0'
    """str: the transform parameter of the instance."""
    template = None
    """TemplateObject.TemplateObject: the master template of the instance."""
    pins = None
    """dict(): pins of the instance."""
    pointers = dict()
    """dict(): pointers of the instance."""

    # frequently used pointers
    left = None
    right = None
    top = None
    bottom = None
    bottom_left = None
    bottom_right = None
    top_left = None
    top_right = None

    @property
    def xy0(self):
        """numpy.ndarray(dtype=[int, int]): the xy-coordinate of the object."""
        return self.xy

    @property
    def xy1(self):
        """numpy.ndarray(dtype=[int, int]): the secondary xy-coordinate of the object."""
        if self.size is None:
            return self.xy
        else:
            return self.xy + np.dot(self.size, tf.Mt(self.transform).T)

    def get_unit_size(self):
        """numpy.ndarray(dtype=[int, int]): gets the unit size of the object."""
        if self._unit_size is None:
            return self.template.size(params = self.params)
        else:
            return self._unit_size()

    def set_unit_size(self, value):
        """numpy.ndarray(dtype=[int, int]): sets the unit size of the object."""
        self._unit_size = value

    unit_size = property(get_unit_size, set_unit_size)
    """np.array([int, int]) or None: the unit size of the instance. The unit size is the size of the instance with 
    [1, 1] shape."""

    @property
    def size(self):
        """numpy.ndarray(dtype=int): the size of the instance. Its default format is [x_size, y_size]."""
        if self.shape is None:
            return self.unit_size
        else:
            return (self.shape - np.array([1, 1])) * self.pitch + self.unit_size

    def get_pitch(self):
        """numpy.ndarray(dtype=int): gets the pitch of the instance."""
        if self._pitch is None:
            return self.unit_size
        else:
            return self._pitch

    def set_pitch(self, value):
        """numpy.ndarray(dtype=int): sets the pitch of the instance."""
        self._pitch = value

    pitch = property(get_pitch, set_pitch)
    """numpy.ndarray(dtype=int): the pitch of the instance. Its default format is [x_pitch, y_pitch].
    None if template size is used for the instance pitch."""

    def get_spacing(self):
        return self.pitch

    def set_spacing(self, value):
        self.pitch = value

    spacing = property(get_spacing, set_spacing)
    """numpy.ndrarray([int, int]): (deprecated) the pitch of the instance. Previously the pitch was named to spacing,
     to be compatible with GDS-II's notations."""

    @property
    def bbox(self):
        """numpy.ndarray(dtype=int): the bounding box of the instance. Its default format is
        [[x_ll, y_ll], [x_ur, y_ur]]"""
        bbox = np.array([self.xy, self.xy + np.dot(self.size, tf.Mt(self.transform).T)])
        return np.sort(bbox, axis=0)

    def __init__(self, name, xy, template, shape=None, pitch=None, transform='R0', params=None):
        """
        Constructor.

        Parameters
        ----------
        name : str
            The name of the object.
        xy : numpy.ndarray(dtype=int)
            The xy-coordinate of the object. The format is [x0, y0].
        template: Template
            The template object handle.
        shape : numpy.ndarray(dtype=int) or None
            The size of the instance array. The format is [col, row].
        pitch : numpy.ndarray(dtype=int) or None
            The stride of the instance array. Its format is [x_pitch, y_pitch]. If none, the template size is used for
            the array pitch.
        transform : str
            The transform parameter. Possible values are 'R0', 'R90', 'R180', 'R270', 'MX', 'MY'.
        params : dict
            Additional parameters of the object.
        """
        # Preprocess parameters
        if isinstance(xy, Pointer):
            xy = xy.xy  # convert to xy coordinate.
        xy = np.asarray(xy)
        self.template = template
        if shape is not None:
            _shape = np.asarray(shape)
            if _shape.shape != (2, ):
                raise ValueError('Instance shape should be a (2, ) numpy array or None.')
            self.shape = _shape
        if pitch is not None:
            self.pitch = np.asarray(pitch)
        self.transform = transform

        # Construct an array for elements.
        if shape is None:
            elements = self  # np.array([])
        else:
            _shape = tuple(shape)
            elements = np.zeros(_shape, dtype=np.object)
            # elements = LayoutObjectArray(np.zeros(_shape, dtype=np.object))
            _it = np.nditer(elements, flags=['multi_index', 'refs_ok'])
            while not _it.finished:
                _idx = _it.multi_index
                _xy = xy + np.dot(self.pitch * np.array(_idx), tf.Mt(self.transform).T)
                inst = Instance(name=name, xy=_xy, template=self.template, shape=None, pitch=pitch,
                                transform=self.transform, params=params)
                elements[_idx] = inst
                _it.iternext()

        LayoutIterableObject.__init__(self, name, xy, params=params, elements=elements)

        # Create the pin dictionary
        self.pins = dict()
        pins = template.pins(params=self.params)
        if pins is not None:
            if not isinstance(pins, dict):
                raise ValueError("laygo2.Template.pins() should return a dictionary that contains pin information.")
            else:
                for pn, p in pins.items():
                    if shape is not None:
                        elements = []
                        for i in range(shape[0]):
                            elements.append([])
                            for j in range(shape[1]):
                                _xy = p['xy'] + np.dot(self.pitch * np.array([i, j]), tf.Mt(transform).T)
                                pin = Pin(name=pn, xy=_xy, netname=p['netname'], layer=p['layer'], master=self,
                                          elements=None)  # master uses self instead of self.elements[i, j].
                                elements[i].append(pin)
                        elements = np.array(elements)
                    else:
                        elements = None
                    self.pins[pn] = Pin(name=pn, xy=p['xy'], netname=p['netname'], layer=p['layer'], master=self,
                                        elements=elements)

    def summarize(self):
        """Summarizes object information."""
        return LayoutObject.summarize(self) + ", " + \
                                              "template: " + str(self.template) + ", " + \
                                              "size: " + str(self.size) + ", " + \
                                              "shape: " + str(self.shape) + ", " + \
                                              "pitch: " + str(self.pitch) + ", " + \
                                              "transform: " + str(self.transform) + ", " + \
                                              "pins: " + str(self.pins)


# Test
if __name__ == '__main__':
    print("Rect test")
    rect0 = Rect(name='R0', xy=[[0, 0], [100, 100]], layer=['M1', 'drawing'], netname='net0', params={'maxI': 0.005})
    print(rect0)
    print("Wire test")
    wire0 = Wire(name='R0', xy=[[0, 0], [0, 100]], width=10, extension=5, layer=['M1', 'drawing'], netname='net0', params={'maxI': 0.005})
    print(wire0)
    print("Pin test")
    pin0 = Pin(name='P0', xy=[[0, 0], [100, 100]], layer=['M1', 'drawing'], netname='net0', master=rect0,
               params={'direction':'input'})
    print(pin0)
    print("Text test")
    text0 = Text(name='T0', xy=[0, 0], layer=['text', 'drawing'], text='test', params=None)
    print(text0)
    print("Pointer test")
    pointer0 = Pointer(name='PT0', xy=[0, 0], direction='left', master=rect0)
    print(pointer0)
    print("Template test")
    from templates import Template
    template0_pins = dict()
    template0_pins['in'] = {'xy': [[0, 0], [10, 10]], 'netname': 'in', 'layer': ['M1', 'drawing']}
    template0_pins['out'] = {'xy': [[90, 90], [100, 100]], 'netname': 'out', 'layer': ['M1', 'drawing']}
    template0 = Template(name='my_template0', xy=[[0, 0], [100, 100]], pins=template0_pins)
    print(template0)
    print("Instance test")
    inst0 = Instance(name='I0', xy=[100, 100], template=template0, shape=[3, 2], pitch=[100, 100], transform='R0')
    print(inst0)
    print(inst0.shape)
    for idx, it in inst0.ndenumerate():
        print(idx, it.pins['in'])
    for idx, it in inst0.pins['in'].ndenumerate():
        print(idx, it)
