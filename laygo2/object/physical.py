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
The physical module implements classes for various physical layout objects.
The types of objects supported are summarized below:

PhysicalObject - a base class for physical layout objects.

IterablePhysicalObject - a base class for iterable physical objects (eg. arrayed instances).

PhysicalObjectGroup - defines a group of physical objects.

Rect - defines a rect.

Path - defines a path.

Pin - defines a pin.

Text - defines a text label.

Instance - defines an instance.

VirtualInstance - defines a virtual instance composed of multiple physical objects.

"""

__author__ = "Jaeduk Han"
__maintainer__ = "Jaeduk Han"
__status__ = "Prototype"

import numpy as np
# from copy import deepcopy
import laygo2.util.transform as tf

class PhysicalObject:
    """
    The base class of physical layout objects.

    Attributes
    ----------
    name
    xy
    bbox
    master
    params
    pointers
    left
    right
    top
    bottom
    center
    bottom_left
    bottom_right
    top_left
    top_right
    """

    name = None
    """str or None: The name of this object."""

    _xy = np.zeros(2, dtype=np.int)
    """numpy.ndarray(dtype=numpy.int): The internal variable of xy."""

    def _get_xy(self):
        """numpy.ndarray(dtype=numpy.int): Gets the x and y coordinate values of this object."""
        return self._xy

    def _set_xy(self, value):
        """numpy.ndarray(dtype=numpy.int): Sets the x and y coordinate values of this object."""
        self._xy = np.asarray(value, dtype=np.int)
        self._update_pointers()

    xy = property(_get_xy, _set_xy)
    """numpy.ndarray(detype=numpy.int): The coordinate of this object represented as a Numpy array [x, y]."""

    @property
    def bbox(self):
        """numpy.ndarray(dtype=int): The bounding box for this object, represented as a Numpy array
        [[x_ll, y_ll], [x_ur, y_ur]]."""
        return np.sort(np.array([self.xy, self.xy]), axis=0)

    master = None
    """PhysicalObject or None: The master object that this object belongs to, if exists. None if there is no master."""

    params = None
    """dict or None: The dictionary that contains the parameters of this object, with parameter names as keys."""

    pointers = None
    """dict or None: The dictionary that contains the pointers of this object, with the pointers' names as keys."""

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

    def __init__(self, xy, name=None, params=None):
        """
        Constructor.

        Parameters
        ----------
        xy : numpy.ndarray(dtype=numpy.int)
            The coordinate of this object represented as a Numpy array [x, y].
        name : str, optional
            The name of this object.
        params : dict or None
            The dictionary that contains the parameters of this object, with parameter names as keys.
        """
        self.name = name

        # Initialize pointers.
        self.pointers = dict()
        self.pointers['left'] = np.array([0, 0], dtype=np.int)
        self.pointers['right'] = np.array([0, 0], dtype=np.int)
        self.pointers['bottom'] = np.array([0, 0], dtype=np.int)
        self.pointers['top'] = np.array([0, 0], dtype=np.int)
        self.pointers['bottom_left'] = np.array([0, 0], dtype=np.int)
        self.pointers['bottom_right'] = np.array([0, 0], dtype=np.int)
        self.pointers['top_left'] = np.array([0, 0], dtype=np.int)
        self.pointers['top_right'] = np.array([0, 0], dtype=np.int)
        self.left = self.pointers['left']
        self.right = self.pointers['right']
        self.bottom = self.pointers['bottom']
        self.top = self.pointers['top']
        self.bottom_left = self.pointers['bottom_left']
        self.bottom_right = self.pointers['bottom_right']
        self.top_left = self.pointers['top_left']
        self.top_right = self.pointers['top_right']

        self.params = params  # deepcopy(params)  # deepcopy increases the memory usage.
        self.xy = xy

    def __str__(self):
        """Returns the summary of this object."""
        return self.summarize()

    def summarize(self):
        """Returns the summary of this object."""
        name = 'None' if self.name is None else self.name
        return \
            self.__repr__() + " " \
            "name: " + name + ", " + \
            "class: " + self.__class__.__name__ + ", " + \
            "xy: " + str(self.xy.tolist()) + ", " + \
            "params: " + str(self.params) + ", " + \
            ""

    def _update_pointers(self):
        """Updates pointers of this object. Called when the xy-coordinate of this object is updated."""
        xy_left = np.diag(np.dot(np.array([[1, 0], [0.5, 0.5]]), self.bbox)).astype(np.int)
        xy_right = np.diag(np.dot(np.array([[0, 1], [0.5, 0.5]]), self.bbox)).astype(np.int)
        xy_bottom = np.diag(np.dot(np.array([[0.5, 0.5], [1, 0]]), self.bbox)).astype(np.int)
        xy_top = np.diag(np.dot(np.array([[0.5, 0.5], [0, 1]]), self.bbox)).astype(np.int)
        xy_bottom_left = np.diag(np.dot(np.array([[1, 0], [1, 0]]), self.bbox)).astype(np.int)
        xy_bottom_right = np.diag(np.dot(np.array([[0, 1], [1, 0]]), self.bbox)).astype(np.int)
        xy_top_left = np.diag(np.dot(np.array([[1, 0], [0, 1]]), self.bbox)).astype(np.int)
        xy_top_right = np.diag(np.dot(np.array([[0, 1], [0, 1]]), self.bbox)).astype(np.int)
        self.pointers['left'] = xy_left
        self.pointers['right'] = xy_right
        self.pointers['bottom'] = xy_bottom
        self.pointers['top'] = xy_top
        self.pointers['bottom_left'] = xy_bottom_left
        self.pointers['bottom_right'] = xy_bottom_right
        self.pointers['top_left'] = xy_top_left
        self.pointers['top_right'] = xy_top_right
        self.left = self.pointers['left']
        self.right = self.pointers['right']
        self.bottom = self.pointers['bottom']
        self.top = self.pointers['top']
        self.bottom_left = self.pointers['bottom_left']
        self.bottom_right = self.pointers['bottom_right']
        self.top_left = self.pointers['top_left']
        self.top_right = self.pointers['top_right']


class IterablePhysicalObject(PhysicalObject):
    """
    The base class of iterable physical objects. This object's iteration feature is supported by the element attribute,
    which is given by a Numpy array object that contains child objects of this object. The ways of iterating, indexing,
    and slicing the elements of this object follow the Numpy conventions, which provide convenient ways of
    multi-dimensional indexing and advanced slicing.

    Attributes
    ----------
    elements
    """

    elements = None
    """numpy.array(dtype=PhysicalObject-like): The elements of this object."""

    def __getitem__(self, pos):
        """Returns the sub-elements of this object, based on the pos parameter."""
        return self.elements[pos]

    def __setitem__(self, pos, item):
        """Sets the sub-elements of this object, based on the pos and item parameter. """
        self.elements[pos] = item

    def __iter__(self):
        """Iterator function. Directly mapped to the iterator of the elements attribute of this object."""
        return self.elements.__iter__()

    def __next__(self):
        """Iterator function. Directly mapped to the iterator of the elements attribute of this object."""
        return self.elements.__next__()

    def ndenumerate(self):
        """Enumerates over the element array. Calls np.ndenumerate() of the elements of this object."""
        return np.ndenumerate(self.elements)

    def _update_elements(self, xy_ofst):
        """Updates xy-coordinates of this object's elements. An internal function for _set_xy()"""
        if np.all(self.elements is not None):
            # Update the x and y coordinate values of elements.
            for n, e in self.ndenumerate():
                if e is not None:
                    e.xy = e.xy + xy_ofst

    def _get_xy(self):
        """numpy.ndarray(dtype=numpy.int): Gets the x and y coordinate values of this object."""
        return self._xy

    def _set_xy(self, value):
        """numpy.ndarray(dtype=numpy.int): Sets the x and y coordinate values of this object."""
        # Update the coordinate value of its elements.
        self._update_elements(xy_ofst=value - self.xy)
        # Update the coordinate value of the object itself.
        PhysicalObject._set_xy(self, value=value)

    xy = property(_get_xy, _set_xy)

    @property
    def shape(self):
        """numpy.ndarray(dtype=int): The shape of this object."""
        if self.elements is None:
            return None
        else:
            return np.array(self.elements.shape, dtype=np.int)

    def __init__(self, xy, name=None, params=None, elements=None):
        """
        Constructor.

        Parameters
        ----------
        xy : numpy.ndarray(dtype=int)
            The coordinate of this object represented as a Numpy array [x, y].
        name : str, optional
            The name of the object.
        params : dict or None
            The dictionary that contains the parameters of this object, with parameter names as keys.
        elements : numpy.ndarray(dtype=LayoutObject) or None
            The iterable elements of the object.
        """
        PhysicalObject.__init__(self, xy=xy, name=name, params=params)
        if elements is None:
            self.elements = None
        else:
            self.elements = np.asarray(elements)


class PhysicalObjectGroup(IterablePhysicalObject):
    """
    A grouped physical object. Intended to be generated as a group in Virtuoso (not implemented yet).
    """
    # TODO: implement this.

    def summarize(self):
        """Returns the summary of the object information."""
        return IterablePhysicalObject.summarize(self) + "\n" + \
               "  elements: " + str(self.elements)

    def __init__(self, xy, name=None, params=None, elements=None):
        """
        Constructor.

        Parameters
        ----------
        xy : numpy.ndarray(dtype=int)
            The coordinate of this object represented as a Numpy array [x, y].
        name : str, optional
            The name of the object.
        params : dict or None
            The dictionary that contains the parameters of this object, with parameter names as keys.
        elements : numpy.ndarray(dtype=LayoutObject) or None
            The iterable elements of the object.
        """
        IterablePhysicalObject.__init__(self, xy=xy, name=name, params=params, elements=elements)


'''
# Deprecated as PhysicalObjectGroup can be used instead in most cases.
class PhysicalObjectArray(np.ndarray):
    """LayoutObject array class for containing multiple layout objects. Subclassing ndarray to utilize advance slicing
     functions."""
    name = None
    """str: the name of the object."""

    params = None
    """dict or None: parameters of the object. """

    _xy = None  # np.zeros(2, dtype=np.int)
    """numpy.ndarray(dtype=numpy.int): the internal variable of xy."""

    def get_xy(self):
        """numpy.ndarray(dtype=numpy.int): gets the x and y coordinate values of this object."""
        return self._xy

    def set_xy(self, value):
        """numpy.ndarray(dtype=numpy.int): sets the x and y coordinate values of this object."""
        if value is None:
            self._xy = value
        else:
            self._xy = np.asarray(value, dtype=np.int)

    xy = property(get_xy, set_xy)
    """numpy.ndarray(detype=numpy.int): the x and y coordinate values of the object."""

    def moveby(self, delta):
        """move the array and child objects by delta."""
        self.xy = self.xy + delta
        for i in self:
            i.xy = i.xy + delta

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
        obj.xy = None if xy is None else np.asarray(xy, dtype=np.int)
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

    def __str__(self):
        """Returns the summary of this object."""
        return self.summarize()

    def summarize(self):
        """Summarizes object information."""
        return "  " + \
               "name:" + self.name + ", " + \
               "class:" + self.__class__.__name__ + ", " + \
               "shape:" + str(self.shape) + ", " + \
               "xy:" + str(self.xy) + ", " + \
               "params:" + str(self.params) + "\n" + \
               "  elements:" + str(np.ndarray.__str__(self)) + "\n"
'''


class Rect(PhysicalObject):
    """
    A rect object.

    Attributes
    ----------
    layer
    netname
    hextension
    vextension
    height
    width
    size
    bbox
    """

    layer = None
    """list(str): The layer information of this object."""
    netname = None
    """str: The name of the net associated with the rect."""
    hextension = 0
    """int or None: the extension of the Rect object in horizontal directions. Used to handle the extensions of routing
     elements."""
    vextension = 0
    """int or None: the extension of the Rect object in vertical directions. Used to handle the extensions of routing
     elements."""

    @property
    def height(self):
        """int: The height of the rect"""
        return abs(self.xy[0, 1] - self.xy[1, 1])

    @property
    def width(self):
        """int: The width of the rect"""
        return abs(self.xy[0, 0] - self.xy[1, 0])

    @property
    def size(self):
        """numpy.ndarray(dtype=int): The size of the rect."""
        return np.array([self.width, self.height])

    @property
    def bbox(self):
        """numpy.ndarray(dtype=int): The bounding box of the object. Its default format is
        [[x_ll, y_ll], [x_ur, y_ur]]"""
        return np.sort(np.array([self.xy[0, :], self.xy[1, :]]), axis=0)

    def __init__(self, xy, layer, hextension=0, vextension=0, name=None, netname=None, params=None):
        """
        Constructor.

        Parameters
        ----------
        xy : numpy.ndarray(dtype=int)
            The coordinates of this object represented as a Numpy array [[x0, y0], [x1, y1]].
        layer : list(str)
            The layer information of this object. Its format is [layer, purpose]
        name : str, optional
            The name of this object.
        netname : str, optional
            The name of net associated with this object.
        params : dict or None
            The dictionary that contains the parameters of this object, with parameter names as keys.
        """
        self.layer = layer
        if netname is None:
            self.netname = name
        else:
            self.netname = netname
        self.hextension = hextension
        self.vextension = vextension
        PhysicalObject.__init__(self, xy=xy, name=name, params=params)

    def summarize(self):
        """Returns the summary of the object information."""
        return PhysicalObject.summarize(self) + ", " + \
                                              "layer: " + str(self.layer) + ", " + \
                                              "netname: " + str(self.netname)


class Path(PhysicalObject):
    """
    A path object.

    Attributes
    ----------
    layer
    netname
    width
    extension
    """
    # TODO: implement pointers.

    layer = None
    """list(str): Path layer. The format is [name, purpose]."""
    netname = None
    """str: The name of the net associated with the rect."""
    width = None
    """int: The width of the wire"""
    extension = 0
    """int: The amount of extension from the both edges of the wire."""

    @property
    def bbox(self):
        """numpy.ndarray(dtype=int): The bounding box for this object, represented as a Numpy array
        [[x_ll, y_ll], [x_ur, y_ur]].
        For path objects, the bbox is computed from the first and last points of the path, which works fine for 2-point
        paths.
        """
        return np.sort(np.array([self.xy[0], self.xy[-1]]), axis=0)

    def _update_pointers(self):
        """Updates pointers of this object. Called when the value of xy of this object is updated."""
        pass

    def __init__(self, xy, layer, width, extension=0, name=None, netname=None, params=None):
        """
        Constructor.

        Parameters
        ----------
        xy : numpy.ndarray(dtype=int)
            The coordinates of this object represented as a Numpy array [[x0, y0], [x1, y1], ..., [xn, yn]].
        layer : list(str)
            The layer information of the object. Its format is [layer, purpose].
        width : int
            The width of the path.
        extension : int
            The extension of the path.
        name : str, optional
            The name of the object.
        netname : str, optional
            The name of net associated with the object.
        params : dict or None
            The dictionary that contains the parameters of this object, with parameter names as keys.
        """
        self.layer = layer
        self.width = width
        self.extension = extension
        self.netname = netname
        PhysicalObject.__init__(self, xy=xy, name=name, params=params)
        self.pointers = dict()  # Pointers are invalid for Path objects.

    def summarize(self):
        """Returns the summary of the object information."""
        return PhysicalObject.summarize(self) + ", " + \
               "width: " + str(self.width) + ", " + \
               "extension: " + str(self.extension) + ", " + \
               "layer: " + str(self.layer) + ", " + \
               "netname: " + str(self.netname)


class Pin(IterablePhysicalObject):
    """
    A pin object.

    Attributes
    ----------
    layer
    netname
    master
    """

    layer = None
    """list(str): The layer of the pin. Its format is [name, purpose]."""

    netname = None
    """str: The name of the net associated with the pin."""

    master = None
    """Instance: The master instance of the pin. Used for instance pins only."""

    @property
    def height(self):
        """int: The height of the rect."""
        return abs(self.xy[0, 1] - self.xy[1, 1])

    @property
    def width(self):
        """int: The width of the rect."""
        return abs(self.xy[0, 0] - self.xy[1, 0])

    @property
    def size(self):
        """numpy.ndarray(dtype=int): The size of the rect."""
        return np.array([self.width, self.height])

    @property
    def bbox(self):
        """numpy.ndarray(dtype=int): The bounding box of the object. Its default format is [[x0, y0], [x1, y1]], where
        [x0, y0] is the lower-left corner of this object, and [x1, y1] is the upper-right one."""
        return np.sort(np.array([self.xy[0, :], self.xy[1, :]]), axis=0)

    def __init__(self, xy, layer, name=None, netname=None, params=None, master=None, elements=None):
        """
        Constructor.

        Parameters
        ----------
        xy : numpy.ndarray(dtype=int)
            The xy-coordinates of the object. Its default format is [[x0, y0], [x1, y1]].
        layer : list(str)
            Layer information. Its default format is [layer, purpose].
        name : str
            The name of the object.
        netname : str, optional
            The name of net associated with the object. If None, name is used for the net name.
        master : Instance, optional
            Master instance handle.
        params : dict or None
            The dictionary that contains the parameters of this object, with parameter names as keys.
        """
        self.layer = layer
        if netname is None:
            netname = name
        self.netname = netname
        self.master = master

        IterablePhysicalObject.__init__(self, xy=xy, name=name, params=params, elements=elements)

    def summarize(self):
        """Returns the summary of the object information."""
        return IterablePhysicalObject.summarize(self) + ", " + \
                                                      "layer: " + str(self.layer) + ", " + \
                                                      "netname: " + str(self.netname) + ", " + \
                                                      "shape: " + str(self.shape) + ", " + \
                                                      "master: " + str(self.master)

    def export_to_dict(self):
        db = dict()
        db['xy'] = self.xy.tolist()
        db['layer'] = self.layer.tolist()
        db['name'] = self.name
        db['netname'] = self.netname
        return db


class Text(PhysicalObject):
    """
    A text object.

    See Also
    --------
    PhysicalObject
    """
    layer = None
    """list(str): The layer information of the text. Its default format is [name, purpose]."""
    text = None
    """str: The text body."""

    def __init__(self, xy, layer, text, name=None, params=None):
        """
        Constructor.

        Parameters
        ----------
        xy : numpy.ndarray(dtype=int)
            The xy-coordinates of the object. Its default format is [x, y].
        layer : list(str)
            The layer information of the text. Its default format is [name, purpose].
        text : str
            The text entry.
        name : str, optional
            The name of the object.
        params : dict or None
            The dictionary that contains the parameters of this object, with parameter names as keys.
        """
        self.layer = layer
        self.text = text

        PhysicalObject.__init__(self, xy=xy, name=name, params=params)

    def summarize(self):
        """Returns the summary of the object information."""
        return PhysicalObject.summarize(self) + ", " + \
                                              "layer: " + str(self.layer) + ", " + \
                                              "text: " + str(self.text)


class Instance(IterablePhysicalObject):
    """
    An iterable instance object, corresponding to a single/mosaic layout instance.
    """
    # TODO: update (maybe) xy and sub-elements after transform property is updated.

    libname = None
    """str: The library name of the instance."""

    cellname = None
    """str: The cell name of the instance."""

    shape = None
    """np.array([int, int]) or None: The shape of the instance mosaic. None if the instance is non-mosaic."""

    _pitch = None
    """np.array([int, int]) or None: The internal variable for pitch."""

    unit_size = None
    """np.array([int, int]) or None: The unit size(shape=[1,1]) of the instance. """

    transform = 'R0'
    """str: The transform parameter of the instance."""

    pins = None
    """Dict[Pins]: The pins of the instance."""

    def _update_pins(self, xy_ofst):
        """Updates xy-coordinates of this object's pins. An internal function for _set_xy()"""
        if self.pins is not None:
            for pn, p in self.pins.items():
                if np.all(p is not None):
                    # Update the x and y coordinate values of elements.
                    for n, e in np.ndenumerate(p):
                        if e is not None:
                            e.xy = e.xy + xy_ofst

    def _get_xy(self):
        """numpy.ndarray(dtype=numpy.int): Gets the x and y coordinate values of this object."""
        return self._xy

    def _set_xy(self, value):
        """numpy.ndarray(dtype=numpy.int): Sets the x and y coordinate values of this object."""
        # Update the coordinate value of its pins.
        self._update_pins(xy_ofst=value - self.xy)
        IterablePhysicalObject._set_xy(self, value=value)

    xy = property(_get_xy, _set_xy)

    @property
    def xy0(self):
        """numpy.ndarray(detype=numpy.int): The primary coordinate of this object represented as a Numpy array [x, y]."""
        return self.xy

    @property
    def xy1(self):
        """numpy.ndarray(detype=numpy.int): The secondary coordinate of this object represented as a Numpy array [x, y]."""
        if self.size is None:
            return self.xy
        else:
            return self.xy + np.dot(self.size, tf.Mt(self.transform).T)

    @property
    def size(self):
        """numpy.ndarray(dtype=int): The size of the instance, represented as a Numpy array [x_size, y_size]."""
        if self.shape is None:
            return self.unit_size
        else:
            return (self.shape - np.array([1, 1])) * self.pitch + self.unit_size

    def get_pitch(self):
        """numpy.ndarray(dtype=int): Gets the pitch of the instance."""
        if self._pitch is None:
            return self.unit_size
        else:
            return self._pitch

    def set_pitch(self, value):
        """numpy.ndarray(dtype=int): Sets the pitch of the instance."""
        self._pitch = value

    pitch = property(get_pitch, set_pitch)
    """numpy.ndarray(dtype=int): The pitch of the instance. Its default format is [x_pitch, y_pitch].
    None if template size is used for the instance pitch."""

    def get_spacing(self):
        return self.pitch

    def set_spacing(self, value):
        self.pitch = value

    spacing = property(get_spacing, set_spacing)
    """numpy.ndrarray([int, int]): (deprecated) The pitch of the instance. Previously the pitch was named to spacing,
     to be compatible with GDS-II's notations."""

    @property
    def bbox(self):
        """numpy.ndarray(dtype=int): The bounding box of the instance. Its default format is
        [[x_ll, y_ll], [x_ur, y_ur]]"""
        bbox = np.array([self.xy0, self.xy1])
        #return bbox
        #return self.xy + np.dot(self.size, tf.Mt(self.transform).T)
        return np.sort(bbox, axis=0)

    @property
    def height(self):
        """int: The height of the instance."""
        return abs(self.bbox[1][1] - self.bbox[0][1])

    @property
    def width(self):
        """int: The width of the instance."""
        return abs(self.bbox[1][0] - self.bbox[0][0])

    def __init__(self, xy, libname, cellname, shape=None, pitch=None, transform='R0', unit_size=np.array([0, 0]),
                 pins=None, name=None, params=None):
        """
        Constructor.

        Parameters
        ----------
        xy : numpy.ndarray(dtype=int)
            The coordinates of this object represented as a Numpy array [[x0, y0], [x1, y1]].
        libname : str
            The library name of the instance.
        cellname : str
            The cell name of th instance.
        shape : numpy.ndarray(dtype=int) or None
            The size of the instance array. The format is [col, row].
        pitch : numpy.ndarray(dtype=int) or None
            The stride of the instance array. Its format is [x_pitch, y_pitch]. If none, the template size is used for
            the array pitch.
        transform : str
            The transform parameter. Possible values are 'R0', 'R90', 'R180', 'R270', 'MX', 'MY'.
        unit_size : List[int] or numpy.ndarray(dtype=np.int)
            The size of the unit element of this instance.
        pins : Dict[Pin]
            The dictionary that contains the pin information.
        name : str
            The name of the object.
        params : dict
            The dictionary that contains the parameters of this object, with parameter names as keys.
        """
        # Assign parameters.
        xy = np.asarray(xy)
        self.libname = libname
        self.cellname = cellname
        if shape is not None:
            _shape = np.asarray(shape)
            if _shape.shape != (2, ):
                raise ValueError('Instance shape should be a (2, ) numpy array or None.')
            self.shape = _shape
        if pitch is not None:
            self.pitch = np.asarray(pitch)
        self.transform = transform
        self.unit_size = np.asarray(unit_size)

        # Construct an array for elements.
        if shape is None:
            # elements = self  # self-referencing causes recursion errors. Deprecated.
            elements = None
        else:
            _shape = tuple(shape)
            elements = np.zeros(_shape, dtype=np.object)
            # elements = LayoutObjectArray(np.zeros(_shape, dtype=np.object))
            _it = np.nditer(elements, flags=['multi_index', 'refs_ok'])
            while not _it.finished:
                _idx = _it.multi_index
                _xy = xy + np.dot(self.pitch * np.array(_idx), tf.Mt(self.transform).T)
                inst = Instance(xy=_xy, libname=libname, cellname=cellname, shape=None, pitch=pitch,
                                transform=self.transform, unit_size=self.unit_size, pins=pins, name=name, params=params)
                elements[_idx] = inst
                _it.iternext()

        IterablePhysicalObject.__init__(self, xy=xy, name=name, params=params, elements=elements)

        # Create the pin dictionary. Can we do the same thing without generating these many Pin objects?
        self.pins = dict()
        if pins is not None:
            if not isinstance(pins, dict):
                raise ValueError("The pins parameter for Instance objects should be a dictionary.")
            for pn, p in pins.items():
                _xy0 = xy + np.dot(p.xy, tf.Mt(transform).T)
                if shape is not None:
                    elements = []
                    for i in range(shape[0]):
                        elements.append([])
                        for j in range(shape[1]):
                            _xy = _xy0 + np.dot(self.pitch * np.array([i, j]), tf.Mt(transform).T)
                            # If p has elements, they need to be copied and transferred to the new pin.
                            _pelem = None
                            if p.elements is not None:
                                _pelem = np.empty(p.elements.shape, dtype=object)
                                for _idx, _pe in np.ndenumerate(p.elements):
                                    _pexy0 = xy + np.dot(_pe.xy, tf.Mt(transform).T) \
                                             + np.dot(self.pitch * np.array([i, j]), tf.Mt(transform).T)
                                    _pelem[_idx] = Pin(xy=_pexy0, netname=_pe.netname, layer=_pe.layer, name=_pe.name, master=self)
                            pin = Pin(xy=_xy, netname=p.netname, layer=p.layer, name=p.name, master=self,
                                      elements=_pelem)  # master uses self instead of self.elements[i, j].
                            elements[i].append(pin)
                    elements = np.array(elements)
                else:
                    # If p has elements, they need to be copied and transferred to the new pin.
                    _pelem = None
                    if p.elements is not None:
                        _pelem = np.empty(p.elements.shape, dtype=object)
                        for _idx, _pe in np.ndenumerate(p.elements):
                            _pexy0 = xy + np.dot(_pe.xy, tf.Mt(transform).T)
                            _pelem[_idx] = Pin(xy=_pexy0, netname=_pe.netname, layer=_pe.layer, name=_pe.name, master=self)
                    elements = _pelem
                self.pins[pn] = Pin(xy=_xy0, netname=p.netname, layer=p.layer, name=p.name, master=self,
                                    elements=elements)

    def summarize(self):
        """Summarizes object information."""
        return PhysicalObject.summarize(self) + ", " + \
               "size: " + str(self.size.tolist()) + ", " + \
               "shape: " + str(None if self.shape is None else self.shape.tolist()) + ", " + \
               "pitch: " + str(self.pitch.tolist()) + ", " + \
               "transform: " + str(self.transform) + ", " + \
               "pins: " + str(self.pins)


class VirtualInstance(Instance):  # IterablePhysicalObject):
    """
    A virtual instance object that is composed of multiple physical object and is considered as an instance.
    The VirtualInstance object is instantiated as a separate cellview (with its library and cell names specified)
    or a group with its native elements instantiated.
    """

    native_elements = None
    # Dict[PhysicalObject] the elements that compose the virtual instance. Its keys represent the names of the elements.

    def __init__(self, xy, libname, cellname, native_elements, shape=None, pitch=None, transform='R0', unit_size=np.array([0, 0]),
                 pins=None, name=None, params=None):
        """
        Constructor.

        Parameters
        ----------
        xy : numpy.ndarray(dtype=int)
            The value of the x and y coordinate values of this object.
            The xy-coordinate of the object. The format is [x0, y0].
        native_elements : Dict[PhysicalObject]
            A dictionary that contains elements that this object is composed of.
            Its keys represent the names of the elements.
        shape : numpy.ndarray(dtype=int) or None
            The size of the instance array. The format is [col, row].
        pitch : numpy.ndarray(dtype=int) or None
            The stride of the instance array. Its format is [x_pitch, y_pitch]. If none, the template size is used for
            the array pitch.
        transform : str
            The transform parameter. Possible values are 'R0', 'R90', 'R180', 'R270', 'MX', 'MY'.
        unit_size : List[int] or numpy.ndarray(dtype=np.int)
            The size of the unit element of this instance.
        pins : Dict[Pin]
            The dictionary that contains the pin information.
        name : str
            The name of the object.
        params : dict
            The dictionary that contains the parameters of this object, with parameter names as keys.
        """
        self.native_elements = native_elements

        Instance.__init__(self, xy=xy, libname=libname, cellname=cellname, shape=shape, pitch=pitch,
                          transform=transform, unit_size=unit_size, pins=pins, name=name, params=params)
        #Instance.__init__(self, xy=xy, libname='VirtualInstance', cellname='VirtualInstance', shape=shape, pitch=pitch,
        #                  transform=transform, unit_size=unit_size, pins=pins, name=name, params=params)

    def summarize(self):
        """Summarizes object information."""
        return Instance.summarize(self) + ", " + \
               "native elements: " + str(self.native_elements)


# Test
if __name__ == '__main__':
    test_rect = False
    test_path = False
    test_pin = False
    test_text = False
    test_pointer = False
    test_instance = True
    test_virtual_instance = False

    # You can create various objects by running part of the following commands.
    if test_rect:
        print("Rect test")
        rect0 = Rect(xy=[[0, 0], [100, 100]], layer=['M1', 'drawing'], netname='net0', params={'maxI': 0.005})
        print(rect0)
    if test_path:
        print("Path test")
        path0 = Path(xy=[[0, 0], [0, 100]], width=10, extension=5, layer=['M1', 'drawing'], netname='net0')
        print(path0)
    if test_pin:
        print("Pin test")
        pin0 = Pin(xy=[[0, 0], [100, 100]], layer=['M1', 'drawing'], netname='net0', master=rect0,
                   params={'direction': 'input'})
        print(pin0)
    if test_text:
        print("Text test")
        text0 = Text(xy=[0, 0], layer=['text', 'drawing'], text='test', params=None)
        print(text0)
    if test_instance:
        print("Instance test - creating a vanilla instance.")
        inst0_pins = dict()
        inst0_pins['in'] = Pin(xy=[[0, 0], [10, 10]], layer=['M1', 'drawing'], netname='in')
        inst0_pins['out'] = Pin(xy=[[90, 90], [100, 100]], layer=['M1', 'drawing'], netname='out')
        inst0 = Instance(name='I0', xy=[100, 100], libname='mylib', cellname='mycell', shape=[3, 2], pitch=[100, 100],
                         unit_size=[100, 100], pins=inst0_pins, transform='R0')
        print("  ", inst0)
        print("  ", inst0.pointers)
        for idx, it in inst0.ndenumerate():
            print("  ", idx, it)
            print("  ", idx, it.pins['in'])
        print("Instance test - updating the instance's coordinate values.")
        inst0.xy = [200, 200]
        print("  ", inst0)
        print("  ", inst0.pointers)
        for idx, it in inst0.ndenumerate():
            print("  ", idx, it)
            print("  ", idx, it.pins['in'])
    if test_virtual_instance:
        print("VirtualInstance test - creating a vanilla virtual instance.")
        inst1_pins = dict()
        inst1_pins['in'] = Pin(xy=[[0, 0], [10, 10]], layer=['M1', 'drawing'], netname='in')
        inst1_pins['out'] = Pin(xy=[[90, 90], [100, 100]], layer=['M1', 'drawing'], netname='out')
        inst1_native_elements = dict()
        inst1_native_elements['R0'] = Rect(xy=[[0, 0], [10, 10]], layer=['M1', 'drawing'])
        inst1_native_elements['R1'] = Rect(xy=[[90, 90], [100, 100]], layer=['M1', 'drawing'])
        inst1_native_elements['R2'] = Rect(xy=[[0, 0], [100, 100]], layer=['prBoundary', 'drawing'])
        inst1 = VirtualInstance(name='I0', libname='mylib', cellname='myvcell', xy=[500, 500],
                                native_elements=inst1_native_elements, shape=[3, 2], pitch=[100, 100],
                                unit_size=[100, 100], pins=inst1_pins, transform='R0')
        print("  ", inst1)
        for idx, it in inst1.ndenumerate():
            print("  ", idx, it.pins['in'])
        for idx, it in inst1.pins['in'].ndenumerate():
            print("  ", idx, it)
