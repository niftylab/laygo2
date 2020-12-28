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
This module implements classes for grid operations, for placement and routing of layout objects.
"""

__author__ = "Jaeduk Han"
__maintainer__ = "Jaeduk Han"
__status__ = "Prototype"

import numpy as np
import laygo2.object
#import laygo2.util.conversion as cv


# Internal functions.
def _extend_index_dim(input_index, new_index, new_index_max):
    """
    A helper function to be used for the multi-dimensional circular array indexing.
    It extends the dimension of the input array (input_index) that contains indexing information, with the additional
    indexing variable (new_index) provided. The new_index_max variable is specified in case of the new_index does not
    contain the maximum index information (perhaps when an open-end slice is given for the new_index).
    """
    # Construct an iterator from new_index
    if isinstance(new_index, (int, np.integer)):
        it = [new_index]
    else:
        if isinstance(new_index, slice):
            # slices don't work very well with multi-dimensional circular mappings.
            it = _conv_slice_to_list(slice_obj=new_index, stop_def=new_index_max)
        else:
            it = new_index
    # Index extension
    if input_index is None:
        output = []
        for i in it:
            output.append(tuple([i]))
        return output
    else:
        output = []
        for _i in input_index:
            output_row = []
            for i in it:
                output_row.append(tuple(list(_i) + [i]))
            output.append(output_row)
        return output


def _conv_slice_to_list(slice_obj, start_def=0, stop_def=100, step_def=1):
    """Convert slice to list"""
    if slice_obj.start is None:
        start = start_def
    else:
        start = slice_obj.start
    if slice_obj.stop is None:
        stop = stop_def
    else:
        stop = slice_obj.stop
    if slice_obj.step is None:
        step = step_def
    else:
        step = slice_obj.step
    return list(range(start, stop, step))


def _conv_bbox_to_array(bbox):
    """
    Convert a bbox object to a 2-d array.
    """
    array = list()
    for r in range(bbox[0, 1], bbox[1, 1] + 1):
        row = list()
        for c in range(bbox[0, 0], bbox[1, 0] + 1):
            row.append([c, r])
        array.append(row)
    return np.array(array)


def _conv_bbox_to_list(bbox):
    """
    Convert a bbox object to a 2-d array.
    """
    array = list()
    for r in range(bbox[0, 1], bbox[1, 1] + 1):
        for c in range(bbox[0, 0], bbox[1, 0] + 1):
            array.append([c, r])
    return array


# Internal classes
class CircularMapping:
    """
    A one-dimensional mapping class, circulating over the defined range.
    """
    _elements = None
    """numpy.ndarray: the internal variable of elements."""
    dtype = np.int
    """type_like: the type of elements."""

    def get_elements(self):
        """numpy.ndarray: gets the elements."""
        return self._elements

    def set_elements(self, value):
        """numpy.ndarray: sets the elements."""
        self._elements = np.asarray(value, dtype=self.dtype)

    elements = property(get_elements, set_elements)
    """numpy.ndarray: the array that contains the physical coordinates of the grid."""

    @property
    def shape(self):
        """numpy.ndarray: the shape of the mapping."""
        return np.array(self.elements.shape)

    def __init__(self, elements=np.array([0]), dtype=np.int):
        """
        Constructor.

        Parameters
        ----------
        elements : numpy.ndarray or list
            The elements of the circular mapping object.
        dtype : type_like
            The data type of the circular mapping object.
        """
        self.dtype = dtype
        self.elements = np.asarray(elements, dtype=dtype)

    # indexing and slicing
    def __getitem__(self, pos):
        """
        Returns elements corresponding to the indices given by pos, assuming the circular indexing.

        Parameters
        ----------
        pos : int or tuple, or list of int or tuple
            The index of elements to be returned.
        """
        if isinstance(pos, (int, np.integer)):
            return self.elements[pos % self.shape[0]]
        elif isinstance(pos, slice):
            return self.__getitem__(pos=_conv_slice_to_list(slice_obj=pos, stop_def=self.shape[0]))
        elif isinstance(pos, np.ndarray):
            return np.array([self.__getitem__(pos=p) for p in pos])
        elif isinstance(pos, list):
            return [self.__getitem__(pos=p) for p in pos]
        elif pos is None:
            return None
        else:
            raise TypeError("CircularMapping received an invalid index:%s" % str(pos))

    # Iterators
    def __iter__(self):
        """Iterator function. Directly mapped to the object's elements."""
        return self.elements.__iter__()

    def __next__(self):
        """Iterator function. Directly mapped to the object's elements."""
        # Check if numpy.ndarray implements __next__()
        return self.elements.__next__()

    # Informative functions
    def __str__(self):
        return self.summarize()

    def summarize(self):
        """Returns the summary of the object information."""
        return self.__repr__() + " " \
               "class: " + self.__class__.__name__ + ", " + \
               "elements: " + str(self.elements)


class CircularMappingArray(CircularMapping):
    """
    A multi-dimensional circular mapping. Split from the original circular mapping class to reduce complexity.
    """
    def __getitem__(self, pos):
        """
        Returns elements corresponding to the indices given by pos, assuming the circular indexing.

        Parameters
        ----------
        pos : int or tuple, or list of int or tuple
            The index of elements to be returned.
        """
        if isinstance(pos, list):  # pos is containing multiple indices as a list
            return [self.__getitem__(pos=p) for p in pos]
        elif pos is None:
            return None
        elif np.all(np.array([isinstance(p, (int, np.integer)) for p in pos])):
            # pos is mapped to a single element (pos is composed of integer indices).
            # just do rounding.
            idx = []
            for p, s in zip(pos, self.shape):
                idx.append(p % s)
            return self.elements[tuple(idx)]
        else:
            # pos is mapped to multiple indices. (possible examples include ([0:5, 3], [[1,2,3], 3], ...).
            # Create a list containing the indices to iterate over, and return a numpy.ndarray containing items
            # corresponding to the indices in the list.
            # When the indices don't specify the lower boundary (e.g., [:5]), it iterates from 0.
            # When the indices don't specify the upper boundary (e.g., [3:]), it iterates to the maximum index defined.
            idx = None
            for i, p in enumerate(pos):
                idx = _extend_index_dim(idx, p, self.shape[i])
            idx = np.asarray(idx)
            # iterate and generate the list to return
            item = np.empty(idx.shape[:-1], dtype=self.dtype)  # -1 because the tuples in idx are flatten.
            for i, _null in np.ndenumerate(item):
                item[i] = self.__getitem__(pos=tuple(idx[i]))
            return np.asarray(item)


class _AbsToPhyGridConverter:
    """
    An internal helper class that maps abstract coordinates to physical ones.
    """

    master = None
    """OneDimGrid or Grid: the master grid object that this converter belongs to."""

    # Constructor
    def __init__(self, master):
        """
        Constructor.

        Parameters
        ----------
        master: OneDimGrid or Grid
            The master grid object of the converter.
        """
        self.master = master

    # Access functions.
    def __call__(self, pos):
        """
        Returns the physical coordinate corresponding the abstract coordinate pos.

        Parameters
        ----------
        pos: np.ndarray(dtype=int)
            Abstract coordinate to be converted.

        Returns
        -------
        np.ndarray(dtype=int)
            Corresponding physical coordinate.
        """
        return self.__getitem__(pos)

    def __getitem__(self, pos):
        """
        Returns the physical coordinate corresponding the abstract coordinate pos.

        Parameters
        ----------
        pos: np.ndarray(dtype=int)
            Abstract coordinate to be converted.

        Returns
        -------
        np.ndarray(dtype=int)
            Corresponding physical coordinate.
        """
        if (self.master.__class__.__name__ == 'OneDimGrid') or (issubclass(self.master.__class__, OneDimGrid)):
            return self._getitem_1d(pos)
        if (self.master.__class__.__name__ == 'Grid') or (issubclass(self.master.__class__, Grid)):
            return self._getitem_2d(pos)
        else:
            return None

    def _getitem_1d(self, pos):
        """An internal function of __getitem__() for 1-d grids."""
        # Check if pos has multiple elements.
        if isinstance(pos, slice):
            return self._getitem_1d(_conv_slice_to_list(slice_obj=pos, stop_def=self.master.shape[0]))
        elif isinstance(pos, np.ndarray):
            return self._getitem_1d(pos.tolist())
        elif isinstance(pos, list):
            return np.array([self._getitem_1d(p) for p in pos])
        elif pos is None:
            raise TypeError("_AbsToPhyConverter._getitem_1d does not accept None as its input.")
        else:
            # pos is a single element. Compute quotient and modulo for grid extension.
            quo = 0
            mod = int(round(pos))
            if pos >= self.master.shape[0]:
                mod = int(round(pos % self.master.shape[0]))
                quo = int(round((pos-mod) / self.master.shape[0]))
            elif pos < 0:
                mod = int(round(pos % self.master.shape[0]))
                quo = int(round((pos-mod)) / self.master.shape[0])
            return quo * self.master.range[1] + self.master.elements[mod]
            # the following command cannot handle the size extension of the grid, disabled.
            # return self.master.elements.take(pos, mode='wrap')

    def _getitem_2d(self, pos):
        """An internal function of __getitem__() for 2-d grids."""
        if isinstance(pos, list):
            if isinstance(pos[0], (int, np.integer)):  # single point
                return self[pos[0], pos[1]]
            else:
                return [self[p] for p in pos]
        elif isinstance(pos, np.ndarray):
            if isinstance(pos[0], (int, np.integer)):  # single point
                return np.array(self[pos[0], pos[1]])
            else:
                return np.array([self[p] for p in pos])
        # compute coordinates from OneDimGrids of its master.
        x = self.master.x[pos[0]]
        y = self.master.y[pos[1]]
        # TODO: Refactor the following code to avoid the use of double for loops and list comprehensions.
        if (not isinstance(x, np.ndarray)) and (not isinstance(y, np.ndarray)):  # x and y are scalars.
            return np.array([x, y])
        if not isinstance(x, np.ndarray):  # x is a scalar.
            return np.array([np.array([x, _y]) for _y in y])
        elif not isinstance(y, np.ndarray):  # y is a scalar.
            return np.array([np.array([_x, y]) for _x in x])
        else:
            xy = []
            for _x in x:  # vectorize this operation.
                row = []
                for _y in y:
                    row.append(np.array([_x, _y]))
                xy.append(np.array(row))
        return np.array(xy)

    # Reverse-access operators (comparison operators are used for reverse-access).
    def __eq__(self, other):
        """Returns the absolute coordinate corresponding to other (Inverse-mapping of __getitem__(pos))."""
        return self.master.phy2abs(pos=other)
    '''
        if (self.master.__class__.__name__ == 'OneDimGrid') or (issubclass(self.master.__class__, OneDimGrid)):
            return self._eq_1d(other=other)
        if (self.master.__class__.__name__ == 'Grid') or (issubclass(self.master.__class__, Grid)):
            return self._eq_2d(other=other)

    def _eq_1d(self, other):
        return self._getitem_1d(pos=other)

    def _eq_2d(self, other):
        return self.__getitem__(pos=other)
        if isinstance(other[0], (int, np.integer)):
            return np.array([self.master.x[other[0]],
                             self.master.y[other[1]])
        else:
            return np.array([self._eq_2d(o) for o in other])
    '''

    def __lt__(self, other):
        """Returns thi abstract coordinate corresponding to the physical coordinate that is the largest but less than
        other."""
        if (self.master.__class__.__name__ == 'OneDimGrid') or (issubclass(self.master.__class__, OneDimGrid)):
            return self._lt_1d(other)
        if (self.master.__class__.__name__ == 'Grid') or (issubclass(self.master.__class__, Grid)):
            return self._lt_2d(other)
        else:
            return None

    def _lt_1d(self, other):
        if isinstance(other, (int, np.integer)):
            quo = None
            mod = None
            for i, e in np.ndenumerate(self.master.elements):
                _quo = int(round(np.floor((other - e) / self.master.width)))
                if not (_quo * self.master.width + e == other):
                    if quo is None:
                        quo = _quo
                        mod = i[0]
                    elif _quo >= quo:  # update quo
                        quo = _quo
                        mod = i[0]
            return quo * self.master.elements.shape[0] + mod
        else:
            return np.array([self._lt_1d(o) for o in other])

    def _lt_2d(self, other):
        if isinstance(other[0], (int, np.integer)):
            return np.array([self.master.x < other[0],
                             self.master.y < other[1]])
        else:
            return np.array([self._lt_2d(o) for o in other])

    def __le__(self, other):
        """Returns the abstract coordinate that is the largest but less than or equal to other."""
        if (self.master.__class__.__name__ == 'OneDimGrid') or (issubclass(self.master.__class__, OneDimGrid)):
            return self._le_1d(other=other)
        if (self.master.__class__.__name__ == 'Grid') or (issubclass(self.master.__class__, Grid)):
            return self._le_2d(other=other)

    def _le_1d(self, other):
        if isinstance(other, (int, np.integer)):
            quo = None
            mod = None
            for i, e in np.ndenumerate(self.master.elements):
                _quo = int(round(np.floor((other - e) / self.master.width)))
                if quo is None:
                    quo = _quo
                    mod = i[0]
                elif _quo >= quo:  # update quo
                    quo = _quo
                    mod = i[0]
            return quo * self.master.elements.shape[0] + mod
        else:
            return np.array([self._le_1d(o) for o in other])

    def _le_2d(self, other):
        if isinstance(other[0], (int, np.integer)):
            return np.array([self.master.x <= other[0],
                             self.master.y <= other[1]])
        else:
            return np.array([self._le_2d(o) for o in other])

    def __gt__(self, other):
        """Returns the abstract coordinate that is the smallest but greater than other."""
        if (self.master.__class__.__name__ == 'OneDimGrid') or (issubclass(self.master.__class__, OneDimGrid)):
            return self._gt_1d(other=other)
        if (self.master.__class__.__name__ == 'Grid') or (issubclass(self.master.__class__, Grid)):
            return self._gt_2d(other=other)

    def _gt_1d(self, other):
        if isinstance(other, (int, np.integer)):
            quo = None
            mod = None
            for i, e in np.ndenumerate(self.master.elements):
                _quo = int(round(np.ceil((other - e) / self.master.width)))
                if not (_quo * self.master.width + e == other):
                    if quo is None:
                        quo = _quo
                        mod = i[0]
                    elif _quo < quo:  # update quo
                        quo = _quo
                        mod = i[0]
            return quo * self.master.elements.shape[0] + mod
        else:
            return np.array([self.__gt__(o) for o in other])

    def _gt_2d(self, other):
        if isinstance(other[0], (int, np.integer)):
            return np.array([self.master.m > other[0],
                             self.master.n > other[1]])
        else:
            return np.array([self._gt_2d(o) for o in other])

    def __ge__(self, other):
        """Returns the abstract coordinate that is the smallest but greater than or equal to other."""
        if (self.master.__class__.__name__ == 'OneDimGrid') or (issubclass(self.master.__class__, OneDimGrid)):
            return self._ge_1d(other=other)
        if (self.master.__class__.__name__ == 'Grid') or (issubclass(self.master.__class__, Grid)):
            return self._ge_2d(other=other)

    def _ge_1d(self, other):
        if isinstance(other, (int, np.integer)):
            quo = None
            mod = None
            for i, e in np.ndenumerate(self.master.elements):
                _quo = int(round(np.ceil((other - e) / self.master.width)))
                if quo is None:
                    quo = _quo
                    mod = i[0]
                elif _quo < quo:  # update quo
                    quo = _quo
                    mod = i[0]
            return quo * self.master.elements.shape[0] + mod
        else:
            return np.array([self.__ge__(o) for o in other])

    def _ge_2d(self, other):
        if isinstance(other[0], (int, np.integer)):
            return np.array([self.master.x >= other[0],
                             self.master.y >= other[1]])
        else:
            return np.array([self._ge_2d(o) for o in other])


class _PhyToAbsGridConverter:
    """
    An internal helper class that maps physical coordinates to abstract ones.
    """

    master = None
    """OneDimGrid or Grid: the master grid object that this converter belongs to."""

    # Constructor
    def __init__(self, master):
        """
        Constructor.

        Parameters
        ----------
        master: OneDimGrid or Grid
            The master grid object of the converter.
        """
        self.master = master

    # Access functions.
    def __call__(self, pos):
        """
        Returns the abstract coordinate corresponding the physical coordinate pos.

        Parameters
        ----------
        pos: np.ndarray(dtype=int)
            Physical coordinate to be converted.

        Returns
        -------
        np.ndarray(dtype=int)
            Corresponding abstract coordinate.
        """
        return self.__getitem__(pos)

    def __getitem__(self, pos):
        """Returns the abstract coordinate corresponding to the physical grid pos.

        Parameters
        ----------
        pos: np.ndarray(dtype=int)
            Physical coordinate to be converted.

        Returns
        -------
        np.ndarray(dtype=int)
            Corresponding abstract coordinate.
        """
        if (self.master.__class__.__name__ == 'OneDimGrid') or (issubclass(self.master.__class__, OneDimGrid)):
            return self._getitem_1d(pos)
        if (self.master.__class__.__name__ == 'Grid') or (issubclass(self.master.__class__, Grid)):
            return self._getitem_2d(pos)
        else:
            return None

    def _getitem_1d(self, pos):
        """An internal function of __getitem__() for 1-d grids."""
        # Check if pos has multiple elements.
        if isinstance(pos, OneDimGrid):
            return self._getitem_1d(pos=pos.elements)
        elif isinstance(pos, slice):
            return self._getitem_1d(_conv_slice_to_list(slice_obj=pos, stop_def=self.master.shape[0]))
        elif isinstance(pos, np.ndarray):
            return self._getitem_1d(pos.tolist())
        elif isinstance(pos, list):
            return np.array([self._getitem_1d(p) for p in pos])
        elif pos is None:
            raise TypeError("_AbsToPhyConverter._getitem_1d does not accept None as its input.")
        else:
            # pos is a single element.
            for i, e in np.ndenumerate(self.master.elements):
                if (pos - e) % self.master.width == 0:
                    return int(round((pos - e) / self.master.width)) * self.master.elements.shape[0] + i[0]
            return None  # no matched coordinate

    def _getitem_2d(self, pos):
        """An internal function of __getitem__() for 2-d grid."""
        # If pos contains multiple coordinates (or objects), convert recursively.
        if isinstance(pos, list):
            if isinstance(pos[0], (int, np.integer)):  # It's actually a single coordinate.
                return self[pos[0], pos[1]]
            else:
                return [self[p] for p in pos]
        elif isinstance(pos, np.ndarray):
            if isinstance(pos[0], (int, np.integer)):  # It's actually a single coordinate.
                return np.array(self[pos[0], pos[1]])
            else:
                return np.array([self[p] for p in pos])
        # If pos contains only one physical object, convert its bounding box to abstract coordinates
        if (pos.__class__.__name__ == 'PhysicalObject') or (issubclass(pos.__class__, laygo2.object.PhysicalObject)):
            return self.bbox(pos)
        # If pos contains only one coordinate, convert it to abstract grid.
        m = self.master.x == pos[0]
        n = self.master.y == pos[1]
        # refactor the following code to avoid the use of double for-loops and list comprehensions.
        if (not isinstance(m, np.ndarray)) and (not isinstance(n, np.ndarray)):  # x and y are scalars.
            return np.array([m, n])
        if not isinstance(m, np.ndarray):  # x is a scalar.
            return np.array([np.array([m, _n]) for _n in n])
        elif not isinstance(n, np.ndarray):  # y is a scalar.
            return np.array([np.array([_m, n]) for _m in m])
        else:
            mn = []
            for _m in m:  # vectorize this operation.
                row = []
                for _n in n:
                    row.append(np.array([_m, _n]))
                mn.append(np.array(row))
        return np.array(mn)

    # Reverse-access operators (comparison operators are used for reverse-access).
    def __eq__(self, other):
        """Returns the physical coordinate corresponding to the abstract coordinate other."""
        self.master.abs2phy(pos=other)
    '''
        if (self.master.__class__.__name__ == 'OneDimGrid') or (issubclass(self.master.__class__, OneDimGrid)):
            return self._eq_1d(other=other)
        if (self.master.__class__.__name__ == 'Grid') or (issubclass(self.master.__class__, Grid)):
            return self._eq_2d(other=other)

    def _eq_1d(self, other):
        return self._getitem_1d(pos=other)

    def _eq_2d(self, other):
        # If other is a physical object, convert its bounding box to abstract coordinates.
        if (other.__class__.__name__ == 'PhysicalObject') or (issubclass(other.__class__, laygo2.object.PhysicalObject)):
            mn0 = self.master >= other.bbox[0]
            mn1 = self.master <= other.bbox[1]
            return np.array([mn0, mn1])
        if isinstance(other[0], (int, np.integer)):
            return np.array([self.master.m[other[0]],
                             self.master.n[other[1]]])
        else:
            return np.array([self._eq_2d(o) for o in other])
    '''
    def __lt__(self, other):
        """Returns the physical coordinate corresponding to the abstract coordinate that is the largest but less than
        other."""
        if (self.master.__class__.__name__ == 'OneDimGrid') or (issubclass(self.master.__class__, OneDimGrid)):
            return self._lt_1d(other=other)
        if (self.master.__class__.__name__ == 'Grid') or (issubclass(self.master.__class__, Grid)):
            return self._lt_2d(other=other)

    def _lt_1d(self, other):
        if isinstance(other, (int, np.integer)):
            return self.master.abs2phy.__getitem__(pos=other-1)
        return np.array([self._lt_1d(o) for o in other])

    def _lt_2d(self, other):
        if isinstance(other[0], (int, np.integer)):
            return self.master.abs2phy.__getitem__(pos=(other[0]-1, other[1]-1))
        return np.array([self._lt_2d(o) for o in other])

    def __le__(self, other):
        """Returns the physical coordinate corresponding to the abstract coordinate that is the largest but less than
        or equal to other.
        Should be equivalent to __eq__.
        """
        return self.master.abs2phy(pos=other)

    def __gt__(self, other):
        """
        Returns the physical grid coordinate whose abstract coordinate is the smallest but greater than other.
        """
        if (self.master.__class__.__name__ == 'OneDimGrid') or (issubclass(self.master.__class__, OneDimGrid)):
            return self._gt_1d(other)
        if (self.master.__class__.__name__ == 'Grid') or (issubclass(self.master.__class__, Grid)):
            return self._gt_2d(other)
        else:
            return None

    def _gt_1d(self, other):
        if isinstance(other, (int, np.integer)):
            return self.master.abs2phy.__getitem__(pos=other+1)
        return np.array([self._gt_1d(o) for o in other])

    def _gt_2d(self, other):
        if isinstance(other[0], (int, np.integer)):
            return self.master.abs2phy.__getitem__(pos=(other[0]+1, other[1]+1))
        return np.array([self._gt_2d(o) for o in other])

    def __ge__(self, other):
        """
        Returns the physical grid coordinate whose abstract coordinate is the smallest but greater than or equal to
        other. Should be equivalent to __eq__.
        """
        return self.master.abs2phy.__getitem__(pos=other)

    def bbox(self, obj):
        """Returns the abstract grid coordinates corresponding to the 'internal' bounding box of obj.
        Strictly speaking, the resulting box may not be a bounding box (as the box is located inside obj if obj.bbox is
        not on grid), but the internal bounding box is more useful than the actual bounding box especially for routing
        and via purposes.

        Parameters
        ----------
        obj: numpy.ndarray or PhysicalObject
            A numpy array representing a bounding box in physical coordinate, or a PhysicalObject.
        """
        if (obj.__class__.__name__ == 'PhysicalObject') or (issubclass(obj.__class__, laygo2.object.PhysicalObject)):
            return self.bbox(obj.bbox)
        mn0 = self.master.xy >= obj[0]
        mn1 = self.master.xy <= obj[1]
        return np.array([mn0, mn1])

    def bottom_left(self, obj):
        """Returns the bottom-left corner of an object on this grid."""
        if (obj.__class__.__name__ == 'PhysicalObject') or (issubclass(obj.__class__, laygo2.object.PhysicalObject)):
            return self.bottom_left(obj.bbox)
        else:
            _i = self.bbox(obj)
            return _i[0]

    def bottom_right(self, obj):
        """Returns the bottom-right corner of an object on this grid."""
        if (obj.__class__.__name__ == 'PhysicalObject') or (issubclass(obj.__class__, laygo2.object.PhysicalObject)):
            return self.bottom_right(obj.bbox)
        else:
            _i = self.bbox(obj)
            return np.array([_i[1, 0], _i[0, 1]])

    def top_left(self, obj):
        """Returns the top-left corner of an object on this grid."""
        if (obj.__class__.__name__ == 'PhysicalObject') or (issubclass(obj.__class__, laygo2.object.PhysicalObject)):
            return self.top_left(obj.bbox)
        else:
            _i = self.bbox(obj)
            return np.array([_i[0, 0], _i[1, 1]])

    def top_right(self, obj):
        """Returns the top-right corner of an object on this grid."""
        if (obj.__class__.__name__ == 'PhysicalObject') or (issubclass(obj.__class__, laygo2.object.PhysicalObject)):
            return self.top_right(obj.bbox)
        else:
            _i = self.bbox(obj)
            return _i[1]

    def width(self, obj):
        """Returns the width of an object on this grid."""
        if (obj.__class__.__name__ == 'PhysicalObject') or (issubclass(obj.__class__, laygo2.object.PhysicalObject)):
            return self.width(obj.bbox)
        else:
            _i = self.bbox(obj)
            return abs(_i[1, 0] - _i[0, 0])

    def height(self, obj):
        """Returns the height of an object on this grid."""
        if (obj.__class__.__name__ == 'PhysicalObject') or (issubclass(obj.__class__, laygo2.object.PhysicalObject)):
            return self.height(obj.bbox)
        else:
            _i = self.bbox(obj)
            return abs(_i[1, 1] - _i[0, 1])

    def height_vec(self, obj):
        """numpy.ndarray(dtype=int): Returns np.array([0, height])."""
        return np.array([0, self.height(obj)])

    def width_vec(self, obj):
        """numpy.ndarray(dtype=int): Returns np.array([width, 0])."""
        return np.array([self.width(obj), 0])

    def size(self, obj):
        """Returns the size of an object on this grid."""
        return np.array([self.width(obj), self.height(obj)])

    def crossing(self, *args):
        """
        Returns a point on this grid, corresponding to the cross-point of bounding boxes given by args.
        This function assumes there's an overlap between input bounding boxes with any exception handlings.
        """
        return self.overlap(*args, type='point')

    def overlap(self, *args, type='bbox'):
        """
        Returns a bounding box on this grid, corresponding to the intersection of bounding boxes given by args.
        This function assumes there's an overlap between input bounding boxes with any exception handlings.

        Parameters
        ----------
        *args: np.ndarray or PhysicalObject
            A collection of array or physical objects where the overlap region will be computed over.
        type: str {'bbox', 'point', 'array'}
            The type of overlap region.
            If 'bbox', a 2x2 numpy array containing lower-left and upper-right corners of the overlap region is returned.
            If 'point', the lower-left corner of the overlap region is returened.
            If 'array' a 2-dimension array containing all points in the overlap region is returned.
        """
        _ib = None
        for _obj in args:
            if _ib is None:
                _ib = self.bbox(_obj)
            else:
                _b = self.bbox(_obj)
                _x = np.sort(np.array([_b[:, 0], _ib[:, 0]]), axis=None)
                _y = np.sort(np.array([_b[:, 1], _ib[:, 1]]), axis=None)
                _ib = np.array([[_x[1], _y[1]], [_x[2], _y[2]]])
        if type == 'bbox':
            return _ib
        elif type == 'point':
            return _ib[0]
        elif type == 'list':
            return _conv_bbox_to_list(_ib)
        elif type == 'array':
            return _conv_bbox_to_array(_ib)
        else:
            raise ValueError('overlap() should receive a valid value for its type (bbox, point, array, ...)')

    def union(self, *args):
        """Returns a bounding box on this grid, corresponding to the union of two bounding boxes given by args."""
        _ub = None
        for _obj in args:
            if _ub is None:
                _ub = self.bbox(_obj)
            else:
                _b = self.bbox(_obj)
                _x = np.sort(np.array([_b[:, 0], _ub[:, 0]]), axis=None)
                _y = np.sort(np.array([_b[:, 1], _ub[:, 1]]), axis=None)
                _ub = np.array([[_x[0], _y[0]], [_x[3], _y[3]]])
        return _ub


class OneDimGrid(CircularMapping):
    """
    Basic one-dimensional layout grid.
    """

    # Member variables and properties
    name = None
    """str: the name of the grid."""

    range = None
    """numpy.array(dtype=int) or None : the range where the grid coordinates to be repeated in every width of the range.
    For example, if the range is np.array([10, 50]), its base coordinates are defined over [10, 50] and the grid pattern 
    is repeated in every 40 (=50-10)."""

    phy2abs = None
    """_PhyToAbsGridConverter(master=self): Physical-to-abstract converter."""

    abs2phy = None
    """_AbsToPhyGridConverter(master=self): Abstract-to-physical converter."""

    @property
    def width(self):
        """float: the width of the grid."""
        return abs(self.range[1] - self.range[0])

    # Constructor
    def __init__(self, name, scope, elements=np.array([0])):
        """
        Constructor.

        Parameters
        ----------
        name: str
            The name of the template.
        scope: numpy.ndarray(dtype=int)
            The scope of the grid where its coordinates are defined is defined. Its format is [start, stop].
        elements: numpy.ndarray or list[int]
        """
        self.name = name
        self.range = np.asarray(scope)
        self.phy2abs = _PhyToAbsGridConverter(master=self)
        self.abs2phy = _AbsToPhyGridConverter(master=self)
        CircularMapping.__init__(self, elements=elements)
        # self.elements = np.asarray(elements)  # commented out because asarray does not woke well with Object arrays.

    # Indexing and slicing functions
    def __getitem__(self, pos):
        """Returns the physical coordinate corresponding to the abstract coordinate pos. """
        return self.abs2phy([pos])

    # Comparison operators
    def __eq__(self, other):
        """Returns the abstract grid coordinate that matches to other."""
        return self.phy2abs.__eq__(other)

    def __lt__(self, other):
        """Returns the abstract grid coordinate that is the largest but less than other."""
        return self.phy2abs.__lt__(other)

    def __le__(self, other):
        """Returns the index of the grid coordinate that is the largest but less than or equal to other."""
        return self.phy2abs.__le__(other)

    def __gt__(self, other):
        """Returns the abstract grid coordinate that is the smallest but greater than other."""
        return self.phy2abs.__gt__(other)

    def __ge__(self, other):
        """Returns the index of the grid coordinate that is the smallest but greater than or equal to other."""
        return self.phy2abs.__ge__(other)

    # Informative functions
    def __str__(self):
        """Returns the string representation of the object."""
        return self.summarize()

    def summarize(self):
        """Returns the summary of the object information."""
        return self.__repr__() + " " \
                                 "name: " + self.name + ", " + \
                                 "class: " + self.__class__.__name__ + ", " + \
                                 "scope: " + str(self.range) + ", " + \
                                 "elements: " + str(self.elements)

    # I/O functions
    def export_to_dict(self):
        """Exports the grid information as a dictionary."""
        export_dict = {
                       'scope': self.range.tolist(),
                       'elements': self.elements.tolist(),
                       }
        return export_dict


class Grid:
    """
    Basic two-dimensional layout grid.
    """

    name = None
    """str: the name of the grid."""

    _xy = None
    """List[OneDimGrid]: the list contains the 1d-grid objects for x and y axes."""

    @property
    def elements(self):
        return [self._xy[0].elements, self._xy[1].elements]
    """List[OneDimGrid]: the list contains the 1d-grid objects for x and y axes."""

    phy2abs = None
    """PhyToAbsGridConverter(master=self)"""

    abs2phy = None
    """AbsToPhyGridConverter(master=self)"""

    @property
    def xy(self):
        return self.abs2phy

    @property
    def x(self):
        return self._xy[0].abs2phy
    """None or OneDimGrid: the grid along the x-axis."""

    @property
    def y(self):
        return self._xy[1].abs2phy
    """None or OneDimGrid: the grid along the y-axis."""

    @property
    def v(self):
        return self.x
    """None or OneDimGrid: the grid along the x-axis."""

    @property
    def h(self):
        return self.y
    """None or OneDimGrid: the grid along the y-axis."""

    @property
    def mn(self):
        return self.phy2abs

    @property
    def m(self):
        return self._xy[0].phy2abs

    @property
    def n(self):
        return self._xy[1].phy2abs

    @property
    def shape(self):
        return np.hstack([self._xy[0].shape, self._xy[1].shape])

    def get_range(self):
        return np.transpose(np.vstack((self._xy[0].range, self._xy[1].range)))

    def set_range(self, value):
        self._xy[0].range = np.transpose(value)[0]
        self._xy[1].range = np.transpose(value)[1]

    range = property(get_range, set_range)
    """numpy.ndarray(dtype=int): the 2d-array that contains the range information of the x and y grids."""

    @property
    def width(self):
        """float: the width of the grid."""
        return self._xy[0].width

    @property
    def height(self):
        """float: the height of the grid."""
        return self._xy[1].width

    @property
    def height_vec(self):
        """numpy.ndarray(dtype=int): Returns np.array([0, height])."""
        return np.array([0, self.height])

    @property
    def width_vec(self):
        """numpy.ndarray(dtype=int): Returns np.array([width, 0])."""
        return np.array([self.width, 0])

    def __init__(self, name, vgrid, hgrid):
        """
        Constructor.

        Parameters
        ----------
        name : str
            The name of the template.
        vgrid : laygo2.object.grid.OndDimGrid
            Vertical grid object.
        """
        self.name = name
        self._xy = [vgrid, hgrid]
        self.phy2abs = _PhyToAbsGridConverter(master=self)
        self.abs2phy = _AbsToPhyGridConverter(master=self)

    @property
    def elements(self):
        """list: returns elements of subgrids ([_xy[0].elements, _xy[1].elements]). """
        return [self._xy[0].elements, self._xy[1].elements]

    # Indexing and slicing functions
    def __getitem__(self, pos):
        return self.abs2phy.__getitem__(pos)

    # Comparison operators
    def __eq__(self, other):
        """Returns the physical grid coordinate that matches to other."""
        return self.abs2phy.__eq__(other)

    def __lt__(self, other):
        """Returns the index of the grid coordinate that is the largest but less than other."""
        return self.abs2phy.__lt__(other)

    def __le__(self, other):
        """Returns the index of the grid coordinate that is the largest but less than or equal to other."""
        return self.abs2phy.__le__(other)

    def __gt__(self, other):
        """Returns the index of the grid coordinate that is the smallest but greater than other."""
        return self.abs2phy.__gt__(other)

    def __ge__(self, other):
        """Returns the index of the grid coordinate that is the smallest but greater than or equal to other."""
        return self.abs2phy.__ge__(other)

    def bbox(self, obj):
        """
        Returns the abstract grid coordinates corresponding to the 'internal' bounding box of obj.
        See _PhyToAbsGridConverter.bbox() for details.
        """
        return self.phy2abs.bbox(obj)

    def bottom_left(self, obj):
        """
        Returns the abstract grid coordinates corresponding to the bottom-left corner of obj.
        See _PhyToAbsGridConverter.bottom_left() for details.
        """
        return self.phy2abs.bottom_left(obj)

    def bottom_right(self, obj):
        """
        Returns the abstract grid coordinates corresponding to the bottom-right corner of obj.
        See _PhyToAbsGridConverter.bottom_right() for details.
        """
        return self.phy2abs.bottom_right(obj)

    def top_left(self, obj):
        """
        Returns the abstract grid coordinates corresponding to the top-left corner of obj.
        See _PhyToAbsGridConverter.top_left() for details.
        """
        return self.phy2abs.top_left(obj)

    def top_right(self, obj):
        """
        Returns the abstract grid coordinates corresponding to the top-right corner of obj.
        See _PhyToAbsGridConverter.top_right() for details.
        """
        return self.phy2abs.top_right(obj)

    def crossing(self, *args):
        """
        Returns the abstract grid coordinates corresponding to the crossing point of args.
        See _PhyToAbsGridConverter.crossing() for details.
        """
        return self.phy2abs.crossing(*args)

    def overlap(self, *args, type='bbox'):
        """
        Returns the abstract grid coordinates corresponding to the overlap of args.
        See _PhyToAbsGridConverter.overlap() for details.
        """
        return self.phy2abs.crossing(*args, type=type)

    def union(self, *args):
        """
        Returns the abstract grid coordinates corresponding to union of args.
        See _PhyToAbsGridConverter.union() for details.
        """
        return self.phy2abs.union(*args)

    # Iterators
    def __iter__(self):
        # TODO: fix this to iterate over the full coordinates
        return np.array([self._xy[0].__iter__(), self._xy[1].__iter__()])

    def __next__(self):
        # TODO: fix this to iterate over the full coordinates
        return np.array([self._xy[0].__next__(), self._xy[1].__next__()])

    # Informative functions
    def __str__(self):
        """Returns the string representation of the object."""
        return self.summarize()

    def summarize(self):
        """Returns the summary of the object information."""
        return self.__repr__() + " " \
                                 "name: " + self.name + ", " + \
               "class: " + self.__class__.__name__ + ", " + \
               "scope: " + str(self.range.tolist()) + ", " + \
               "elements: " + str(self.elements)


# Regular classes.
class PlacementGrid(Grid):
    """Placement grid class."""
    type = 'placement'

    def place(self, inst, mn):
        """Places an instance on the specified coordinate mn, on this grid."""
        inst.xy = self[mn]
        return inst


class RoutingGrid(Grid):
    """Routing grid class."""
    type = 'routing'
    vwidth = None
    """CircularMapping: the array containing the width of the routing wires on the vertical grid."""
    hwidth = None
    """CircularMapping: the array containing the width of the routing wires on the horizontal grid. """
    vextension = None
    """CircularMapping: the array containing the extension of the routing wires on the vertical grid."""
    hextension = None
    """CircularMapping: the array containing the extension of the routing wires on the horizontal grid. """
    vextension0 = None
    """CircularMapping: the array containing the extension of the zero-length wires on the vertical grid."""
    hextension0 = None
    """CircularMapping: the array containing the extension of the zero-length wires on the horizontal grid. """
    vlayer = None
    """CircularMapping: the array containing the layer info [name, purpose] of the routing wires on the vertical grid."""
    hlayer = None
    """CircularMapping: the array containing the layer info [name, purpose] of the routing wires on the horizontal grid."""
    pin_vlayer = None
    """CircularMapping: the array containing the pin layer info [name, purpose] of the routing wires on the vertical grid."""
    pin_hlayer = None
    """CircularMapping: the array containing the pin layer info [name, purpose] of the routing wires on the horizontal grid."""
    viamap = None
    """CircularMappingArray or None: the array containing the information of vias on the grid."""
    primary_grid = 'vertical'
    """str: The primary routing direction of the grid. Should be either vertical or horizontal. 
    Used when the direction of the routing wire is undetermined. """

    def __init__(self, name, vgrid, hgrid, vwidth, hwidth, vextension, hextension, vlayer, hlayer, pin_vlayer,
                 pin_hlayer, viamap, primary_grid='vertical', vextension0=None, hextension0=None):
        """
        Constructor.

        Parameters
        ----------
        name : str
            The name of the template.
        """
        self.vwidth = vwidth
        self.hwidth = hwidth
        self.vextension = vextension
        self.hextension = hextension
        if vextension0 is None:
            self.vextension0 = vextension
        else:
            self.vextension0 = vextension0
        if hextension0 is None:
            self.hextension0 = hextension
        else:
            self.hextension0 = hextension0
        self.vlayer = vlayer
        self.hlayer = hlayer
        self.pin_vlayer = pin_vlayer
        self.pin_hlayer = pin_hlayer
        self.viamap = viamap
        self.primary_grid = primary_grid
        Grid.__init__(self, name=name, vgrid=vgrid, hgrid=hgrid)

    def route(self, mn, direction=None, via_tag=None):
        """
        Creates Path and Via objects over xy-coordinates specified by mn, on this routing grid.

        Notes
        -----
        Initially, paths are used for implementing routing wires but they are replaced to rects, as paths cannot handle
        zero-length wires (with extensions) very well.
        """
        mn = np.asarray(mn)
        _mn = list()
        for i in range(1, mn.shape[0]):  
            # when more than two points are given,
            # create a multi-point wire compose of sub-routing wires
            # connecting the points given by mn in sequence.
            _mn.append([mn[i - 1, :], mn[i, :]])
        route = list()
        # via at the starting point
        if via_tag is not None:
            if via_tag[0] is True:
                route.append(self.via(mn=_mn[0][0], params=None))
        # routing wires
        for i, __mn in enumerate(_mn):
            xy0 = self.abs2phy[__mn[0]]
            xy1 = self.abs2phy[__mn[1]]
            _xy = np.array([[xy0[0], xy0[1]], [xy1[0], xy1[1]]])
            if np.all(xy0 == xy1):  # if two points are identical, generate a metal stub on the bottom layer.
                if (direction == 'vertical') or ((direction is None) and (self.primary_grid == 'vertical')):
                    width = self.vwidth[__mn[0][0]]
                    hextension = int(width/2)
                    vextension = self.vextension0[__mn[0][0]]
                    layer = self.vlayer[__mn[0][0]]
                else:
                    width = self.hwidth[__mn[0][1]]
                    hextension = self.hextension0[__mn[0][1]]
                    vextension = int(width/2)
                    layer = self.hlayer[__mn[0][1]]
            else:
                if (xy0[0] == xy1[0]) or (direction == 'vertical'):  # vertical routing
                    width = self.vwidth[__mn[0][0]]
                    hextension = int(width/2)
                    vextension = self.vextension[__mn[0][0]]
                    layer = self.vlayer[__mn[0][0]]
                else:  # horizontal routing
                    width = self.hwidth[__mn[0][1]]
                    hextension = self.hextension[__mn[0][1]]
                    vextension = int(width/2)
                    layer = self.hlayer[__mn[0][1]]
            p = laygo2.object.physical.Rect(xy=_xy, layer=layer, hextension=hextension, vextension=vextension)
            route.append(p)
            # via placement
            if via_tag is None:
                if (i > 0) and (i < mn.shape[0] - 1):
                    route.append(self.via(mn=__mn[0], params=None))
            else:
                if via_tag[i + 1] == True:
                    route.append(self.via(mn=__mn[1], params=None))
        if len(route) == 1:  # not isinstance(mn[0][0], list):
            return route[0]
        else:
            return route

    def via(self, mn=np.array([0, 0]), params=None):
        """Creates vias on xy-coordinates specified by mn, on this routing grid.

        """
        # If mn contains multiple coordinates (or objects), place iteratively.
        if isinstance(mn, list):
            if isinstance(list[0], (int, np.integer)):  # It's actually a single coordinate.
                return self.via(mn=np.asarray(mn), params=params)
            else:
                return [self.via(mn=_mn, params=params) for _mn in mn]
        elif isinstance(mn, np.ndarray):
            if isinstance(mn[0], (int, np.integer)):  # It's actually a single coordinate.
                pass
            else:
                return np.array([self.via(mn=_mn, params=params) for _mn in mn])
        if not isinstance(mn, tuple):
            mn = tuple(mn)  # viamap (CircularMapping) works only with tuples
        tvia = self.viamap[mn]
        via = tvia.generate(params=params)
        via.xy = self[mn]
        return via
    
    def route_via_track(self, mn, track ):
        """
        create multi routes on one track
        Parameters
        ----------
        mn : list
            The list of abstract point for routing, [ mn, mn...]
        track : list
            abstract point of track, [x, y]
        """
        mn = np.array( mn )    
        route   = list()
        
        if track[1] != None:                # x direction
            t = 0                           # index of track axis
            p = 1                           # index of perpendicular track
            mn_pivot = track[1]
        else:                               # y direction
            t = 1                           
            p = 0                           
            mn_pivot = track[0]

        mn_b       = np.array([ [0,0], [0,0]] )                    # 1.branch
        min_t, max_t = mn[0][t], mn[0][t]

        for i in range( len(mn) ) :
            mn_b[0]    = mn[i]
            mn_b[1][t] = mn_b[0][t]
            mn_b[1][p] = mn_pivot
            if np.array_equal( mn_b[0] , mn_b[1] ) :           #### via only
                route.append(self.via( mn= mn_b[0], params=None))
            else:
                route.append( self.route( mn= [ mn_b[0], mn_b[1] ], via_tag=[None,True] ) )

            center_t = mn[i][t]
            if center_t < min_t:
                min_t = center_t
            elif max_t < center_t:
                max_t = center_t

        mn_track = np.array([ [0,0], [0,0]] )                       # 2.track
        mn_track[0][t], mn_track[0][p] = min_t, mn_pivot  # min
        mn_track[1][t], mn_track[1][p] = max_t, mn_pivot # max

        if np.array_equal( mn_track[0] , mn_track[1] )  :  # Skip
            route.append(None)
        else:
            route.append( self.route( mn= mn_track ) )
        
        return route
 
    def pin(self, name, mn, direction=None, netname=None, params=None):
        #pin0 = Pin(xy=[[0, 0], [100, 100]], layer=['M1', 'drawing'], netname='net0', master=rect0,
        #           params={'direction': 'input'})
        """Creates a pin object over xy-coordinates specified by mn, on this routing grid. """
        xy0 = self.abs2phy[mn[0]]
        xy1 = self.abs2phy[mn[1]]
        #_xy = np.array([[xy0[0], xy0[1]], [xy1[0], xy1[1]]])
        if np.all(xy0 == xy1):  # if two points are identical, generate a metal stub on the bottom layer.
            if (direction == 'vertical') or ((direction is None) and (self.primary_grid == 'vertical')):
                width = self.vwidth[mn[0][0]]
                hextension = int(width / 2)
                vextension = 0
                layer = self.pin_vlayer[mn[0][0]]
            else:
                width = self.hwidth[mn[0][1]]
                hextension = 0
                vextension = int(width / 2)
                layer = self.pin_hlayer[mn[0][1]]
        else:
            if (xy0[0] == xy1[0]) or (direction == 'vertical'):  # vertical routing
                width = self.vwidth[mn[0][0]]
                hextension = int(width / 2)
                vextension = 0
                layer = self.pin_vlayer[mn[0][0]]
            else:  # horizontal routing
                width = self.hwidth[mn[0][1]]
                hextension = 0
                vextension = int(width / 2)
                layer = self.pin_hlayer[mn[0][1]]
        _xy = np.array([[xy0[0]-hextension, xy0[1]-vextension], [xy1[0]+hextension, xy1[1]+vextension]])
        p = laygo2.object.physical.Pin(name=name, xy=_xy, layer=layer, netname=netname, params=params)
        return p


class ParameterizedGrid(Grid):
    """A parameterized grid to support flexible templates."""
    # TODO: implement this.
    pass

class ParameterizedPlacementGrid(Grid):
    # TODO: implement this.
    pass

class ParameterizedRoutingGrid(Grid):
    # TODO: implement this.
    pass

# Tests.
if __name__ == '__main__':
    print("Testing 1d grid")
    g1a = OneDimGrid(name='mygrid', scope=[0, 100], elements=[0, 10, 20, 40, 50])
    g1b = OneDimGrid(name='mygrid', scope=[0, 200], elements=[10, 20, 25])
    print(g1a)
    print(g1a.elements)
    test_pair = [
        [g1a[:10], np.array([0, 10, 20, 40, 50, 100, 110, 120, 140, 150])],
        [g1a[1:], np.array([10, 20, 40, 50])],
        [g1a[-6:10], np.array([-150, -100, -90, -80, -60, -50, 0, 10, 20, 40, 50, 100, 110, 120, 140, 150])],
        [g1a[[3, 5, 7]], np.array([40, 100, 120])],
        [g1a == [-200, -180, -50, 10, 60, 110, 220, 250], np.array([-10, -8, -1, 1, None, 6, 12, 14])],
        [g1a[g1a == [-200, -180, -50, -45, 10, 60, 110, 220, 250]], np.array([-200, -180, -50, None, 10, None, 110, 220, 250])],
        [g1a == g1b, np.array([1, 2, None])],
        [g1a[g1a == g1b], np.array([10, 20, None])],
        [g1a < -60, -3],
        [g1a < 40, 2],
        [g1a < 140, 7],
        [g1a < 145, 8],
        [g1a <= -60, -2],
        [g1a <= 40, 3],
        [g1a <= 140, 8],
        [g1a <= 145, 8],
        [g1a > 20, 3],
        [g1a >= 20, 2],
        [g1a > 140, 9],
        [g1a >= 140, 8],
        [g1a.abs2phy == 1, 10],
        [g1a.abs2phy <= 1, 10],
        [g1a.abs2phy < 1, 0],
        [g1a.abs2phy == 10, 200],
        [g1a.abs2phy <= 10, 200],
        [g1a.abs2phy < 10, 150],
    ]
    for q, a in test_pair:
        print(q, np.all(q == a))

    print("Testing 2d grid")
    g2 = Grid(name='mygrid', vgrid=g1a, hgrid=g1b)
    print(g2)
    print(g2.xy)
    test_pair = [
        [g2[2, 5], np.array([20, 225])],
        [g2[2, :], np.array([[20, 10], [20, 20], [20, 25]])],
        [g2[0:8, 1:3], np.array([[[0, 20], [0, 25]], [[10, 20], [10, 25]], [[20, 20], [20, 25]], [[40, 20], [40, 25]],
                                 [[50, 20], [50, 25]], [[100, 20], [100, 25]], [[110, 20], [110, 25]],
                                 [[120, 20], [120, 25]]])],
        [g2 == [20, 10], np.array([2, 0])],
        [g2 == [[20, 10], [40, 25]], np.array([[2, 0], [3, 2]])],
        [g2.phy2abs[[20, 40], [10, 25]], np.array([[[2, 0], [2, 2]], [[3, 0], [3, 2]]])],
    ]
    for q, a in test_pair:
        print(q, np.all(q == a))
    print(g2 < [20, 10])
    print(g2 <= [20, 10])
    print(g2 > [20, 10])
    print(g2 >= [20, 10])
    print(g2.y.elements)
    print('test', g2.y > 10)
    print('test', g2.y >= 10)

    print("Testing 2d routing grid")
    #width = CircularMapping(elements=[[2, 4], [2, 4], [2, 4], [5, 4], [5, 4]])
    #_l = [['M1', 'drawing'], ['M2', 'drawing']]
    #layer = CircularMapping(elements=[[_l]*5]*3, dtype=object)
    vwidth = CircularMapping(elements=[2, 2, 2, 5, 5])
    hwidth = CircularMapping(elements=[4, 4, 4])
    vextension = CircularMapping(elements=[1, 1, 1, 1, 1])
    hextension = CircularMapping(elements=[1, 1, 1])
    vlayer = CircularMapping(elements=[['M1', 'drawing']] * 5, dtype=object)
    hlayer = CircularMapping(elements=[['M2', 'drawing']] * 3, dtype=object)
    from laygo2.object.template import NativeInstanceTemplate
    tvia0 = NativeInstanceTemplate(libname='mylib', cellname='mynattemplate', bbox=[[0, 0], [100, 100]])
    viamap = CircularMappingArray(elements=np.array([[tvia0]*5]*3), dtype=object)
    #ivia0 = tvia0.generate(name='IV0', xy=[0, 0], transform='R0')
    #via_inst = ivia0
    #viamap = CircularMapping(elements=np.array([[via_inst]*5]*3), dtype=object)
    #g3 = RoutingGrid(name='mygrid', x=g1a, y=g1b, widthmap=width, layermap=layer, viamap=viamap)
    g3 = RoutingGrid(name='mygrid', vgrid=g1a, hgrid=g1b, vwidth=vwidth, hwidth=hwidth, vextension=vextension,
                     hextension=hextension, vlayer=vlayer, hlayer=hlayer, viamap=viamap)
    print(g3.x)
    print(g3.y)
    print(g3.vwidth)
    print(g3.hwidth)
    print(g3.vlayer)
    print(g3.hlayer)
    print(g3.viamap)
    print(g3.viamap[2, 3])
    print(g3.viamap[-5, -4])
    print(g3.viamap[15, 32])
    print(g3.viamap[3:5, -2:8])
    test_pair = [
        [g3[2, 5], np.array([20, 225])],
        [g3[2, :], np.array([[20, 10], [20, 20], [20, 25]])],
        [g3[0:8, 1:3], np.array([[[0, 20], [0, 25]], [[10, 20], [10, 25]], [[20, 20], [20, 25]], [[40, 20], [40, 25]],
                                 [[50, 20], [50, 25]], [[100, 20], [100, 25]], [[110, 20], [110, 25]],
                                 [[120, 20], [120, 25]]])],
        [g3.mn[40, 25], np.array([3, 2])],
        [g3 == [20, 10], np.array([2, 0])],
        [g3 == [[20, 10], [40, 25]], np.array([[2, 0], [3, 2]])],
        [g3.phy2abs[[20, 40], [10, 25]], np.array([[[2, 0], [2, 2]], [[3, 0], [3, 2]]])],
    ]
    for q, a in test_pair:
        print(q, np.all(q == a))

    '''
    def summarize(self):
        """Returns the summary of the grid information."""
        return self.__repr__() + " " \
                                 "name: " + self.name + ", " + \
                                 "class: " + self.__class__.__name__ + ", " + \
                                 " width:" + str(self.width) + \
                                 " height:" + str(self.height)

    def export(self, fmt='dict'):
        """Exports the object information."""
        if fmt == 'dict':
            return self.export_dict()

    def export_dict(self):
        """Exports the object information as a dictionary."""
        export_dict={'type': self.type,
                     'xy0': self.xy0.tolist(),
                     'xy1': self.xy1.tolist(),
                     'xgrid': self.xgrid.tolist(),
                     'ygrid': self.ygrid.tolist(),
                     }
        return export_dict

    def _add_grid(self, grid, v):
        """
        add a grid coordinate

        Parameters
        ----------
        grid : np.array
            grid array
        v : float
            value

        Returns
        -------
        np.array
            updated grid array
        """
        grid.append(v)
        grid.sort()
        return grid

    def add_xgrid(self, x):
        """
        add a coordinate value to grid in x direction

        Parameters
        ----------
        x : float
            value to be added
        """
        self.xgrid=self._add_grid(self.xgrid, x)

    def add_ygrid(self, y):
        """
        add a coordinate value to grid in y direction

        Parameters
        ----------
        y : float
            value to be added
        """
        self.ygrid=self._add_grid(self.ygrid, y)

    def get_xgrid(self):
        """use laygo2.GridObject.GridObject.xgrid instead"""
        return self.xgrid

    def get_ygrid(self):
        """use laygo2.GridObject.GridObject.ygrid instead"""
        return self.ygrid

    #abstract to physical conversion functions
    def _get_absgrid_coord(self, v, grid, size, method='ceil'):
        """
        Convert physical value to abstract value on grid. Since physical values are continuous and grid values are
        discrete, boundary conditions are need to be defined for the conversion.
        Default (and only supported in the current version) converting method is ceil, which is described as follows:
        #   physical grid: 0----grid0----grid1----grid2----...----gridN---size
        # abstracted grid: 0  0   0    1   1    2   2           N   N   N+1
        The ceiling method works well with stacked grids. For example, in the example, phyical values ranging from
        (larger than) GridN to (equal to) gridN+1 (grid0 of the second stack) will be converted to N+1.

        Parameters
        ----------
        v : float
            grid value
        grid : np.array
            grid array
        size : float
            grid size
        method : str, optional
            converting method

        Returns
        -------
        int
            abstract grid value
        """
        quo=np.floor(np.divide(v, size))
        mod=v-quo*size #mod=np.mod(v, size) not working well
        mod_ongrid=np.searchsorted(grid+1e-10, mod) #add 1e-10 to handle floating point precision errors
        #print('physical:' + str(v.tolist()) + ' size:'+ str(size) +
        #      ' quo:' + str(quo.tolist()) + ' mod:' + str(mod) +
        #      ' abs:' + str(np.add(np.multiply(quo,grid.shape[0]), mod_ongrid).tolist()))
        return np.add(np.multiply(quo,grid.shape[0]), mod_ongrid)

    def get_absgrid_coord_x(self, x): return self.get_absgrid_x(x)
    def get_absgrid_x(self, x):
        """
        Convert a physical x coordinate to a corresponding value in abstract grid

        Parameters
        ----------
        x : float
            value

        Returns
        -------
        int
            converted value
        """
        return self._get_absgrid_coord(x, self.xgrid, self.width).astype(int)

    def get_absgrid_coord_y(self, y): return self.get_absgrid_y(y)
    def get_absgrid_y(self, y):
        """
        Convert a physical y coordinate to a corresponding value in abstract grid

        Parameters
        ----------
        y : float
            value

        Returns
        -------
        int
            converted value
        """
        return self._get_absgrid_coord(y, self.ygrid, self.height).astype(int)

    def get_absgrid_coord_xy(self, xy): return self.get_absgrid_xy(xy)
    def get_absgrid_xy(self, xy):
        """
        Convert a physical xy coordinate to a corresponding vector in abstract grid

        Parameters
        ----------
        xy : np.array([float, float])

        Returns
        -------
        np.array([int, int])
        """
        _xy=np.vstack((self.get_absgrid_x(xy.T[0]), self.get_absgrid_y(xy.T[1]))).T
        if _xy.shape[0]==1: return _xy[0]
        else: return _xy

    def get_absgrid_coord_region(self, xy0, xy1): return self.get_absgrid_region(xy0, xy1)
    def get_absgrid_region(self, xy0, xy1):
        """
        Convert a physical regin to a corresponding region in abstract grid

        Parameters
        ----------
        xy0 : np.array([float, float])
            first coordinate
        xy1 : np.array([float, float])
            second coordinate

        Returns
        -------
        np.array([[int, int], [int, int]])
            converted coordinate
        """
        _xy0 = np.vstack((self.get_absgrid_coord_x(xy0.T[0]), self.get_absgrid_coord_y(xy0.T[1]))).T
        _xy1 = np.vstack((self.get_absgrid_coord_x(xy1.T[0]), self.get_absgrid_coord_y(xy1.T[1]))).T
        if _xy0.shape[0] == 1: _xy0 = _xy0[0]
        if _xy1.shape[0] == 1: _xy1 = _xy1[0]
        #upper right boundary adjust
        #check by re-converting to physical grid and see if the points are within original [xy0, xy1]
        #xy0_check = self.get_phygrid_coord_xy(_xy0)[0]
        #xy1_check = self.get_phygrid_coord_xy(_xy1)[0]
        xy0_check = self.get_phygrid_xy(_xy0)
        xy1_check = self.get_phygrid_xy(_xy1)
        xy0_check = np.around(xy0_check, decimals=self.max_resolution)
        xy1_check = np.around(xy1_check, decimals=self.max_resolution)
        xy0 = np.around(xy0, decimals=self.max_resolution)
        xy1 = np.around(xy1, decimals=self.max_resolution)
        #print("phy:" + str(xy0) + " " + str(xy1) + " abs:" + str(_xy0) + " " + str(_xy1) + " chk:" + str(xy0_check) + " " + str(xy1_check))
        if xy0_check[0] > xy0[0] and xy0_check[0] > xy1[0]: _xy0[0] -= 1
        if xy1_check[0] > xy0[0] and xy1_check[0] > xy1[0]: _xy1[0] -= 1
        if xy0_check[1] > xy0[1] and xy0_check[1] > xy1[1]: _xy0[1] -= 1
        if xy1_check[1] > xy0[1] and xy1_check[1] > xy1[1]: _xy1[1] -= 1
        #if _xy1[1]==7:
        #    print("phy:"+str(xy0)+" "+str(xy1)+" abs:"+str(_xy0)+" "+str(_xy1)+" chk:"+str(xy0_check)+" "+str(xy1_check))
        #print(xy1)

        return(np.vstack((_xy0, _xy1)))

    # physical to abstract conversion functions
    def _get_phygrid_coord(self, v, grid, size):
        """
        Internal function for abstract to physical coordinate conversions

        Parameters
        ----------
        v : int
            value in abstract coordinate
        grid : np.array
            grid array
        size : float
            grid size

        Returns
        -------
        float
            value in physical coordinate
        """
        quo = np.floor(np.divide(v, grid.shape[0]))
        mod = np.mod(v, grid.shape[0])  # mod = v - quo * grid.shape[0]
        return np.add(np.multiply(quo, size), np.take(grid, mod))

    def get_phygrid_coord_x(self, x): return self.get_phygrid_x(x)
    def get_phygrid_x(self, x):
        """
        Get the physical value of a abstract x coordinate

        Parameters
        ----------
        x : int
            abstract coordinate value

        Returns
        -------
        float
            physical coordinate value
        """
        return self._get_phygrid_coord(x, self.xgrid, self.width)

    def get_phygrid_coord_y(self, y): return self.get_phygrid_y(y)
    def get_phygrid_y(self, y):
        """
        Get the physical value of a abstract y coordinate

        Parameters
        ----------
        y : int
            abstract coordinate value

        Returns
        -------
        float
            physical coordinate value
        """
        return self._get_phygrid_coord(y, self.ygrid, self.height)

    def get_phygrid_coord_xy(self, xy): return self.get_phygrid_xy(xy)
    def get_phygrid_xy(self, xy):
        """
        Get the physical value of a abstract vector

        Parameters
        ----------
        xy : np.array([int, int])
            abstract coordinate vector

        Returns
        -------
        np.array([float, float])
            physical coordinate vector
        """
        if xy.ndim==1: #single coordinate
            return np.vstack((self.get_phygrid_coord_x(xy.T[0]), self.get_phygrid_coord_y(xy.T[1]))).T[0]
        else:
            return np.vstack((self.get_phygrid_coord_x(xy.T[0]), self.get_phygrid_coord_y(xy.T[1]))).T


class PlacementGrid(GridObject):
    """
    Placement grid class
    """
    type = 'placement'


class RouteGrid(GridObject):
    """
    Routing grid class
    """
    type = 'route'
    """str: type of grid. can be 'native', 'route', 'placement'"""
    xwidth = np.array([])
    """np.array: width of xgrid"""
    ywidth = np.array([])
    """np.array: width of ygrid"""
    viamap = dict()
    """dict: via map information"""
    xlayer = None
    """list: layer of xgrid"""
    ylayer = None
    """list: layer of ygrid"""

    def __init__(self, name, libname, xy, xgrid, ygrid, xwidth, ywidth, xlayer=None, ylayer=None, viamap=None):
        """
        Constructor
        """
        self.name = name
        self.libname = libname
        self.xy = xy
        self.xgrid = xgrid
        self.ygrid = ygrid
        self.xwidth = xwidth
        self.ywidth = ywidth
        self.xlayer = xlayer
        self.ylayer = ylayer
        self.viamap = viamap

    def _get_route_width(self, v, _width):
        """
        Get metal width

        Parameters
        ----------
        v : int
            value
        _width : np.array
            width vector

        Returns
        -------
        float
            route width
        """
        #quo=np.floor(np.divide(v, self._width.shape[0]))
        mod=np.mod(v, _width.shape[0])
        #if not isinstance(mod, int):
        #    print(v, _width, mod)
        return _width[round(np.round(mod))]

    def _get_route_layer(self, v, _layer):
        """
        Get metal layer

        Parameters
        ----------
        v : int
            index value
        _layer : list
            layer list

        Returns
        -------
        [str, str]
            route layer
        """
        mod=np.mod(v, len(_layer))
        return _layer[mod]

    #def get_xwidth(self): return self.xwidth
    #def get_ywidth(self): return self.ywidth
    #def get_viamap(self): return self.viamap

    def get_route_width_xy(self, xy):
        """
        Get metal width vector

        Parameters
        ----------
        xy : np.array([int, int])
            coordinate

        Returns
        -------
        np.array([float, float])
            route width vector ([xgrid, ygrid])
        """
        return np.array([self._get_route_width(xy[0], self.xwidth),
                         self._get_route_width(xy[1], self.ywidth)])

    def get_route_xlayer_xy(self, xy):
        """
        Get route layer in xgrid direction (vertical)

        Parameters
        ----------
        xy : np.array([int, int])
            coordinate

        Returns
        -------
        [str, str]
            route layer information
        """
        return self._get_route_layer(xy[0], self.xlayer)

    def get_route_ylayer_xy(self, xy):
        """
        Get route layer in ygrid direction (horizontal)

        Parameters
        ----------
        xy : np.array([int, int])
            coordinate

        Returns
        -------
        [str, str]
            route layer information
        """
        return self._get_route_layer(xy[1], self.ylayer)

    def get_vianame(self, xy):
        """
        Get the name of via on xy

        Parameters
        ----------
        xy : np.array([int, int])
            coordinate

        Returns
        -------
        str
            cellname of via
        """
        mod = np.array([np.mod(xy[0], self.xgrid.shape[0]), np.mod(xy[1], self.ygrid.shape[0])])
        for vianame, viacoord in self.viamap.items():
            if viacoord.ndim==1:
                if np.array_equal(mod, viacoord): return vianame
            else:
                for v in viacoord:
                    if np.array_equal(mod,v):
                        return vianame

    def display(self):
        """
        Display RouteGrid information
        """
        display_str="  " + self.name + " width:" + str(np.around(self.width, decimals=self.max_resolution))\
                    + " height:" + str(np.around(self.height, decimals=self.max_resolution))\
                    + " xgrid:" + str(np.around(self.xgrid, decimals=self.max_resolution))\
                    + " ygrid:" + str(np.around(self.ygrid, decimals=self.max_resolution))\
                    + " xwidth:" + str(np.around(self.xwidth, decimals=self.max_resolution))\
                    + " ywidth:" + str(np.around(self.ywidth, decimals=self.max_resolution))\
                    + " xlayer:" + str(self.xlayer)\
                    + " ylayer:" + str(self.ylayer)\
                    + " viamap:{"
        for vm_name, vm in self.viamap.items():
            display_str+=vm_name + ": " + str(vm.tolist()) + " "
        display_str+="}"
        print(display_str)

    def export_dict(self):
        """
        Export grid information in dict format

        Returns
        -------
        dict
            Grid information
        """
        export_dict=GridObject.export_dict(self)
        export_dict['xwidth'] = np.around(self.xwidth, decimals=self.max_resolution).tolist()
        export_dict['ywidth'] = np.around(self.ywidth, decimals=self.max_resolution).tolist()
        export_dict['xlayer'] = self.xlayer
        export_dict['ylayer'] = self.ylayer
        export_dict['viamap'] = dict()
        for vn, v in self.viamap.items():
            export_dict['viamap'][vn]=[]
            for _v in v:
                export_dict['viamap'][vn].append(_v.tolist())
        return export_dict

    def update_viamap(self, viamap):
        """use laygo2.GridObject.RouteGrid.viamap instead"""
        self.viamap=viamap

if __name__ == '__main__':
    lgrid=GridObject()
    print('LayoutGrid test')
    lgrid.xgrid = np.array([0.2, 0.4, 0.6])
    lgrid.ygrid = np.array([0, 0.4, 0.9, 1.2, 2, 3])
    lgrid.width = 1.2
    lgrid.height = 3.2
    phycoord = np.array([[-0.2, -0.2], [0, 2], [4,2.2], [0.5, 1.5], [1.3, 3.6], [8,2.3]])
    print("  xgrid:" + str(lgrid.xgrid) + " width:" + str(lgrid.width))
    print("  ygrid:" + str(lgrid.ygrid) + " height:" + str(lgrid.height))
    print('physical grid to abstract grid')
    print("  input:"+str(phycoord.tolist()))
    abscoord=lgrid.get_absgrid_coord_xy(phycoord)
    print("  output:"+str(abscoord.tolist()))
    print('abstract grid to physical grid')
    phycoord=lgrid.get_phygrid_xy(abscoord).tolist()
    print("  output:"+str(phycoord))
'''
'''
class Grid():
    """
    Basic layout grid.
    """

    name = None
    """str: the name of the grid."""

    elements = None
    """list(numpy.ndarray): the list contains the physical coordinates as numpy array."""

    def __getitem__(self, pos):
        """Returns the physical coordinate corresponding to the pos parameter."""
        if isinstance(self.elements, list):  # multi-dimensional grid
            p = np.asarray(pos)
            p_divisor = [np.shape(e)[0] - 1 for e in self.elements]
            # get the modular of pos
            p_mod = np.remainder(p, p_divisor)
            # return the corresponding coordinates
            return np.array([self.elements[i][p_mod[i]] for i in range(len(self.elements))])
        else:  # 1d grid
            return self.elements(pos)

    def __iter__(self):
        """Iterator function. Directly mapped to its elements."""
        return self.elements.__iter__()

    def __next__(self):
        """Iterator function. Directly mapped to its elements."""
        return self.elements.__next__()

    #type="native"
    #"""str: the type of the grid object. Can be "native", "route", "placement". """


    _xy = np.array([[0, 0], [0, 0]])
    """numpy.ndarray(dtype=numpy.int): the internal variable of xy."""

    def get_xy(self):
        """numpy.ndarray(dtype=int): gets the xy."""
        return self._xy

    def set_xy(self, value):
        """numpy.ndarray(dtype=int): sets the xy."""
        self._xy = np.asarray(value, dtype=np.int)

    xy = property(get_xy, set_xy)
    """numpy.ndarray(dtype=int): the size of the grid object."""

    xgrid = np.array([])
    """numpy.array(dtype=int): the valid x coordinates of the grid."""

    ygrid = np.array([])
    """numpy.array(dtype=int): the valid y coordinates of the grid."""

    def xy0(self):
        """numpy.ndarray(dtype=int): returns the primary corner of the grid."""
        return self.xy[0]

    def xy1(self):
        """numpy.ndarray(dtype=int): returns the secondary corner of the grid."""
        return self.xy[1]

    @property
    def height(self):
        """float: the height of the grid."""
        return abs(self.xy[1, 1]-self.xy[0, 1])

    @property
    def width(self):
        """float: the width of the grid."""
        return abs(self.xy[1, 0]-self.xy[0, 0])

    @property
    def height_vec(self):
        """numpy.ndarray(dtype=int): Returns np.array([0, height])."""
        return np.array([0, self.height])

    @property
    def width_vec(self):
        """numpy.ndarray(dtype=int): Returns np.array([width, 0])."""
        return np.array([self.width, 0])

    def __init__(self, name, xy, xgrid=np.array([0]), ygrid=np.array([0])):
        """
        Constructor.

        Parameters
        ----------
        name : str
            The name of the template.
        xy : numpy.ndarray(dtype=int)
            The size of the grid.
        xgrid : numpy.ndarray(dtype=int)
        ygrid : numpy.ndarray(dtype=int)
        """
        self.name = name
        self.xy = xy
        self.xgrid = xgrid
        self.ygrid = ygrid

    def summarize(self):
        """Returns the summary of the grid information."""
        return self.__repr__() + " " \
                                 "name: " + self.name + ", " + \
               "class: " + self.__class__.__name__ + ", " + \
               " width:" + str(self.width) + \
               " height:" + str(self.height)

    def export(self, fmt='dict'):
        """Exports the object information."""
        if fmt == 'dict':
            return self.export_dict()

    def export_dict(self):
        """Exports the object information as a dictionary."""
        export_dict={'type': self.type,
                     'xy0': self.xy0.tolist(),
                     'xy1': self.xy1.tolist(),
                     'xgrid': self.xgrid.tolist(),
                     'ygrid': self.ygrid.tolist(),
                     }
        return export_dict

    def _add_grid(self, grid, v):
        """
        add a grid coordinate

        Parameters
        ----------
        grid : np.array
            grid array
        v : float
            value

        Returns
        -------
        np.array
            updated grid array
        """
        grid.append(v)
        grid.sort()
        return grid

    def add_xgrid(self, x):
        """
        add a coordinate value to grid in x direction

        Parameters
        ----------
        x : float
            value to be added
        """
        self.xgrid=self._add_grid(self.xgrid, x)

    def add_ygrid(self, y):
        """
        add a coordinate value to grid in y direction

        Parameters
        ----------
        y : float
            value to be added
        """
        self.ygrid=self._add_grid(self.ygrid, y)

    def get_xgrid(self):
        """use laygo2.GridObject.GridObject.xgrid instead"""
        return self.xgrid

    def get_ygrid(self):
        """use laygo2.GridObject.GridObject.ygrid instead"""
        return self.ygrid

    #abstract to physical conversion functions
    def _get_absgrid_coord(self, v, grid, size, method='ceil'):
        """
        Convert physical value to abstract value on grid. Since physical values are continuous and grid values are
        discrete, boundary conditions are need to be defined for the conversion.
        Default (and only supported in the current version) converting method is ceil, which is described as follows:
        #   physical grid: 0----grid0----grid1----grid2----...----gridN---size
        # abstracted grid: 0  0   0    1   1    2   2           N   N   N+1
        The ceiling method works well with stacked grids. For example, in the example, phyical values ranging from
        (larger than) GridN to (equal to) gridN+1 (grid0 of the second stack) will be converted to N+1.

        Parameters
        ----------
        v : float
            grid value
        grid : np.array
            grid array
        size : float
            grid size
        method : str, optional
            converting method

        Returns
        -------
        int
            abstract grid value
        """
        quo=np.floor(np.divide(v, size))
        mod=v-quo*size #mod=np.mod(v, size) not working well
        mod_ongrid=np.searchsorted(grid+1e-10, mod) #add 1e-10 to handle floating point precision errors
        #print('physical:' + str(v.tolist()) + ' size:'+ str(size) +
        #      ' quo:' + str(quo.tolist()) + ' mod:' + str(mod) +
        #      ' abs:' + str(np.add(np.multiply(quo,grid.shape[0]), mod_ongrid).tolist()))
        return np.add(np.multiply(quo,grid.shape[0]), mod_ongrid)

    def get_absgrid_coord_x(self, x): return self.get_absgrid_x(x)
    def get_absgrid_x(self, x):
        """
        Convert a physical x coordinate to a corresponding value in abstract grid

        Parameters
        ----------
        x : float
            value

        Returns
        -------
        int
            converted value
        """
        return self._get_absgrid_coord(x, self.xgrid, self.width).astype(int)

    def get_absgrid_coord_y(self, y): return self.get_absgrid_y(y)
    def get_absgrid_y(self, y):
        """
        Convert a physical y coordinate to a corresponding value in abstract grid

        Parameters
        ----------
        y : float
            value

        Returns
        -------
        int
            converted value
        """
        return self._get_absgrid_coord(y, self.ygrid, self.height).astype(int)

    def get_absgrid_coord_xy(self, xy): return self.get_absgrid_xy(xy)
    def get_absgrid_xy(self, xy):
        """
        Convert a physical xy coordinate to a corresponding vector in abstract grid

        Parameters
        ----------
        xy : np.array([float, float])

        Returns
        -------
        np.array([int, int])
        """
        _xy=np.vstack((self.get_absgrid_x(xy.T[0]), self.get_absgrid_y(xy.T[1]))).T
        if _xy.shape[0]==1: return _xy[0]
        else: return _xy

    def get_absgrid_coord_region(self, xy0, xy1): return self.get_absgrid_region(xy0, xy1)
    def get_absgrid_region(self, xy0, xy1):
        """
        Convert a physical regin to a corresponding region in abstract grid

        Parameters
        ----------
        xy0 : np.array([float, float])
            first coordinate
        xy1 : np.array([float, float])
            second coordinate

        Returns
        -------
        np.array([[int, int], [int, int]])
            converted coordinate
        """
        _xy0 = np.vstack((self.get_absgrid_coord_x(xy0.T[0]), self.get_absgrid_coord_y(xy0.T[1]))).T
        _xy1 = np.vstack((self.get_absgrid_coord_x(xy1.T[0]), self.get_absgrid_coord_y(xy1.T[1]))).T
        if _xy0.shape[0] == 1: _xy0 = _xy0[0]
        if _xy1.shape[0] == 1: _xy1 = _xy1[0]
        #upper right boundary adjust
        #check by re-converting to physical grid and see if the points are within original [xy0, xy1]
        #xy0_check = self.get_phygrid_coord_xy(_xy0)[0]
        #xy1_check = self.get_phygrid_coord_xy(_xy1)[0]
        xy0_check = self.get_phygrid_xy(_xy0)
        xy1_check = self.get_phygrid_xy(_xy1)
        xy0_check = np.around(xy0_check, decimals=self.max_resolution)
        xy1_check = np.around(xy1_check, decimals=self.max_resolution)
        xy0 = np.around(xy0, decimals=self.max_resolution)
        xy1 = np.around(xy1, decimals=self.max_resolution)
        #print("phy:" + str(xy0) + " " + str(xy1) + " abs:" + str(_xy0) + " " + str(_xy1) + " chk:" + str(xy0_check) + " " + str(xy1_check))
        if xy0_check[0] > xy0[0] and xy0_check[0] > xy1[0]: _xy0[0] -= 1
        if xy1_check[0] > xy0[0] and xy1_check[0] > xy1[0]: _xy1[0] -= 1
        if xy0_check[1] > xy0[1] and xy0_check[1] > xy1[1]: _xy0[1] -= 1
        if xy1_check[1] > xy0[1] and xy1_check[1] > xy1[1]: _xy1[1] -= 1
        #if _xy1[1]==7:
        #    print("phy:"+str(xy0)+" "+str(xy1)+" abs:"+str(_xy0)+" "+str(_xy1)+" chk:"+str(xy0_check)+" "+str(xy1_check))
        #print(xy1)

        return(np.vstack((_xy0, _xy1)))

    # physical to abstract conversion functions
    def _get_phygrid_coord(self, v, grid, size):
        """
        Internal function for abstract to physical coordinate conversions

        Parameters
        ----------
        v : int
            value in abstract coordinate
        grid : np.array
            grid array
        size : float
            grid size

        Returns
        -------
        float
            value in physical coordinate
        """
        quo = np.floor(np.divide(v, grid.shape[0]))
        mod = np.mod(v, grid.shape[0])  # mod = v - quo * grid.shape[0]
        return np.add(np.multiply(quo, size), np.take(grid, mod))

    def get_phygrid_coord_x(self, x): return self.get_phygrid_x(x)
    def get_phygrid_x(self, x):
        """
        Get the physical value of a abstract x coordinate

        Parameters
        ----------
        x : int
            abstract coordinate value

        Returns
        -------
        float
            physical coordinate value
        """
        return self._get_phygrid_coord(x, self.xgrid, self.width)

    def get_phygrid_coord_y(self, y): return self.get_phygrid_y(y)
    def get_phygrid_y(self, y):
        """
        Get the physical value of a abstract y coordinate

        Parameters
        ----------
        y : int
            abstract coordinate value

        Returns
        -------
        float
            physical coordinate value
        """
        return self._get_phygrid_coord(y, self.ygrid, self.height)

    def get_phygrid_coord_xy(self, xy): return self.get_phygrid_xy(xy)
    def get_phygrid_xy(self, xy):
        """
        Get the physical value of a abstract vector

        Parameters
        ----------
        xy : np.array([int, int])
            abstract coordinate vector

        Returns
        -------
        np.array([float, float])
            physical coordinate vector
        """
        if xy.ndim==1: #single coordinate
            return np.vstack((self.get_phygrid_coord_x(xy.T[0]), self.get_phygrid_coord_y(xy.T[1]))).T[0]
        else:
            return np.vstack((self.get_phygrid_coord_x(xy.T[0]), self.get_phygrid_coord_y(xy.T[1]))).T


class PlacementGrid(GridObject):
    """
    Placement grid class
    """
    type = 'placement'


class RouteGrid(GridObject):
    """
    Routing grid class
    """
    type = 'route'
    """str: type of grid. can be 'native', 'route', 'placement'"""
    xwidth = np.array([])
    """np.array: width of xgrid"""
    ywidth = np.array([])
    """np.array: width of ygrid"""
    viamap = dict()
    """dict: via map information"""
    xlayer = None
    """list: layer of xgrid"""
    ylayer = None
    """list: layer of ygrid"""

    def __init__(self, name, libname, xy, xgrid, ygrid, xwidth, ywidth, xlayer=None, ylayer=None, viamap=None):
        """
        Constructor
        """
        self.name = name
        self.libname = libname
        self.xy = xy
        self.xgrid = xgrid
        self.ygrid = ygrid
        self.xwidth = xwidth
        self.ywidth = ywidth
        self.xlayer = xlayer
        self.ylayer = ylayer
        self.viamap = viamap

    def _get_route_width(self, v, _width):
        """
        Get metal width

        Parameters
        ----------
        v : int
            value
        _width : np.array
            width vector

        Returns
        -------
        float
            route width
        """
        #quo=np.floor(np.divide(v, self._width.shape[0]))
        mod=np.mod(v, _width.shape[0])
        #if not isinstance(mod, int):
        #    print(v, _width, mod)
        return _width[round(np.round(mod))]

    def _get_route_layer(self, v, _layer):
        """
        Get metal layer

        Parameters
        ----------
        v : int
            index value
        _layer : list
            layer list

        Returns
        -------
        [str, str]
            route layer
        """
        mod=np.mod(v, len(_layer))
        return _layer[mod]

    #def get_xwidth(self): return self.xwidth
    #def get_ywidth(self): return self.ywidth
    #def get_viamap(self): return self.viamap

    def get_route_width_xy(self, xy):
        """
        Get metal width vector

        Parameters
        ----------
        xy : np.array([int, int])
            coordinate

        Returns
        -------
        np.array([float, float])
            route width vector ([xgrid, ygrid])
        """
        return np.array([self._get_route_width(xy[0], self.xwidth),
                         self._get_route_width(xy[1], self.ywidth)])

    def get_route_xlayer_xy(self, xy):
        """
        Get route layer in xgrid direction (vertical)

        Parameters
        ----------
        xy : np.array([int, int])
            coordinate

        Returns
        -------
        [str, str]
            route layer information
        """
        return self._get_route_layer(xy[0], self.xlayer)

    def get_route_ylayer_xy(self, xy):
        """
        Get route layer in ygrid direction (horizontal)

        Parameters
        ----------
        xy : np.array([int, int])
            coordinate

        Returns
        -------
        [str, str]
            route layer information
        """
        return self._get_route_layer(xy[1], self.ylayer)

    def get_vianame(self, xy):
        """
        Get the name of via on xy

        Parameters
        ----------
        xy : np.array([int, int])
            coordinate

        Returns
        -------
        str
            cellname of via
        """
        mod = np.array([np.mod(xy[0], self.xgrid.shape[0]), np.mod(xy[1], self.ygrid.shape[0])])
        for vianame, viacoord in self.viamap.items():
            if viacoord.ndim==1:
                if np.array_equal(mod, viacoord): return vianame
            else:
                for v in viacoord:
                    if np.array_equal(mod,v):
                        return vianame

    def display(self):
        """
        Display RouteGrid information
        """
        display_str="  " + self.name + " width:" + str(np.around(self.width, decimals=self.max_resolution)) \
                    + " height:" + str(np.around(self.height, decimals=self.max_resolution)) \
                    + " xgrid:" + str(np.around(self.xgrid, decimals=self.max_resolution)) \
                    + " ygrid:" + str(np.around(self.ygrid, decimals=self.max_resolution)) \
                    + " xwidth:" + str(np.around(self.xwidth, decimals=self.max_resolution)) \
                    + " ywidth:" + str(np.around(self.ywidth, decimals=self.max_resolution)) \
                    + " xlayer:" + str(self.xlayer) \
                    + " ylayer:" + str(self.ylayer) \
                    + " viamap:{"
        for vm_name, vm in self.viamap.items():
            display_str+=vm_name + ": " + str(vm.tolist()) + " "
        display_str+="}"
        print(display_str)

    def export_dict(self):
        """
        Export grid information in dict format

        Returns
        -------
        dict
            Grid information
        """
        export_dict=GridObject.export_dict(self)
        export_dict['xwidth'] = np.around(self.xwidth, decimals=self.max_resolution).tolist()
        export_dict['ywidth'] = np.around(self.ywidth, decimals=self.max_resolution).tolist()
        export_dict['xlayer'] = self.xlayer
        export_dict['ylayer'] = self.ylayer
        export_dict['viamap'] = dict()
        for vn, v in self.viamap.items():
            export_dict['viamap'][vn]=[]
            for _v in v:
                export_dict['viamap'][vn].append(_v.tolist())
        return export_dict

    def update_viamap(self, viamap):
        """use laygo2.GridObject.RouteGrid.viamap instead"""
        self.viamap=viamap

if __name__ == '__main__':
    lgrid=GridObject()
    print('LayoutGrid test')
    lgrid.xgrid = np.array([0.2, 0.4, 0.6])
    lgrid.ygrid = np.array([0, 0.4, 0.9, 1.2, 2, 3])
    lgrid.width = 1.2
    lgrid.height = 3.2
    phycoord = np.array([[-0.2, -0.2], [0, 2], [4,2.2], [0.5, 1.5], [1.3, 3.6], [8,2.3]])
    print("  xgrid:" + str(lgrid.xgrid) + " width:" + str(lgrid.width))
    print("  ygrid:" + str(lgrid.ygrid) + " height:" + str(lgrid.height))
    print('physical grid to abstract grid')
    print("  input:"+str(phycoord.tolist()))
    abscoord=lgrid.get_absgrid_coord_xy(phycoord)
    print("  output:"+str(abscoord.tolist()))
    print('abstract grid to physical grid')
    phycoord=lgrid.get_phygrid_xy(abscoord).tolist()
    print("  output:"+str(phycoord))
'''
