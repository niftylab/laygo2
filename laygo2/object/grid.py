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
Module containing functions and objects related to abstract coordinate system and coordinate conversion.
"""

__author__ = "Jaeduk Han"
__maintainer__ = "Jaeduk Han"
__status__ = "Prototype"

import numpy as np
import laygo2.object

# import laygo2.util.conversion as cv


# Internal functions.
def _extend_index_dim(input_index, new_index, new_index_max):
    """
    A helper function to be used for the multi-dimensional circular array
    indexing of CircularMappingArray. It extends the dimension of the input
    array (input_index) that contains indexing information, with the
    additional indexing variable (new_index) provided. The new_index_max
    variable is specified in case of the new_index does not contain the
    maximum index information (perhaps when an open-end slice is given for
    the new_index).

    Parameters
    ----------
    input_index : iterable
        The iterable object to be extended.
    new_index : iterable
        The iterable object to be added to input_ind to extend its dimension.
    new_index_max : iterable
        The maximum index of new_index when its upper boundary is not
        provided.

    Returns
    -------
    Iterable
        The extended index.

    Example
    --------
    >>> import laygo2
    >>> # 0-dim to 1-dim
    >>> laygo2.object.grid._extend_index_dim(None, [3, 4, 6], None)
    [(3,), (4,), (6,)]
    >>> # 0-dim to 1-dim (with slicing input)
    >>> laygo2.object.grid._extend_index_dim(None, slice(3, 8, 2), None)
    [(3,), (5,), (7,)]
    >>> # 0-dim to 1-dim, with maximum index given
    >>> laygo2.object.grid._extend_index_dim(None, slice(3, None, 2), 8)
    [(3,), (4,), (6,)]
    >>> # 1-dim to 2-dim
    >>> laygo2.object.grid._extend_index_dim([(3,), (4,), (6,)], [1, 2], None)
    [[(3, 1), (3, 2)], [(4, 1), (4, 2)], [(6, 1), (6, 2)]]
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
    """Convert slice to a list.

    Parameters
    ----------
    slice_obj : slice
        The slice object to be converted.
    start_def : int, optional
        The default starting index if the slice object has no lower boundary.
    stop_def : int, optional
        The default stopping index if the slice object has no upper boundary.
    step_def : int, optional
        The default stepping index if the slice object has no step specified.

    Example
    --------
    >>> import laygo2
    >>> laygo2.object.grid._conv_slice_to_list(slice(0, 10, 2))               
    [0, 2, 4, 6, 8]
    """
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

    Parameters
    ----------
    bbox : numpy.ndarray
        The bounding box to be converted.

    Example
    --------
    >>> import laygo2
    >>> import numpy as np
    >>> laygo2.object.grid._conv_bbox_to_array(np.array([[0, 0], [1, 2]])) 
    array([[[0, 0], [1, 0]],
           [[0, 1], [1, 1]],
           [[0, 2], [1, 2]]])
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
    Convert a bbox object to a 1-d list.

    Parameters
    ----------
    bbox : numpy.ndarray
        The bounding box to be converted.

    Example
    --------
    >>> import laygo2
    >>> import numpy as np
    >>> laygo2.object.grid._conv_bbox_to_list(np.array([[0, 0], [1, 2]])) 
    [[0, 0], [1, 0], [0, 1], [1, 1], [0, 2], [1, 2]]
    """
    array = list()
    for r in range(bbox[0, 1], bbox[1, 1] + 1):
        for c in range(bbox[0, 0], bbox[1, 0] + 1):
            array.append([c, r])
    return array


# Internal classes
class CircularMapping:
    """
    Basic circular mapping class (index number expands infinitely).

    Notes
    -----
    **(Korean)** 기본 순환 맵핑 (인덱싱 넘버가 무한히 확장) 클래스.
    """

    _elements = None
    """list: Array consisting of the elements of circular mapping.

    Example
    --------
    >>> from laygo2.object.grid import CircularMapping
    >>> elements = [0, 35, 85, 130, 180] 
    >>> cm = CircularMapping(elements) 
    >>> cm.elements
    [0, 35, 85, 130, 180]

    .. image:: ../assets/img/object_grid_CircularMapping_elements.png
           :height: 250 

    Notes
    -----
    **(Korean)**
    순환 맵핑의 구성 요소로 이루어진 배열. 
    """

    dtype = np.int
    """type: Data type of the circular mapping.

    Example
    --------
    >>> from laygo2.object.grid import CircularMapping
    >>> elements = [0, 35, 85, 130, 180] 
    >>> cm = CircularMapping(elements) 
    >>> cm.dtype
    int

    .. image:: ../assets/img/object_grid_CircularMapping_dtype.png
           :height: 250

    Notes
    -----
    **(Korean)**
    순환 맵핑의 데이터 유형.
    """

    def get_elements(self):
        """numpy.ndarray: getter of elements."""
        return self._elements

    def set_elements(self, value):
        """numpy.ndarray: setter of elements."""
        self._elements = np.asarray(value, dtype=self.dtype)

    elements = property(get_elements, set_elements)
    """numpy.ndarray: the array that contains the physical coordinates of the grid."""

    @property
    def shape(self):
        """numpy.ndarray: The shape of circular mapping.

        Example
        --------
        >>> from laygo2.object.grid import CircularMapping
        >>> elements = [0, 35, 85, 130, 180]
        >>> cm = CircularMapping(elements)
        >>> cm.shape
        array([5])

        .. image:: ../assets/img/object_grid_CircularMapping_shape.png
           :height: 250

        Notes
        -----
        **(Korean)**
        순환 맵핑의 shape.
        """
        return np.array(self.elements.shape)

    def __init__(self, elements=np.array([0]), dtype=np.int):
        """
        Constructor function of CircularMapping class.

        Parameters
        ----------
        elements : list
            elements.
        dtype : type
            data type of elements.

        Example
        --------
        >>> from laygo2.object.grid import CircularMapping
        >>> elements = [0, 35, 85, 130, 180]
        >>> cm = CircularMapping(elements)
        >>> cm.shape
        [5]
        >>> cm[5]
        35
        >>> cm[0:10]
        [0, 35, 85, 130, 0, 35, 85, 130, 0, 35]

        .. image:: ../assets/img/object_grid_CircularMapping_init.png
           :height: 250

        Notes
        -----
        **(Korean)** CircularMapping 클래스의 생성자함수
            파라미터
            - elements(list): 구성 요소
            - dtype(type): 구성 요소의 datatype
        """
        self.dtype = dtype
        self.elements = np.asarray(elements, dtype=dtype)

    # indexing and slicing
    def __getitem__(self, pos):
        """Element access function of circular mapping."""
        if isinstance(pos, (int, np.integer)):
            return self.elements[pos % self.shape[0]]
        elif isinstance(pos, slice):
            return self.__getitem__(
                pos=_conv_slice_to_list(slice_obj=pos, stop_def=self.shape[0])
            )
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
        """Iteration function of circular mapping."""
        return self.elements.__iter__()

    def __next__(self):
        """Next element access function of circular mapping."""
        # Check if numpy.ndarray implements __next__()
        return self.elements.__next__()

    # Informative functions
    def __str__(self):
        return self.summarize()

    def summarize(self):
        """Return the summary of the object information."""
        return (
            self.__repr__() + " "
            "class: "
            + self.__class__.__name__
            + ", "
            + "elements: "
            + str(self.elements)
        )


class CircularMappingArray(CircularMapping):
    """
    Multi-dimensional circular mapping class (index number expands infinitely).

    Notes
    -----
    **(Korean)** 다차원 순환맵핑(인덱싱 넘버가 무한히 확장) 클래스.
    """

    def __getitem__(self, pos):
        """
        Element access function.

        Parameters
        ----------
        pos : int
            index number being accessed.

        Returns
        -------
        numpy.ndarray 

        Example
        --------
        >>> from laygo2.object.grid import CircularMappingArray
        >>> elements = [[0, 0], [35, 0], [85, 0], [130, 0]]
        >>> cm = CircularMappingArray(elements = elements)
        >>> cm[1, :]
        array([[35, 0]])
        >>> cm[3, 0]
        130

        .. image:: ../assets/img/object_grid_CircularMappingArray_getitem.png
           :height: 250

        Notes
        -----
        **(Korean)** 순환 맵핑의 요소 접근함수
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
            # pos is mapped to multiple indices. (possible Example include ([0:5, 3], [[1,2,3], 3], ...).
            # Create a list containing the indices to iterate over, and return a numpy.ndarray containing items
            # corresponding to the indices in the list.
            # When the indices don't specify the lower boundary (e.g., [:5]), it iterates from 0.
            # When the indices don't specify the upper boundary (e.g., [3:]), it iterates to the maximum index defined.
            idx = None
            for i, p in enumerate(pos):  # iterate over input indices (x and y).
                idx = _extend_index_dim(idx, p, self.shape[i])
            idx = np.asarray(idx)
            # iterate and generate the list to return
            item = np.empty(
                idx.shape[:-1], dtype=self.dtype
            )  # -1 because the tuples in idx are flatten.
            for i, _null in np.ndenumerate(item):
                item[i] = self.__getitem__(pos=tuple(idx[i]))
            return np.asarray(item)


class _AbsToPhyGridConverter:
    """
    A class that converts abstract coordinates into physical coordinates. 
    Conversely, conditional operation converts physical coordinates into 
    abstract coordinates.

    Notes
    -----
    **(Korean)** 추상 좌표를 물리 좌표로 변환하는 클래스, 조건부연산은 역변환을 
    수행한다.
    """

    master = None
    """laygo2.Grid or laygo2.OneDimGrid: Coordinate system to which 
    _AbsToPhyGridConverter object belongs.

    Example
    --------
    >>> from laygo2.object.grid import OneDimGrid, Grid
    >>> g1_x = OneDimGrid(name='xg', scope=[0, 180], elements=[0, 35, 85, 130, 180]) 
    >>> g1_y = OneDimGrid(name='yg', scope=[0, 30], elements=[0]) 
    >>> g2   = Grid(name='g', vgrid=g1_x, hgrid=g1_y) 
    >>> print(g1_x.abs2phy) 
    <laygo2.object.grid._AbsToPhyGridConverter object> 
    >>> print(g2.xy) 
    <laygo2.object.grid._AbsToPhyGridConverter object>
    >>> print(g1_x.abs2phy.master) 
    <laygo2.object.grid.OneDimGrid object>
    >>> print(g2.xy.master) 
    <laygo2.object.grid.Grid object>

    Notes
    -----
    **(Korean)** _AbsToPhyGridConverter 객체가 속한 좌표계.
    """

    # Constructor
    def __init__(self, master):
        """Constructor function of _AbsToPhyGridConverter class."""
        self.master = master

    # Access functions.
    def __call__(self, pos):
        """
        Convert abstract coordinates of the master grid into corresponding 
        physical coordinates.

        Parameters
        ----------
        pos : int
            abstract coordinates.

        Returns
        -------
        int or numpy.ndarray
            physical coordinates.

        Example
        --------
        >>> from laygo2.object.grid import OneDimGrid, Grid
        >>> g1_x = OneDimGrid(name='xg', scope=[0, 180], elements=[0, 35, 85, 130, 180])
        >>> g1_y = OneDimGrid(name='yg', scope=[0,30], elements=[0])
        >>> g2   = Grid(name='g', vgrid=g1_x, hgrid=g1_y)
        >>> g1_x.abs2phy(0)
        0
        >>> g2.xy(0,0)
        [0, 0]

        .. image:: ../assets/img/object_grid_AbsToPhyGridConverter_call.png
           :height: 250

        Notes
        -----
        **(Korean)** 추상 좌표를 master 좌표계에서 대응되는 물리 좌표로 변환.
        """
        return self.__getitem__(pos)

    def __getitem__(self, pos):
        """
        Convert abstract coordinates of the master grid into corresponding 
        physical coordinates.

        Parameters
        ----------
        pos : int
            abstract coordinates.

        Returns
        -------
        int or numpy.ndarray
            physical coordinates.

        Example
        --------
        >>> from laygo2.object.grid import OneDimGrid, Grid
        >>> g1_x = OneDimGrid(name='xg', scope=[0, 180], elements=[0, 35, 85, 130, 180])
        >>> g1_y = OneDimGrid(name='yg', scope=[0, 30], elements=[0])
        >>> g2   = Grid(name='g', vgrid=g1_x, hgrid=g1_y)
        >>> g1_x.abs2phy(0)
        0
        >>> g2.xy(0,0)
        [0, 0]

        .. image:: ../assets/img/object_grid_AbsToPhyGridConverter_getitem.png
           :height: 250

        Notes
        -----
        **(Korean)** 추상 좌표를 master 좌표계에서 대응되는 물리 좌표로 변환.
        """
        if (self.master.__class__.__name__ == "OneDimGrid") or (
            issubclass(self.master.__class__, OneDimGrid)
        ):
            return self._getitem_1d(pos)
        if (self.master.__class__.__name__ == "Grid") or (
            issubclass(self.master.__class__, Grid)
        ):
            return self._getitem_2d(pos)
        else:
            return None

    def _getitem_1d(self, pos):
        """An internal function of __getitem__() for 1-d grids."""
        # Check if pos has multiple elements.
        if isinstance(pos, slice):
            return self._getitem_1d(
                _conv_slice_to_list(slice_obj=pos, stop_def=self.master.shape[0])
            )
        elif isinstance(pos, np.ndarray):
            return self._getitem_1d(pos.tolist())
        elif isinstance(pos, list):
            return np.array([self._getitem_1d(p) for p in pos])
        elif pos is None:
            raise TypeError(
                "_AbsToPhyConverter._getitem_1d does not accept None as its input."
            )
        else:
            # pos is a single element. Compute quotient and modulo for grid extension.
            quo = 0
            mod = int(round(pos))
            if pos >= self.master.shape[0]:
                mod = int(round(pos % self.master.shape[0]))
                quo = int(round((pos - mod) / self.master.shape[0]))
            elif pos < 0:
                mod = int(round(pos % self.master.shape[0]))
                quo = int(round((pos - mod)) / self.master.shape[0])
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
        if (not isinstance(x, np.ndarray)) and (
            not isinstance(y, np.ndarray)
        ):  # x and y are scalars.
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
        """
        Convert physical coordinates into abstract coordinates of the master grid 
        satisfying conditional operations.

        Parameters
        ----------
        other : int
            physical coordinates.

        Returns
        -------
        int or numpy.ndarray
            abstract coordinates.

        Example
        --------
        >>> from laygo2.object.grid import OneDimGrid, Grid
        >>> g1_x = OneDimGrid(name='xg', scope=[0, 180], elements=[0, 35, 85, 130, 180])
        >>> g1_y = OneDimGrid(name='yg', scope=[0, 30], elements=[0])
        >>> g2   = Grid(name='g', vgrid=g1_x, hgrid=g1_y)
        >>> g1_x.abs2phy == 35
        1
        >>> g2.xy == [35, 35]
        [1, None]

        .. image:: ../assets/img/object_grid_AbsToPhyGridConverter_eq.png
           :height: 250

        Notes
        -----
        **(Korean)** 물리 좌표를 master 좌표계에서 조건부 연산을 만족하는 추상 좌표로 변환.
        """
        return self.master.phy2abs(pos=other)

    def __lt__(self, other):
        """
        Convert physical coordinates into abstract coordinates of the master grid 
        satisfying conditional operations.

        Parameters
        ----------
        other : int
            physical coordinates.

        Returns
        -------
        int or numpy.ndarray
            abstract coordinates.

        Example
        --------
        >>> from laygo2.object.grid import OneDimGrid, Grid
        >>> g1_x = OneDimGrid(name='xg', scope=[0, 180], elements=[0, 35, 85, 130, 180])
        >>> g1_y = OneDimGrid(name='yg', scope=[0, 30], elements=[0])
        >>> g2   = Grid(name='g', vgrid=g1_x, hgrid=g1_y)
        >>> g1_x.abs2phy < 35
        0
        >>> g2.xy < [35, 35]
        [0, 1]

        .. image:: ../assets/img/object_grid_AbsToPhyGridConverter_lt.png
           :height: 250

        Notes
        -----
        **(Korean)** 물리 좌표를 master 좌표계에서 조건부 연산을 만족하는 추상 좌표로 변환.
        """
        if (self.master.__class__.__name__ == "OneDimGrid") or (
            issubclass(self.master.__class__, OneDimGrid)
        ):
            return self._lt_1d(other)
        if (self.master.__class__.__name__ == "Grid") or (
            issubclass(self.master.__class__, Grid)
        ):
            return self._lt_2d(other)
        else:
            return None

    @staticmethod
    def _phy2abs_operator(other, elements, width, shape, op):
        def phy2abs(x):
            if x > 0:
                quo_coarce = 0 + x // width
                msb_sub = 1
            else:
                quo_coarce = 0 + x // width
                msb_sub = 0

            remain = x % width  # positive
            msb = quo_coarce * shape - 1
            for i, e in np.ndenumerate(elements):
                # print("e: %d r:%d, m:%d, i:%d off:%d phy:%d " %(e, remain, msb + i[0], i[0], lsb_offset, quo_coarce*width + e   ))
                # print(comp( e , remain ))

                if comp(e, remain) == True:  # find maximum less then remain , e < r
                    pass
                else:  # when it is False, latest true index
                    return msb + i[0] + lsb_offset

            return msb + shape + lsb_offset

        if op == "<":  ## max lesser
            comp = lambda e, r: e < r
            lsb_offset = 0

        elif op == "<=":  ## eq or max lesser eq
            comp = lambda e, r: e <= r
            lsb_offset = 0

        elif op == ">":  ## min greater
            comp = lambda e, r: e <= r
            lsb_offset = 1

        elif op == ">=":  ## eq or min greater
            comp = lambda e, r: e < r
            lsb_offset = 1

        if isinstance(other, (int, np.integer)):
            return phy2abs(other)
        else:
            list_return = []
            for o in other:
                list_return.append(phy2abs(o))
            return np.array(list_return)

    def _lt_1d(self, other):
        return self._phy2abs_operator(
            other,
            self.master.elements,
            self.master.width,
            self.master.elements.shape[0],
            "<",
        )

    def _lt_2d(self, other):
        if isinstance(other[0], (int, np.integer)):
            return np.array([self.master.x < other[0], self.master.y < other[1]])
        else:
            return np.array([self._lt_2d(o) for o in other])

    def __le__(self, other):
        """
        Convert physical coordinates into abstract coordinates of the master grid 
        satisfying conditional operations.

        Parameters
        ----------
        other : int
            physical coordinates.

        Returns
        -------
        int or numpy.ndarray
            abstract coordinates.

        Example
        --------
        >>> from laygo2.object.grid import OneDimGrid, Grid
        >>> g1_x = OneDimGrid(name='xgrid', scope=[0, 180], elements=[0, 35, 85, 130, 180])
        >>> g1_y = OneDimGrid(name='ygrid', scope=[0,30], elements=[0])
        >>> g2   = Grid(name='test', vgrid=g1_x, hgrid=g1_y)
        >>> g1_x.abs2phy <= 35
        1
        >>> g2.xy <= [35,35]
        [1,1]

        .. image:: ../assets/img/object_grid_AbsToPhyGridConverter_le.png
           :height: 250

        Notes
        -----
        **(Korean)** 물리 좌표를 master 좌표계에서 조건부 연산을 만족하는 추상 좌표로 변환.
        """
        if (self.master.__class__.__name__ == "OneDimGrid") or (
            issubclass(self.master.__class__, OneDimGrid)
        ):
            return self._le_1d(other=other)
        if (self.master.__class__.__name__ == "Grid") or (
            issubclass(self.master.__class__, Grid)
        ):
            return self._le_2d(other=other)

    def _le_1d(self, other):
        return self._phy2abs_operator(
            other,
            self.master.elements,
            self.master.width,
            self.master.elements.shape[0],
            "<=",
        )

    def _le_2d(self, other):
        if isinstance(other[0], (int, np.integer)):
            return np.array([self.master.x <= other[0], self.master.y <= other[1]])
        else:
            return np.array([self._le_2d(o) for o in other])

    def __gt__(self, other):
        """
        Convert physical coordinates into abstract coordinates of the master grid 
        satisfying conditional operations.

        Parameters
        ----------
        other : int
            physical coordinates.

        Returns
        -------
        int or numpy.ndarray
            abstract coordinates.

        Example
        --------
        >>> from laygo2.object.grid import OneDimGrid, Grid
        >>> g1_x = OneDimGrid(name='xgrid', scope=[0, 180], elements=[0, 35, 85, 130, 180])
        >>> g1_y = OneDimGrid(name='ygrid', scope=[0, 30], elements=[0])
        >>> g2   = Grid(name='test', vgrid=g1_x, hgrid=g1_y)
        >>> g1_x.abs2phy > 35
        2
        >>> g2.xy > [35, 35]
        [2, 2]

        .. image:: ../assets/img/object_grid_AbsToPhyGridConverter_gt.png
           :height: 250

        Notes
        -----
        **(Korean)** 물리 좌표를 master 좌표계에서 조건부 연산을 만족하는 추상 좌표로 변환.
        """
        if (self.master.__class__.__name__ == "OneDimGrid") or (
            issubclass(self.master.__class__, OneDimGrid)
        ):
            return self._gt_1d(other=other)
        if (self.master.__class__.__name__ == "Grid") or (
            issubclass(self.master.__class__, Grid)
        ):
            return self._gt_2d(other=other)

    def _gt_1d(self, other):
        return self._phy2abs_operator(
            other,
            self.master.elements,
            self.master.width,
            self.master.elements.shape[0],
            ">",
        )

    def _gt_2d(self, other):
        if isinstance(other[0], (int, np.integer)):
            return np.array([self.master.x > other[0], self.master.y > other[1]])
        else:
            return np.array([self._gt_2d(o) for o in other])

    def __ge__(self, other):
        """
        Convert physical coordinates into abstract coordinates of the master grid 
        satisfying conditional operations.

        Parameters
        ----------
        other : int
            physical coordinates.

        Returns
        -------
        int or numpy.ndarray
            abstract coordinates.

        Example
        --------
        >>> from laygo2.object.grid import OneDimGrid, Grid
        >>> g1_x = OneDimGrid(name='xgrid', scope=[0, 180], elements=[0, 35, 85, 130, 180])
        >>> g1_y = OneDimGrid(name='ygrid', scope=[0,30], elements=[0])
        >>> g2   = Grid(name='test', vgrid=g1_x, hgrid=g1_y)
        >>> g1_x.abs2phy >= 35
        1
        >>> g2.xy >= [35, 35]
        [1, 2]

        .. image:: ../assets/img/object_grid_AbsToPhyGridConverter_ge.png
           :height: 250

        Notes
        -----
        **(Korean)** 물리 좌표를 master 좌표계에서 조건부 연산을 만족하는 추상 좌표로 변환.
        """
        if (self.master.__class__.__name__ == "OneDimGrid") or (
            issubclass(self.master.__class__, OneDimGrid)
        ):
            return self._ge_1d(other=other)
        if (self.master.__class__.__name__ == "Grid") or (
            issubclass(self.master.__class__, Grid)
        ):
            return self._ge_2d(other=other)

    def _ge_1d(self, other):
        return self._phy2abs_operator(
            other,
            self.master.elements,
            self.master.width,
            self.master.elements.shape[0],
            ">=",
        )

    def _ge_2d(self, other):
        if isinstance(other[0], (int, np.integer)):
            return np.array([self.master.x >= other[0], self.master.y >= other[1]])
        else:
            return np.array([self._ge_2d(o) for o in other])


class _PhyToAbsGridConverter:
    """
    A class that converts physical coordinates into abstract coordinates.
    Conversely, conditional operations convert abstract coordinates into 
    physical coordinates. 

    Notes
    -----
    **(Korean)**
    물리 좌표를 추상 좌표로 변환하는 클래스, 조건부연산은 반대로 추상 좌표를 
    물리 좌표로 변환한다.

    """

    master = None
    """laygo2.Grid or laygo2.OneDimGrid: Coordinate system to which 
    _PhyToAbsGridConverter object belongs.

    Example
    --------
    >>> from laygo2.object.grid import OneDimGrid, Grid
    >>> g1_x = OneDimGrid(name='xg', scope=[0, 180], elements=[0, 35, 85, 130, 180]) 
    >>> g1_y = OneDimGrid(name='yg', scope=[0, 30], elements=[0]) 
    >>> g2   = Grid(name='g', vgrid=g1_x, hgrid=g1_y) 
    >>> print(g1_x.phy2abs) 
    <laygo2.object.grid._PhyToAbsGridConverter object> 
    >>> print(g2.mn) 
    <laygo2.object.grid._PhyToAbsGridConverter object>
    >>> print(g1_x.phy2abs.master) 
    <laygo2.object.grid.OneDimGrid object>
    >>> print(g2.mn.master) 
    <laygo2.object.grid.Grid object>

    Notes
    -----
    **(Korean)** _PhyToAbsGridConverter 객체가 속한 좌표계.
    """

    # Constructor
    def __init__(self, master):
        """Constructor function of _PhyToAbsGridConverter class."""
        self.master = master

    # Access functions.
    def __call__(self, pos):
        """
        Convert physical coordinates into the corresponding abstract coordinates of 
        the master grid.

        Parameters
        ----------
        pos : int
            physical coordinates.

        Returns
        -------
        int or numpy.ndarray
            abstract coordinates.

        Example
        --------
        >>> from laygo2.object.grid import OneDimGrid, Grid
        >>> g1_x = OneDimGrid(name='xgrid', scope=[0, 180], elements=[0, 35, 85, 130, 180])
        >>> g1_y = OneDimGrid(name='ygrid', scope=[0, 30], elements=[0])
        >>> g2   = Grid(name='test', vgrid=g1_x, hgrid=g1_y)
        >>> g1_x.phy2abs(35)
        1
        >>> g2.mn([[35, 35]])
        [1, None]

        .. image:: ../assets/img/object_grid_PhyToAbsGridConverter_call.png
           :height: 250

        Notes
        -----
        **(Korean)** 물리 좌표를 master 좌표계에서 대응되는 추상 좌표로 변환.
        """
        return self.__getitem__(pos)

    def __getitem__(self, pos):
        """
        Convert physical coordinates into the corresponding abstract coordinates of the master grid.

        Parameters
        ----------
        pos : int
            physical coordinates.

        Returns
        -------
        int or numpy.ndarray
            abstract coordinates.

        Example
        --------
        >>> from laygo2.object.grid import OneDimGrid, Grid
        >>> g1_x = OneDimGrid(name='xg', scope=[0, 180], elements=[0, 35, 85, 130, 180])
        >>> g1_y = OneDimGrid(name='yg', scope=[0, 30], elements=[0])
        >>> g2   = Grid(name='g', vgrid=g1_x, hgrid=g1_y)
        >>> g1_x.phy2abs(35)
        1
        >>> g2.mn( [[35, 35]])
        [1, None]

        .. image:: ../assets/img/object_grid_PhyToAbsGridConverter_getItem.png
           :height: 250

        Notes
        -----
        **(Korean)** 물리 좌표를 master 좌표계에서 대응되는 추상 좌표로 변환.
        """
        if (self.master.__class__.__name__ == "OneDimGrid") or (
            issubclass(self.master.__class__, OneDimGrid)
        ):
            return self._getitem_1d(pos)
        if (self.master.__class__.__name__ == "Grid") or (
            issubclass(self.master.__class__, Grid)
        ):
            return self._getitem_2d(pos)
        else:
            return None

    def _getitem_1d(self, pos):
        """An internal function of __getitem__() for 1-d grids."""
        # Check if pos has multiple elements.
        if isinstance(pos, OneDimGrid):
            return self._getitem_1d(pos=pos.elements)
        elif isinstance(pos, slice):
            return self._getitem_1d(
                _conv_slice_to_list(slice_obj=pos, stop_def=self.master.shape[0])
            )
        elif isinstance(pos, np.ndarray):
            return self._getitem_1d(pos.tolist())
        elif isinstance(pos, list):
            return np.array([self._getitem_1d(p) for p in pos])
        elif pos is None:
            raise TypeError(
                "_AbsToPhyConverter._getitem_1d does not accept None as its input."
            )
        else:
            # pos is a single element.
            for i, e in np.ndenumerate(self.master.elements):
                if (pos - e) % self.master.width == 0:
                    return (
                        int(round((pos - e) / self.master.width))
                        * self.master.elements.shape[0]
                        + i[0]
                    )
            return None  # no matched coordinate

    def _getitem_2d(self, pos):
        """An internal function of __getitem__() for 2-d grid."""
        # If pos contains multiple coordinates (or objects), convert recursively.
        if isinstance(pos, list):
            if isinstance(
                pos[0], (int, np.integer)
            ):  # It's actually a single coordinate.
                return self[pos[0], pos[1]]
            else:
                return [self[p] for p in pos]
        elif isinstance(pos, np.ndarray):
            if isinstance(
                pos[0], (int, np.integer)
            ):  # It's actually a single coordinate.
                return np.array(self[pos[0], pos[1]])
            else:
                return np.array([self[p] for p in pos])
        # If pos contains only one physical object, convert its bounding box to abstract coordinates
        if (pos.__class__.__name__ == "PhysicalObject") or (
            issubclass(pos.__class__, laygo2.object.PhysicalObject)
        ):
            return self.bbox(pos)
        # If pos contains only one coordinate, convert it to abstract grid.
        m = self.master.x == pos[0]
        n = self.master.y == pos[1]
        # refactor the following code to avoid the use of double for-loops and list comprehensions.
        if (not isinstance(m, np.ndarray)) and (
            not isinstance(n, np.ndarray)
        ):  # x and y are scalars.
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
        """
        Convert abstract coordinates into physical coordinates satisfying 
        conditional operations in the master grid.

        Parameters
        ----------
        other : int
            abstract coordinates.

        Returns
        -------
        int or numpy.ndarray
            physical coordinates.

        Example
        --------
        >>> from laygo2.object.grid import OneDimGrid, Grid
        >>> g1_x = OneDimGrid(name='xg', scope=[0, 180], elements=[0, 35, 85, 130, 180])
        >>> g1_y = OneDimGrid(name='yg', scope=[0, 30], elements=[0])
        >>> g2   = Grid(name='g', vgrid=g1_x, hgrid=g1_y)
        >>> g1_x.phy2abs == 1
        35
        >>> g2.mn == [1, 1]
        [35, 30]

        .. image:: ../assets/img/object_grid_PhyToAbsGridConverter_eq.png
           :height: 250

        Notes
        -----
        **(Korean)** 추상 좌표를 master 좌표계에서 조건부 연산을 만족하는 물리 좌표로 변환.
        """
        return self.master.abs2phy(pos=other)

    """
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
    """

    def __lt__(self, other):
        """
        Convert abstract coordinates into physical coordinates satisfying 
        conditional operations in the master grid.

        Parameters
        ----------
        other : int
            abstract coordinates.

        Returns
        -------
        int or numpy.ndarray
            physical coordinates.

        Example
        --------
        >>> from laygo2.object.grid import OneDimGrid, Grid
        >>> g1_x = OneDimGrid(name='xgrid', scope=[0, 180], elements=[0, 35, 85, 130, 180])
        >>> g1_y = OneDimGrid(name='ygrid', scope=[0, 30], elements=[0])
        >>> g2   = Grid(name='test', vgrid=g1_x, hgrid=g1_y)
        >>> g1_x.phy2abs < 1
        0
        >>> g2.mn < [1, 1]
        [0, 0]

        .. image:: ../assets/img/object_grid_PhyToAbsGridConverter_lt.png
           :height: 250

        Notes
        -----
        **(Korean)** 추상 좌표를 master 좌표계에서 조건부 연산을 만족하는 물리 좌표로 변환.
        """
        if (self.master.__class__.__name__ == "OneDimGrid") or (
            issubclass(self.master.__class__, OneDimGrid)
        ):
            return self._lt_1d(other=other)
        if (self.master.__class__.__name__ == "Grid") or (
            issubclass(self.master.__class__, Grid)
        ):
            return self._lt_2d(other=other)

    def _lt_1d(self, other):
        if isinstance(other, (int, np.integer)):
            return self.master.abs2phy.__getitem__(pos=other - 1)
        return np.array([self._lt_1d(o) for o in other])

    def _lt_2d(self, other):
        if isinstance(other[0], (int, np.integer)):
            return self.master.abs2phy.__getitem__(pos=(other[0] - 1, other[1] - 1))
        return np.array([self._lt_2d(o) for o in other])

    def __le__(self, other):
        """
        Convert abstract coordinates into physical coordinates satisfying 
        conditional operations in the master grid.

        Parameters
        ----------
        other : int
            abstract coordinates.

        Returns
        -------
        int or numpy.ndarray
            physical coordinates.

        Example
        --------
        >>> from laygo2.object.grid import OneDimGrid, Grid
        >>> g1_x = OneDimGrid(name='xg', scope=[0, 180], elements=[0, 35, 85, 130, 180])
        >>> g1_y = OneDimGrid(name='yg', scope=[0, 30], elements=[0])
        >>> g2   = Grid(name='g', vgrid=g1_x, hgrid=g1_y)
        >>> g1_x.phy2abs <= 1
        35
        >>> g2.mn <= [1, 1]
        [35, 30]

        .. image:: ../assets/img/object_grid_PhyToAbsGridConverter_le.png
           :height: 250

        Notes
        -----
        **(Korean)** 추상 좌표를 master 좌표계에서 조건부 연산을 만족하는 물리 좌표로 변환.
        """
        return self.master.abs2phy(pos=other)

    def __gt__(self, other):
        """
        Convert abstract coordinates into physical coordinates satisfying 
        conditional operations in the master grid.

        Parameters
        ----------
        other : int
            abstract coordinates.

        Returns
        -------
        int or numpy.ndarray
            physical coordinates.

        Example
        --------
        >>> from laygo2.object.grid import OneDimGrid, Grid
        >>> g1_x = OneDimGrid(name='xgrid', scope=[0, 180], elements=[0, 35, 85, 130, 180])
        >>> g1_y = OneDimGrid(name='ygrid', scope=[0, 30], elements=[0])
        >>> g2   = Grid(name='test', vgrid=g1_x, hgrid=g1_y)
        >>> g1_x.phy2abs > 1
        85
        >>> g2.mn > [1, 1]
        [85, 60]

        .. image:: ../assets/img/object_grid_PhyToAbsGridConverter_gt.png
           :height: 250

        Notes
        -----
        **(Korean)** 추상 좌표를 master 좌표계에서 조건부 연산을 만족하는 물리 좌표로 변환.
        """
        if (self.master.__class__.__name__ == "OneDimGrid") or (
            issubclass(self.master.__class__, OneDimGrid)
        ):
            return self._gt_1d(other)
        if (self.master.__class__.__name__ == "Grid") or (
            issubclass(self.master.__class__, Grid)
        ):
            return self._gt_2d(other)
        else:
            return None

    def _gt_1d(self, other):
        if isinstance(other, (int, np.integer)):
            return self.master.abs2phy.__getitem__(pos=other + 1)
        return np.array([self._gt_1d(o) for o in other])

    def _gt_2d(self, other):
        if isinstance(other[0], (int, np.integer)):
            return self.master.abs2phy.__getitem__(pos=(other[0] + 1, other[1] + 1))
        return np.array([self._gt_2d(o) for o in other])

    def __ge__(self, other):
        """
        Convert abstract coordinates into physical coordinates satisfying 
        conditional operations in the master grid.

        Parameters
        ----------
        other : int
            abstract coordinates.

        Returns
        -------
        int or numpy.ndarray
            physical coordinates.

        Example
        --------
        >>> from laygo2.object.grid import OneDimGrid, Grid
        >>> g1_x = OneDimGrid(name='xgrid', scope=[0, 180], elements=[0, 35, 85, 130, 180])
        >>> g1_y = OneDimGrid(name='ygrid', scope=[0, 30], elements=[0])
        >>> g2   = Grid(name='test', vgrid=g1_x, hgrid=g1_y)
        >>> g1_x.phy2abs >= 1
        35
        >>> g2.mn >=[1, 1]
        [35, 30]

        .. image:: ../assets/img/object_grid_PhyToAbsGridConverter_ge.png
           :height: 250

        Notes
        -----
        **(Korean)** 추상 좌표를 master 좌표계에서 조건부 연산을 만족하는 물리 좌표로 변환.
        """
        return self.master.abs2phy.__getitem__(pos=other)

    def bbox(self, obj):
        """
        Convert the bounding box of the object into the abstract coordinates 
        of the master grid.

        Parameters
        ----------
        obj : laygo2.physical
            object having physical coordinate.

        Returns
        -------
        numy.ndarray
            abstract coordinates.

        Example
        --------
        >>> from laygo2.object.grid import OneDimGrid, Grid
        >>> g1_x    = OneDimGrid(name='xg', scope=[0, 100], elements=[10, 20, 40, 50, 60])
        >>> g1_y    = OneDimGrid(name='yg', scope=[0, 100], elements=[10, 20, 40, 50, 60])
        >>> g2      = Grid(name='g', vgrid=g1_x, hgrid=g1_y)
        >>> phy2abs = _PhyToAbsGridConverter(master=g2)
        >>> rect0 = physical.Rect(xy=[[0, 0], [100, 100]], layer=['M1', 'drawing'], netname='net0’)
        >>> phy2abs.bbox(rect0)
        [[0, 0] , [4, 4]]
        >>> g2.mn.bbox(rect0)
        [[0, 0] , [4, 4]]

        .. image:: ../assets/img/object_grid_PhyToAbsGridConverter_bbox.png
           :height: 250

        Notes
        -----
        **(Korean)** 객체의 bounding box를 master 좌표계의 추상 좌표로 변환.
        _AbsToPhyGridConverter 객체의 >=, <=를 사용하므로 추상면적이 작아질수있다.
        """
        if (obj.__class__.__name__ == "PhysicalObject") or (
            issubclass(obj.__class__, laygo2.object.PhysicalObject)
        ):
            obj = obj.bbox

        # phy -> abs
        mn0 = self.master.xy >= obj[0]  ## ge than lower left
        mn1 = self.master.xy <= obj[1]  ## le than upper right\

        return np.array([mn0, mn1])

    def bottom_left(self, obj):
        """
        Convert an object's physical corner coordinates into abstract coordinates 
        of the master grid.

        Parameters
        ----------
        obj : laygo2.physical
            object having physical coordinate.

        Returns
        -------
        numy.ndarray
            abstract coordinates.

        Example
        --------
        >>> from laygo2.object.grid import OneDimGrid, Grid
        >>> g1_x    = OneDimGrid(name='xgrid', scope=[0, 100], elements=[10, 20, 40, 50, 60])
        >>> g1_y    = OneDimGrid(name='ygrid', scope=[0, 100], elements=[10, 20, 40, 50, 60])
        >>> g2      = Grid(name='test', vgrid=g1_x, hgrid=g1_y)
        >>> phy2abs = _PhyToAbsGridConverter(master=g2)
        >>> rect0 = physical.Rect(xy=[[0, 0], [100, 100]], layer=['M1', 'drawing'], netname='net0’)
        >>> phy2abs.bottom_left(rect0)
        [0, 0]
        >>> g2.mn.bottom_left(rect0)
        [0, 0]

        .. image:: ../assets/img/object_grid_PhyToAbsGridConverter_bottom_left.png
           :height: 250

        Notes
        -----
        **(Korean)** 객체의 물리 코너 좌표를 master 좌표계의 추상 좌표로 변환.
        """
        if (obj.__class__.__name__ == "PhysicalObject") or (
            issubclass(obj.__class__, laygo2.object.PhysicalObject)
        ):
            return self.bottom_left(obj.bbox)
        else:
            _i = self.bbox(obj)
            return _i[0]

    def bottom_right(self, obj):
        """
        Convert an object's physical corner coordinates into abstract coordinates 
        of the master grid.

        Parameters
        ----------
        obj : laygo2.physical
            object having physical coordinate.

        Returns
        -------
        numy.ndarray
            abstract coordinates.

        Example
        --------
        >>> from laygo2.object.grid import OneDimGrid, Grid
        >>> g1_x    = OneDimGrid(name='xg', scope=[0, 100], elements=[10, 20, 40, 50, 60])
        >>> g1_y    = OneDimGrid(name='yg', scope=[0, 100], elements=[10, 20, 40, 50, 60])
        >>> g2      = Grid(name='g', vgrid=g1_x, hgrid=g1_y)
        >>> phy2abs = _PhyToAbsGridConverter(master=g2)
        >>> rect0 = physical.Rect(xy=[[0, 0], [100, 100]], layer=['M1', 'drawing'], netname='net0’)
        >>> phy2abs.bottom_right(rect0)
        [4, 0]
        >>> g2.mn.bottom_right(rect0)
        [4, 0]

        .. image:: ../assets/img/object_grid_PhyToAbsGridConverter_bottom_right.png
           :height: 250

        Notes
        -----
        **(Korean)** 객체의 물리 코너 좌표를 master 좌표계의 추상 좌표로 변환.
        """
        if (obj.__class__.__name__ == "PhysicalObject") or (
            issubclass(obj.__class__, laygo2.object.PhysicalObject)
        ):
            return self.bottom_right(obj.bbox)
        else:
            _i = self.bbox(obj)
            return np.array([_i[1, 0], _i[0, 1]])

    def top_left(self, obj):
        """
        Convert an object's physical corner coordinates into abstract coordinates 
        of the master grid.

        Parameters
        ----------
        obj : laygo2.physical
            object having physical coordinate.

        Returns
        -------
        numy.ndarray
            abstract coordinates.

        Example
        --------
        >>> from laygo2.object.grid import OneDimGrid, Grid
        >>> g1_x    = OneDimGrid(name='xg', scope=[0, 100], elements=[10, 20, 40, 50, 60])
        >>> g1_y    = OneDimGrid(name='yg', scope=[0, 100], elements=[10, 20, 40, 50, 60])
        >>> g2      = Grid(name='g', vgrid=g1_x, hgrid=g1_y)
        >>> phy2abs = _PhyToAbsGridConverter(master=g2)
        >>> rect0 = physical.Rect(xy=[[0, 0], [100, 100]], layer=['M1', 'drawing'], netname='net0’)
        >>> phy2abs.top_left(rect0)
        [0, 4]
        >>> g2.mn.top_left(rect0)
        [0, 4]

        .. image:: ../assets/img/object_grid_PhyToAbsGridConverter_top_left.png
           :height: 250

        Notes
        -----
        **(Korean)** 객체의 물리 코너 좌표를 master 좌표계의 추상 좌표로 변환.
        """
        if (obj.__class__.__name__ == "PhysicalObject") or (
            issubclass(obj.__class__, laygo2.object.PhysicalObject)
        ):
            return self.top_left(obj.bbox)
        else:
            _i = self.bbox(obj)
            return np.array([_i[0, 0], _i[1, 1]])

    def top_right(self, obj):
        """
        Convert an object's physical corner coordinates into abstract 
        coordinates of the master grid.

        Parameters
        ----------
        obj : laygo2.physical
            object having physical coordinate.

        Returns
        -------
        numy.ndarray
            abstract coordinates.

        Example
        --------
        >>> from laygo2.object.grid import OneDimGrid, Grid
        >>> g1_x    = OneDimGrid(name='xg', scope=[0, 100], elements=[10, 20, 40, 50, 60])
        >>> g1_y    = OneDimGrid(name='yg', scope=[0, 100], elements=[10, 20, 40, 50, 60])
        >>> g2      = Grid(name='g', vgrid=g1_x, hgrid=g1_y)
        >>> phy2abs = _PhyToAbsGridConverter(master=g2)
        >>> rect0 = physical.Rect(xy=[[0, 0], [100, 100]], layer=['M1', 'drawing'], netname='net0’)
        >>> phy2abs.top_right(rect0)
        [4, 4]
        >>> g2.mn.top_right(rect0)
        [4, 4]

        .. image:: ../assets/img/object_grid_PhyToAbsGridConverter_top_right.png
           :height: 250

        Notes
        -----
        **(Korean)** 객체의 물리 코너 좌표를 master 좌표계의 추상 좌표로 변환.
        """
        if (obj.__class__.__name__ == "PhysicalObject") or (
            issubclass(obj.__class__, laygo2.object.PhysicalObject)
        ):
            return self.top_right(obj.bbox)
        else:
            _i = self.bbox(obj)
            return _i[1]

    def width(self, obj):
        """Return the width of an object on this grid."""
        if (obj.__class__.__name__ == "PhysicalObject") or (
            issubclass(obj.__class__, laygo2.object.PhysicalObject)
        ):
            return self.width(obj.bbox)
        else:
            _i = self.bbox(obj)
            return abs(_i[1, 0] - _i[0, 0])

    def height(self, obj):
        """Return the height of an object on this grid."""
        if (obj.__class__.__name__ == "PhysicalObject") or (
            issubclass(obj.__class__, laygo2.object.PhysicalObject)
        ):
            return self.height(obj.bbox)
        else:
            _i = self.bbox(obj)
            return abs(_i[1, 1] - _i[0, 1])

    def height_vec(self, obj):
        """numpy.ndarray(dtype=int): Return np.array([0, height])."""
        return np.array([0, self.height(obj)])

    def width_vec(self, obj):
        """numpy.ndarray(dtype=int): Return np.array([width, 0])."""
        return np.array([self.width(obj), 0])

    def size(self, obj):
        """
        Convert an object's size ([width, height]) into abstract coordinates 
        of the master grid.

        Parameters
        ----------
        obj : laygo2.physical
            object having physical coordinate.

        Returns
        -------
        numpy.ndarray
            abstract coordinates.

        Example
        --------
        >>> from laygo2.object.grid import OneDimGrid, Grid
        >>> g1_x    = OneDimGrid(name='xg', scope=[0, 100], elements=[10, 20, 40, 50, 60])
        >>> g1_y    = OneDimGrid(name='yg', scope=[0, 100], elements=[10, 20, 40, 50, 60])
        >>> g2      = Grid(name='g', vgrid=g1_x, hgrid=g1_y)
        >>> phy2abs = _PhyToAbsGridConverter(master=g2)
        >>> rect0 = physical.Rect(xy=[[0, 0], [100, 100]], layer=['M1', 'drawing'], netname='net0’)
        >>> phy2abs.size(rect0)
        [4, 4]
        >>> g2.mn.size(rect0)
        [4, 4]

        .. image:: ../assets/img/object_grid_PhyToAbsGridConverter_size.png
           :height: 250

        Notes
        -----
        **(Korean)** 객체의 크기([width, height])를 master 좌표계의 추상 좌표로 변환.
        """
        return np.array([self.width(obj), self.height(obj)])

    def crossing(self, *args):
        """
        Convert the physical intersections of objects into abstract coordinates 
        of the master grid.

        Parameters
        ----------
        args : laygo2.Physical
            physical object having bbox.

        Returns
        -------
        numpy.ndarray(int, int)
            abstract points.

        Example
        --------
        >>> from laygo2.object.grid import OneDimGrid, Grid
        >>> g1_x = OneDimGrid(name='xg', scope=[0, 10], elements=[0])
        >>> g1_y = OneDimGrid(name='yg’, scope=[0, 120], elements=[0, 20, 40, 80, 100, 120])
        >>> g2   = Grid(name='g', vgrid = g1_x, hgrid = g1_y )
        >>> phy2abs = _PhyToAbsGridConverter(master=g2)
        >>> rect0= physical.Rect(xy=[[0, 0], [60, 90]])
        >>> rect1= physical.Rect(xy=[[30, 30], [120, 120]])
        >>> phy2abs.crossing(rect0, rect1)
        [3, 2]
        >>> g2.mn.crossing(rect0, rect1)
        [3, 2]

        .. image:: ../assets/img/object_grid_PhyToAbsGridConverter_crossing.png
           :height: 250

        Notes
        -----
        **(Korean)** 객체들의 물리적 교차점을 master 좌표계의 추상 좌표로 변환.
        """
        return self.overlap(*args, type="point")

    def overlap(self, *args, type="bbox"):
        """
        Convert the overlapping area of objects into abstract coordinates of 
        the master grid and return in a format specified in type.

        A bounding box is returned if type='bbox'
        
        All coordinates in the overlapped region are returned in a 
        two-dimensional array if type='array'
        
        An one-dimensional list is returned if type='list'.

        Parameters
        ----------
        args : laygo2.Physical
            physical object having bbox.

        Returns
        -------
        numpy.ndarray
            bbox abstract coordinates.

        Example
        --------
        >>> from laygo2.object.grid import OneDimGrid, Grid
        >>> g1_x = OneDimGrid(name='xg', scope=[0, 10], elements=[0])
        >>> g1_y = OneDimGrid(name='yg', scope=[0, 120], elements=[0, 20, 40, 80, 100, 120])
        >>> g2   = Grid(name='g', vgrid = g1_x, hgrid = g1_y )
        >>> phy2abs = _PhyToAbsGridConverter(master=g2)
        >>> rect0= physical.Rect(xy=[[0, 0], [60, 90]])
        >>> rect1= physical.Rect(xy=[[30, 30], [120, 120]])
        >>> phy2abs.overlap(rect0, rect1)
        [[3, 2], [6,4]]
        >>> g2.mn.overlap(rect0, rect1)
        [[3, 2], [6,4]]

        .. image:: ../assets/img/object_grid_PhyToAbsGridConverter_overlap.png
           :height: 250

        Notes
        -----
        **(Korean)** 객체들의 겹치는 면적을 master 좌표계의 추상 좌표로 변환 후 
        type에 따른 형태로 반환.
        
        'bbox'인 경우, bounding box로 반환.
        
        'array' 인 경우 모든 교점을 2차원 array로 반환. 
        
        'list' 인경우 모든 교점을 1차원 list로 변환.
        """
        _ib = None
        for _obj in args:
            if _ib is None:
                _ib = self.bbox(_obj)  ## shaped
            else:
                _b = self.bbox(_obj)
                _x = np.sort(np.array([_b[:, 0], _ib[:, 0]]), axis=None)
                _y = np.sort(np.array([_b[:, 1], _ib[:, 1]]), axis=None)
                _ib = np.array([[_x[1], _y[1]], [_x[2], _y[2]]])
        if type == "bbox":
            return _ib
        elif type == "point":
            return _ib[0]
        elif type == "list":
            return _conv_bbox_to_list(_ib)
        elif type == "array":
            return _conv_bbox_to_array(_ib)
        else:
            raise ValueError(
                "overlap() should receive a valid value for its type (bbox, point, array, ...)"
            )

    def union(self, *args):
        """
        Convert the bounding box containing all objects into abstract coordinates 
        of the master grid.

        Parameters
        ----------
        args : laygo2.Physical
            physical object having bbox.

        Returns
        -------
        numpy.ndarray
            bbox abstract coordinates.

        Example
        --------
        >>> from laygo2.object.grid import OneDimGrid, Grid
        >>> g1_x = OneDimGrid(name='xgrid', scope=[0, 10], elements=[0])
        >>> g1_y = OneDimGrid(name='ygrid', scope=[0, 120], elements=[0, 20, 40, 80, 100, 120 )
        >>> g2   = Grid(name="test", vgrid = g1_x, hgrid = g1_y )
        >>> rect0= physical.Rect(xy=[[0, 0], [60, 90]])
        >>> rect1= physical.Rect(xy=[[30, 30], [120, 120]])
        >>> g2.mn.union(rect0, rect1)
        [[0, 0], [12,7]]

        .. image:: ../assets/img/object_grid_PhyToAbsGridConverter_union.png
           :height: 250

        Notes
        -----
        **(Korean)** 객체들을 모두 포함하는 bounding box를 master 좌표계의 
        추상 좌표로 변환.
        """
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

    def center(self, obj):
        """
        Convert an object's physical center coordinates into abstract coordinates 
        of the master grid.

        Parameters
        ----------
        obj : laygo2.physical
            object having physical coordinate.

        Returns
        -------
        numy.ndarray
            abstract coordinates.

        Example
        --------
        >>> from laygo2.object.grid import OneDimGrid, Grid
        >>> g1_x    = OneDimGrid(name='xg', scope=[0, 100], elements=[10, 20, 40, 50, 60])
        >>> g1_y    = OneDimGrid(name='yg', scope=[0, 100], elements=[10, 20, 40, 50, 60])
        >>> g2      = Grid(name='g', vgrid=g1_x, hgrid=g1_y)
        >>> phy2abs = _PhyToAbsGridConverter(master=g2)
        >>> rect0 = physical.Rect(xy=[[0, 0], [100, 100]], layer=['M1', 'drawing'], netname='net0’)
        >>> phy2abs.center(rect0)
        [3, 3]
        >>> g2.mn.center(rect0)
        [3, 3]

        .. image:: ../assets/img/object_grid_PhyToAbsGridConverter_center.png
           :height: 250

        Notes
        -----
        **(Korean)** 객체의 물리 중앙 좌표를 master 좌표계의 추상 좌표로 변환.
        """
        mn0 = self.master.xy >= obj.center
        mn1 = self.master.xy <= obj.center

        point_list = [
            self.master.xy[mn0],
            self.master.xy[mn1],
            self.master.xy[mn0[0], mn1[1]],
            self.master.xy[mn1[0], mn0[1]],
        ]  # 4 physical points near the center coordinate.
        dist_list = []
        idx = 0
        for point in point_list:
            dist_list.append(
                [idx, np.linalg.norm(point - obj.center)]
            )  # Calculate Euclidean distances.
            idx += 1
        dist_sorted = sorted(
            dist_list, key=lambda distance: distance[1]
        )  # Sort distances in ascending order.
        return self.master.mn(
            point_list[dist_sorted[0][0]]
        )  # Convert the closest point to abstract coordinate and then return.


class OneDimGrid(CircularMapping):
    """
    Class implementing one-dimensional abstract coordinates.

    Notes
    -----
    **(Korean)**
    1차원 추상좌표를 구현하는 클래스.

    """

    # Member variables and properties
    name = None
    """str: Coordinate system name.

    Example
    --------
    >>> from laygo2.object.grid import OneDimGrid
    >>> g1_x = OneDimGrid(name='xgrid', scope=[0, 180], elements=[0, 35, 85, 130, 50]) 
    >>> g1_x.name
    "xgrid"

    .. image:: ../assets/img/object_grid_OneDimGrid_name.png
           :height: 250

    Notes
    -----
    **(Korean)** 좌표계 이름.
    """

    range = None
    """str: Region in which the coordinate system is defined Coordinates in 
    the defined region are repeatedly expanded.

    Example
    --------
    >>> from laygo2.object.grid import OneDimGrid
    >>> g1_x = OneDimGrid(name='xgrid', scope=[0, 180], elements=[0, 35, 85, 130, 50]) 
    >>> g1_x.range
    [0, 180]
    
    .. image:: ../assets/img/object_grid_OneDimGrid_range.png
           :height: 250

    Notes
    -----
    **(Korean)** 좌표계가 정의된 영역. 정의된 영역의 좌표들이 반복되는 형태로 확장된다.
    """

    phy2abs = None
    """self.phy2abs (laygo2._PhyToAbsGridConverter): Object that converts physical 
    coordinates into abstract coordinates.

    Example
    --------
    >>> from laygo2.object.grid import OneDimGrid
    >>> g1_x  = OneDimGrid(name='xgrid', scope=[0, 180], elements=[0, 35, 85, 130, 50]) 
    >>> g1_x.phy2abs
    <_PhyToAbsGridConverter object>

    Notes
    -----
    **(Korean)** 물리 좌표에서 추상 좌표로 변환연산을 해주는 객체. 
    """

    abs2phy = None
    """self.abs2phy (laygo2._AbsToPhyGridConverter): Object that converts abstract 
    coordinates into physical coordinates.

    Example
    --------
    >>> from laygo2.object.grid import OneDimGrid
    >>> g1_x = OneDimGrid(name='xgrid', scope=[0, 180], elements=[0, 35, 85, 130, 50]) 
    >>> g1_x.abs2phy
    <_AbsToPhyGridConverter object>

    Notes
    -----
    **(Korean)** 추상 좌표에서 물리 좌표로 변환연산을 해주는 객체. 
    """

    @property
    def width(self):
        """int: The size of the region in which the coordinate system is defined.

        Example
        --------
        >>> from laygo2.object.grid import OneDimGrid
        >>> g1_x = OneDimGrid(name='xgrid', scope=[0, 180], elements=[0, 35, 85, 130, 50])
        >>> g1_x.width
        180

        .. image:: ../assets/img/object_grid_OneDimGrid_width.png
           :height: 250

        Notes
        -----
        **(Korean)** 좌표계가 정의된 영역의 크기.
        """
        return abs(self.range[1] - self.range[0])

    # Constructor
    def __init__(self, name, scope, elements=np.array([0])):
        """
        Constructor function of OneDimGrid class.

        Parameters
        ----------
        name : str
        scope : numpy.ndarray
            scope of one-dimensional coordinate system
        elements: numpy.ndarray
            members of one-dimensional coordinate system

        Returns
        -------
        laygo2.OneDimGrid



        Example
        --------
        >>> from laygo2.object.grid import OneDimGrid
        >>> g1_x = OneDimGrid(name='xgrid', scope=[0, 180], elements=[0, 35, 85, 130, 50])
        >>> print(g1_x)
        OneDimGrid object name: xgrid, class: OneDimGrid, scope: [0, 180], elements: [0, 35, 85, 130  50]

        .. image:: ../assets/img/object_grid_OneDimGrid_init.png
           :height: 250

        Notes
        -----
        **(Korean)** OneDimGrid 클래스의 생성자함수.
        """
        self.name = name
        self.range = np.asarray(scope)
        self.phy2abs = _PhyToAbsGridConverter(master=self)
        self.abs2phy = _AbsToPhyGridConverter(master=self)
        CircularMapping.__init__(self, elements=elements)
        # self.elements = np.asarray(elements)  # commented out because asarray does not woke well with Object arrays.

    # Indexing and slicing functions
    def __getitem__(self, pos):
        """Return the physical coordinate corresponding to the abstract coordinate pos."""
        return self.abs2phy([pos])

    # Comparison operators
    def __eq__(self, other):
        """Return the abstract grid coordinate that matches to other."""
        return self.abs2phy.__eq__(other)

    def __lt__(self, other):
        """Return the abstract grid coordinate that is the largest but less than other."""
        return self.abs2phy.__lt__(other)

    def __le__(self, other):
        """Return the index of the grid coordinate that is the largest but less than or equal to other."""
        return self.abs2phy.__le__(other)

    def __gt__(self, other):
        """Return the abstract grid coordinate that is the smallest but greater than other."""
        return self.abs2phy.__gt__(other)

    def __ge__(self, other):
        """Return the index of the grid coordinate that is the smallest but greater than or equal to other."""
        return self.abs2phy.__ge__(other)

    # Informative functions
    def __str__(self):
        """Return the string representation of the object."""
        return self.summarize()

    def summarize(self):
        """Return the summary of the object information."""
        return (
            self.__repr__() + " "
            "name: "
            + self.name
            + ", "
            + "class: "
            + self.__class__.__name__
            + ", "
            + "scope: "
            + str(self.range)
            + ", "
            + "elements: "
            + str(self.elements)
        )

    # I/O functions
    def export_to_dict(self):
        """
        Return dict object containing grid information.

        Parameters
        ----------
        None

        Returns
        -------
        dict

        Example
        --------
        >>> from laygo2.object.grid import OneDimGrid
        >>> g1_x = OneDimGrid(name='xgrid', scope=[0, 180], elements=[0, 35, 85, 130, 50])
        >>> g1_x.export_to_dict()
        {'scope': [0, 180], 'elements': [0, 35, 85, 130, 50]}

        .. image:: ../assets/img/object_grid_OneDimGrid_export_to_dict.png
           :height: 250

        Notes
        -----
        **(Korean)** 그리드의 정보를 담은 dict객체 반환.
        """
        export_dict = {
            "scope": self.range.tolist(),
            "elements": self.elements.tolist(),
        }
        return export_dict


class Grid:
    """
    A base class having conversion operators and the mapping information (element) 
    between two-dimensional physical coordinates and abstract coordinates.

    Notes
    -----
    **(Korean)** 2차원 물리좌표와 추상좌표간 mapping 정보(element) 를 갖고 있으며 
    해당 element를 활용하는 좌표 연산자를 가지고 있는 기본 클래스.
    """

    name = None
    """str: the name of the grid."""

    _xy = None
    """List[OneDimGrid]: the list contains the 1d-grid objects for x and y axes."""

    @property
    def elements(self):
        """numpy.ndarray: Two-dimensional element of a coordinate system.
            x elements, y elements

        Example
        --------
        >>> from laygo2.object.grid import OneDimGrid, Grid
        >>> g1_x = OneDimGrid(name='xgrid', scope=[0, 100], elements=[0, 10, 20, 40, 50])
        >>> g1_y = OneDimGrid(name='ygrid', scope=[0, 100], elements=[10, 20, 40, 50, 60])
        >>> g2   = Grid(name="test", vgrid = g1_x, hgrid = g1_y )
        >>> g2.elements
        [ array([0, 10, 20, 40, 50]), array( [10, 20, 40, 50, 60] ) ]

        .. image:: ../assets/img/object_grid_Grid_Elements.png
           :height: 250

        Notes
        -----
        **(Korean)** 좌표계 2차원 element.
        """
        return [self._xy[0].elements, self._xy[1].elements]

    phy2abs = None
    """PhyToAbsGridConverter(master=self)"""

    abs2phy = None
    """AbsToPhyGridConverter(master=self)"""

    @property
    def xy(self):
        """_AbsToPhyGridConverter: Two-dimensional 
        _AbsToPhyConverter of a coordinate system.

        Example
        --------
        >>> from laygo2.object.grid import OneDimGrid, Grid
        >>> g1_x = OneDimGrid(name='xgrid', scope=[0, 100], elements=[0, 10, 20, 40, 50])
        >>> g1_y = OneDimGrid(name='ygrid', scope=[0, 100], elements=[10, 20, 40, 50, 60])
        >>> g2   = Grid(name="test", vgrid = g1_x, hgrid = g1_y )
        >>> g2.xy[10,10]
        [200 210]
        >>> g2.xy([10, 10])
        [200 210]
        >>> g2.xy < [10,10]
        [0,-1]
        >>> g2.xy <= [10,10]
        [1,0]
        >>> g2.xy > [10,10]
        [2,1]
        >>> g2.xy >= [10,10]
        [1,0]

        Notes
        -----
        **(Korean)** 2차원 _AbsToPhyConverter.
        """
        return self.abs2phy

    @property
    def x(self):
        """_AbsToPhyGridConverter: One-dimensional _AbsToPhyGridConverter 
            of the x-coordinate system.

        Example
        --------
        >>> from laygo2.object.grid import OneDimGrid, Grid
        >>> g1_x = OneDimGrid(name='xgrid', scope=[0, 100], elements=[0, 10, 20, 40, 50 ])
        >>> g1_y = OneDimGrid(name='ygrid', scope=[0, 100], elements=[10, 20, 40, 50, 60 ])
        >>> g2   = Grid(name="test", vgrid = g1_x, hgrid = g1_y )
        >>> g2.x[10]
        200
        >>> g2.x <  10
        [0]
        >>> g2.x <= 10
        [1]
        >>> g2.x >  10
        [2]
        >>> g2.x >= 10
        [1]

        Notes
        -----
        **(Korean)**
        x좌표계 1차원 _AbsToPhyGridConverter.
        """
        return self._xy[0].abs2phy

    @property
    def y(self):
        """_AbsToPhyGridConverter: One-dimensional _AbsToPhyGridConverter 
        of the y-coordinate system.

        Example
        --------
        >>> from laygo2.object.grid import OneDimGrid, Grid
        >>> g1_x = OneDimGrid(name='xgrid', scope=[0, 100], elements=[0, 10, 20, 40, 50 ])
        >>> g1_y = OneDimGrid(name='ygrid', scope=[0, 100], elements=[10, 20, 40, 50, 60 ])
        >>> g2   = Grid(name="test", vgrid = g1_x, hgrid = g1_y )
        >>> g2.y[10]
        210
        >>> g2.y <  10
        [-1]
        >>> g2.y <= 10
        [0]
        >>> g2.y >  10
        [1]
        >>> g2.y >= 10
        [0]

        Notes
        -----
        **(Korean)** y좌표계 1차원 _AbsToPhyGridConverter.
        """
        return self._xy[1].abs2phy

    @property
    def v(self):
        """OneDimGrid: OneDimGrid of the x-coordinate system (=self.x).

        Example
        --------
        >>> from laygo2.object.grid import OneDimGrid, Grid
        >>> g1_x = OneDimGrid(name='xgrid', scope=[0, 100], elements=[0, 10, 20, 40, 50 ])
        >>> g1_y = OneDimGrid(name='ygrid', scope=[0, 100], elements=[10, 20, 40, 50, 60 ])
        >>> g2   = Grid(name="test", vgrid = g1_x, hgrid = g1_y )
        >>> g2.v
        g1_x

        Notes
        -----
        **(Korean)** x좌표계 OneDimGrid.
        """
        return self.x

    @property
    def h(self):
        """OneDimGrid: OneDimGrid of the y-coordinate system (=self.y).

        Example
        --------
        >>> from laygo2.object.grid import OneDimGrid, Grid
        >>> g1_x = OneDimGrid(name='xgrid', scope=[0, 100], elements=[0, 10, 20, 40, 50 ])
        >>> g1_y = OneDimGrid(name='ygrid', scope=[0, 100], elements=[10, 20, 40, 50, 60 ])
        >>> g2   = Grid(name="test", vgrid = g1_x, hgrid = g1_y )
        >>> g2.h
        g1_y

        Notes
        -----
        **(Korean)** y좌표계 OneDimGrid.
        """
        return self.y

    @property
    def mn(self):
        """laygo2._PhyToAbsGridConverter: Two-dimensional _PhyToAbsConverter of 
        a coordinate system.

        Example
        --------
        >>> from laygo2.object.grid import OneDimGrid, Grid
        >>> g1_x = OneDimGrid(name='xgrid', scope=[0, 100], elements=[0, 10, 20, 40, 50 ])
        >>> g1_y = OneDimGrid(name='ygrid', scope=[0, 100], elements=[10, 20, 40, 50, 60 ])
        >>> g2   = Grid(name="test", vgrid = g1_x, hgrid = g1_y )
        >>> g2.mn[40,40]
        [3, 2]
        >>> g2.mn([40, 40])
        [3, 2]
        >>> g2.mn <  [40,40]
        [750, 760]
        >>> g2.mn <= [40,40]
        [800, 810]
        >>> g2.mn >  [40,40]
        [810, 820]
        >>> g2.mn >= [40,40]
        [800, 810]

        Notes
        -----
        **(Korean)**
        좌표계 2차원 _PhyToAbsConverter.
        """
        return self.phy2abs

    @property
    def m(self):
        """_PhyToAbsGridConverter: One-dimensional _PhyToAbsConverter of 
        the x-coordinate system.

        Example
        --------
        >>> from laygo2.object.grid import OneDimGrid, Grid
        >>> g1_x = OneDimGrid(name='xgrid', scope=[0, 100], elements=[0, 10, 20, 40, 50 ])
        >>> g1_y = OneDimGrid(name='ygrid', scope=[0, 100], elements=[10, 20, 40, 50, 60 ])
        >>> g2   = Grid(name="test", vgrid = g1_x, hgrid = g1_y )
        >>> g2.n[40]
        2
        >>> g2.n(40)
        2
        >>> g2.n <  40
        760
        >>> g2.n <= 40
        810
        >>> g2.n >  40
        820
        >>> g2.n >= 40
        810

        Notes
        -----
        **(Korean)** x좌표계 1차원 _PhyToAbsConverter.
        """
        return self._xy[0].phy2abs

    @property
    def n(self):
        """_PhyToAbsGridConverter: One-dimensional _PhyToAbsConverter of
         the y-coordinate system.

        Example
        --------
        >>> from laygo2.object.grid import OneDimGrid, Grid
        >>> g1_x = OneDimGrid(name='xgrid', scope=[0, 100], elements=[0, 10, 20, 40, 50 ])
        >>> g1_y = OneDimGrid(name='ygrid', scope=[0, 100], elements=[10, 20, 40, 50, 60 ])
        >>> g2   = Grid(name="test", vgrid = g1_x, hgrid = g1_y )
        >>> g2.n[40]
        2
        >>> g2.n(40)
        2
        >>> g2.n <  40
        760
        >>> g2.n <= 40
        810
        >>> g2.n >  40
        820
        >>> g2.n >= 40
        810

        Notes
        -----
        **(Korean)** y좌표계 1차원 _PhyToAbsConverter.
        """
        return self._xy[1].phy2abs

    @property
    def shape(self):
        """numpy.ndarray: Two-dimensional element length in a coordinate system.
            length of x-axis elements, length of y-axis elements

        Example
        --------
        >>> from laygo2.object.grid import OneDimGrid, Grid
        >>> g1_x = OneDimGrid(name='xgrid', scope=[0, 100], elements=[0, 10, 20, 40, 50 ])
        >>> g1_y = OneDimGrid(name='ygrid', scope=[0, 100], elements=[10, 20, 40, 50, 60 ])
        >>> g2   = Grid(name="test", vgrid = g1_x, hgrid = g1_y )
        >>> g2.shape
        [5, 5]

        Notes
        -----
        **(Korean)** 좌표게 2차원 element의 길이.
        """
        return np.hstack([self._xy[0].shape, self._xy[1].shape])

    def get_range(self):
        return np.transpose(np.vstack((self._xy[0].range, self._xy[1].range)))

    def set_range(self, value):
        self._xy[0].range = np.transpose(value)[0]
        self._xy[1].range = np.transpose(value)[1]

    range = property(get_range, set_range)
    """numpy.ndarray: Region in which the coordinate system is defined.
        bbox of the respective Grid

    Example
    --------
    >>> from laygo2.object.grid import OneDimGrid, Grid
    >>> g1_x = OneDimGrid(name='xgrid', scope=[0, 100], elements=[0, 10, 20, 40, 50 ]) 
    >>> g1_y = OneDimGrid(name='ygrid', scope=[0, 100], elements=[10, 20, 40, 50, 60 ]) 
    >>> g2   = Grid(name="test", vgrid = g1_x, hgrid = g1_y )
    >>> g2.range 
    [ [0, 0], [100, 100 ]]

    Notes
    -----
    **(Korean)** 좌표계가 정의된 영역.
    """

    @property
    def width(self):
        """numpy.int32: Width of the region in which the coordinate system is defined.
            x scope

        Example
        --------
        >>> from laygo2.object.grid import OneDimGrid, Grid
        >>> g1_x = OneDimGrid(name='xgrid', scope=[0, 100], elements=[0, 10, 20, 40, 50 ])
        >>> g1_y = OneDimGrid(name='ygrid', scope=[0, 100], elements=[10, 20, 40, 50, 60 ])
        >>> g2   = Grid(name="test", vgrid = g1_x, hgrid = g1_y )
        >>> g2.width
        100

        Notes
        -----
        **(Korean)** 좌표계가 정의된 영역의 폭.
        """
        return self._xy[0].width

    @property
    def height(self):
        """numpy.int32: Height of the region in which the coordinate system is defined.

        Example
        --------
        >>> from laygo2.object.grid import OneDimGrid, Grid
        >>> g1_x = OneDimGrid(name='xgrid', scope=[0, 100], elements=[0, 10, 20, 40, 50 ])
        >>> g1_y = OneDimGrid(name='ygrid', scope=[0, 100], elements=[10, 20, 40, 50, 60 ])
        >>> g2   = Grid(name="test", vgrid = g1_x, hgrid = g1_y )
        >>> g2.height
        100

        Notes
        -----
        **(Korean)** 좌표계가 정의된 영역의 높이.
        """
        return self._xy[1].width

    @property
    def height_vec(self):
        """numpy.ndarray: Return the height vector [0, h].

        Example
        --------
        >>> from laygo2.object.grid import OneDimGrid, Grid
        >>> g1_x = OneDimGrid(name='xgrid', scope=[0, 100], elements=[0, 10, 20, 40, 50 ])
        >>> g1_y = OneDimGrid(name='ygrid', scope=[0, 100], elements=[10, 20, 40, 50, 60 ])
        >>> g2   = Grid(name="test", vgrid = g1_x, hgrid = g1_y )
        >>> g2.height_vec
        [0, 100]

        Notes
        -----
        **(Korean)** height를 list로 반환.
        """
        return np.array([0, self.height])

    @property
    def width_vec(self):
        """numpy.ndarray: Return width as a list.
            length of the respective axis and zero

        Example
        --------
        >>> from laygo2.object.grid import OneDimGrid, Grid
        >>> g1_x = OneDimGrid(name='xgrid', scope=[0, 100], elements=[0, 10, 20, 40, 50 ])
        >>> g1_y = OneDimGrid(name='ygrid', scope=[0, 100], elements=[10, 20, 40, 50, 60 ])
        >>> g2   = Grid(name="test", vgrid = g1_x, hgrid = g1_y )
        >>> g2.width_vec
        [100, 0]

        Notes
        -----
        **(Korean)** width를 list로 반환.
        """
        return np.array([self.width, 0])

    def __init__(self, name, vgrid, hgrid):
        """
        Constructor function of Grid class.

        Parameters
        ----------
        name : str
        vgrid : laygo2.object.grid.OndDimGrid
            OneDimGrid object of the x-coordinate system
        hgrid : laygo2.object.grid.OndDimGrid
            OneDimGrid object of the y-coordinate system

        Returns
        -------
        laygo2.object.grid.Grid

        Example
        --------
        >>> from laygo2.object.grid import OneDimGrid, Grid
        >>> g1_x = OneDimGrid(name='xgrid', scope=[0, 100], elements=[0, 10, 20, 40, 50 ])
        >>> g1_y = OneDimGrid(name='ygrid', scope=[0, 100], elements=[10, 20, 40, 50, 60 ])
        >>> g2   = Grid(name="test", vgrid = g1_x, hgrid = g1_y )
        >>> print(g2)
        <laygo2.object.grid.Grid object> name: test, class: Grid, scope: [[0, 0], [100, 100]], elements: [array([ 0, 10, 20, 40, 50]), array([10, 20, 40, 50, 60])

        Notes
        -----
        **(Korean)** Grid 클래스의 생성자함수.
        파라미터
            - name(str): 이름
            - vgrid(laygo2.OneDimGrid): x좌표계 OneDimGrid 객체
            - hgrid(laygo2.OneDimGrid): y좌표계 OneDimGrid 객체
        반환값
            - laygo2.Grid
        """
        self.name = name
        self._xy = [vgrid, hgrid]
        self.phy2abs = _PhyToAbsGridConverter(master=self)
        self.abs2phy = _AbsToPhyGridConverter(master=self)

    @property
    def elements(self):
        """list: return elements of subgrids 
        ([_xy[0].elements, _xy[1].elements]).
        
        """
        return [self._xy[0].elements, self._xy[1].elements]

    # Indexing and slicing functions
    def __getitem__(self, pos):
        return self.abs2phy.__getitem__(pos)

    # Comparison operators
    def __eq__(self, other):
        """Return the physical grid coordinate that matches to other."""
        return self.abs2phy.__eq__(other)

    def __lt__(self, other):
        """Return the index of the grid coordinate that is the largest 
        but less than other.
        """
        return self.abs2phy.__lt__(other)

    def __le__(self, other):
        """Return the index of the grid coordinate that is the largest 
        but less than or equal to other.
        """
        return self.abs2phy.__le__(other)

    def __gt__(self, other):
        """Return the index of the grid coordinate that is the smallest 
        but greater than other.
        """
        return self.abs2phy.__gt__(other)

    def __ge__(self, other):
        """Return the index of the grid coordinate that is the smallest 
        but greater than or equal to other.
        """
        return self.abs2phy.__ge__(other)

    def bbox(self, obj):
        """
        Return the abstract grid coordinates corresponding to the 
        'internal' bounding box of obj.
        
        See Also
        --------
        _PhyToAbsGridConverter.bbox
        """
        return self.phy2abs.bbox(obj)

    def bottom_left(self, obj):
        """
        Return the abstract grid coordinates corresponding to the 
        bottom-left corner of obj.
        
        See Also
        --------
        _PhyToAbsGridConverter.bottom_left
        """
        return self.phy2abs.bottom_left(obj)

    def bottom_right(self, obj):
        """
        Return the abstract grid coordinates corresponding to the 
        bottom-right corner of obj.
        
        See Also
        --------
        _PhyToAbsGridConverter.bottom_right
        """
        return self.phy2abs.bottom_right(obj)

    def top_left(self, obj):
        """
        Return the abstract grid coordinates corresponding to the top-left 
        corner of obj.

        See Also
        --------
        _PhyToAbsGridConverter.top_left
        """
        return self.phy2abs.top_left(obj)

    def top_right(self, obj):
        """
        Return the abstract grid coordinates corresponding to the top-right 
        corner of obj.
        
        See Also
        --------
        _PhyToAbsGridConverter.top_right
        """
        return self.phy2abs.top_right(obj)

    def crossing(self, *args):
        """
        Return the abstract grid coordinates corresponding to the crossing 
        point of args.
        
        See Also
        --------
        laygo2.object.grid._PhyToAbsGridConverter.crossing
        """
        return self.phy2abs.crossing(*args)

    def overlap(self, *args, type="bbox"):
        """
        Return the abstract grid coordinates corresponding to the overlap 
        of args.

        See Also
        --------
        laygo2.object.grid._PhyToAbsGridConverter.overlap
        """
        return self.phy2abs.overlap(*args, type=type)

    def union(self, *args):
        """
        Return the abstract grid coordinates corresponding to union of args.

        See Also
        --------
        laygo2.object.grid._PhyToAbsGridConverter.union
        """
        return self.phy2abs.union(*args)

    def center(self, obj):
        """
        Return the abstract grid coordinates corresponding to the center 
        point of obj.

        Parameters
        ----------
        obj : laygo2.object.physical.PhysicalObject
            The object of which center coordinate is computed.

        See Also
        --------
        laygo2.object.grid._PhyToAbsGridConverter.center
        """
        return self.phy2abs.center(obj)

    # Iterators
    def __iter__(self):
        # TODO: fix this to iterate over the full coordinates
        return np.array([self._xy[0].__iter__(), self._xy[1].__iter__()])

    def __next__(self):
        # TODO: fix this to iterate over the full coordinates
        return np.array([self._xy[0].__next__(), self._xy[1].__next__()])

    # Informative functions
    def __str__(self):
        """Return the string representation of the object."""
        return self.summarize()

    def summarize(self):
        """
        Output the information of the respective grid.

        Parameters
        ----------
        None

        Returns
        -------
        str

        Example
        --------
        >>> from laygo2.object.grid import OneDimGrid, Grid
        >>> g1_x = OneDimGrid(name='xgrid', scope=[0, 100], elements=[0, 10, 20, 40, 50 ])
        >>> g1_y = OneDimGrid(name='ygrid', scope=[0, 100], elements=[10, 20, 40, 50, 60 ])
        >>> g2   = Grid(name="test", vgrid = g1_x, hgrid = g1_y )
        >>> g2.summarize()
        <laygo2.object.grid.Grid object> name: test, class: Grid, scope: [[0, 0], [100, 100]], elements: [array([ 0, 10, 20, 40, 50]), array([10, 20, 40, 50, 60])

        Notes
        -----
        **(Korean)** 해당 Grid의 정보 출력.
        """
        return (
            self.__repr__() + " "
            "name: "
            + self.name
            + ", "
            + "class: "
            + self.__class__.__name__
            + ", "
            + "scope: "
            + str(self.range.tolist())
            + ", "
            + "elements: "
            + str(self.elements)
        )


# Regular classes.
class PlacementGrid(Grid):
    """
    PlacementGrid class implements a grid for placement of Instance and 
    VirtualInstance objects.

    Notes
    -----
    **(Korean)** PlacementGrid 클래스는 Instance 및 VirtualInstance 개체들의 
        배치를 위한 격자 그리드를 구현한다.

    """

    type = "placement"
    """ Type of grid. Should be 'placement' for placement grids."""

    def place(self, inst, mn):
        """
        Place the instance on the specified coordinate mn, on this grid.

        Parameters
        ----------
        inst : laygo2.object.physical.Instance or laygo2.object.physical.VirtualInstance
            The instance to be placed on the grid.
        mn : numpy.ndarray or list
            The abstract coordinate [m, n] to place the instance.

        Returns
        -------
        laygo2.object.physical.Instance or 
        laygo2.object.physical.VirtualInstance :
            The placed instance.

        Example
        --------
        >>> import laygo2
        >>> from laygo2.object.grid import OneDimGrid, PlacementGrid
        >>> from laygo2.object.physical import Instance
        >>> # Create a grid (not needed if laygo2_tech is set up).
        >>> gx  = OneDimGrid(name="gx", scope=[0, 20], elements=[0])
        >>> gy  = OneDimGrid(name="gy", scope=[0, 100], elements=[0])
        >>> g   = PlacementGrid(name="test", vgrid=gx, hgrid=gy)
        >>> # Create an instance
        >>> i0 = Instance(libname="tlib", cellname="t0", name="I0", xy=[0, 0])
        >>> print(inst0.xy)
        [100, 100]
        >>> # Place the instance
        >>> g.place(inst=i0, mn=[10,10])
        >>> # Print parameters of the placed instance.
        >>> print(i0.xy)
        [200, 1000]

        Notes
        -----
        **(Korean)** 인스턴스 xy속성에 추상좌표를 매핑함.
            파라미터
            - inst(laygo2.physical.instance): 배치할 인스턴스
            - mn(numpy.ndarray or list): 인스턴스를 배치할 추상좌표
            반환값
            - laygo2.physical.instance: 좌표가 수정된 인스턴스
        """
        inst.xy = self[mn]
        return inst


class RoutingGrid(Grid):
    """
    A class that implements wire connections in an abstract coordinate system.

    Notes
    -----
    **(Korean)** 추상 좌표계 상의 배선 동작을 구현하는 클래스.
    """

    type = "routing"
    """ Type of grid. Should be 'routing' for routing grids."""

    vwidth = None
    """CircularMapping: Width of vertical wires.

    Example
    --------
    >>> import laygo2
    >>> from laygo2.object.grid import CircularMapping as CM
    >>> from laygo2.object.grid import CircularMappingArray as CMA
    >>> from laygo2.object.grid import OneDimGrid, RoutingGrid
    >>> from laygo2.object.template import NativeInstanceTemplate
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
    >>> print(g.vwidth)
    <laygo2.object.grid.CircularMapping object > 
        class: CircularMapping, 
        elements: [10]

    .. image:: ../assets/img/object_grid_RoutingGrid_vwidth.png
           :height: 250

    See Also
    --------
    vwidth, hwidth, vextension, hextension, vextension0, hextension0

    Notes
    -----
    **(Korean)** 수직 wire들의 폭.
    """

    hwidth = None
    """CircularMapping: Width of horizontal wires.

    Example
    --------
    >>> import laygo2
    >>> from laygo2.object.grid import CircularMapping as CM
    >>> from laygo2.object.grid import CircularMappingArray as CMA
    >>> from laygo2.object.grid import OneDimGrid, RoutingGrid
    >>> from laygo2.object.template import NativeInstanceTemplate
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
    >>> print(g.hwidth)
    <laygo2.object.grid.CircularMapping object > 
        class: CircularMapping, 
        elements: [20, 10, 10]

    .. image:: ../assets/img/object_grid_RoutingGrid_hwidth.png
           :height: 250 

    See Also
    --------
    vwidth, hwidth, vextension, hextension, vextension0, hextension0

    Notes
    -----
    **(Korean)** 수평 wire들의 폭.
    """

    vextension = None
    """CircularMapping: Extension of vertical wires.

    Example
    --------
    >>> import laygo2
    >>> from laygo2.object.grid import CircularMapping as CM
    >>> from laygo2.object.grid import CircularMappingArray as CMA
    >>> from laygo2.object.grid import OneDimGrid, RoutingGrid
    >>> from laygo2.object.template import NativeInstanceTemplate
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
    >>> print(g.vextension)
    <laygo2.object.grid.CircularMapping object > 
        class: CircularMapping, 
        elements: [10]

    .. image:: ../assets/img/object_grid_RoutingGrid_vextension.png
           :height: 250

    Notes
    -----
    **(Korean)** 수직 wire들의 extension.
    """

    hextension = None
    """CircularMapping: Extension of horizontal wires.

    Example
    --------
    >>> import laygo2
    >>> from laygo2.object.grid import CircularMapping as CM
    >>> from laygo2.object.grid import CircularMappingArray as CMA
    >>> from laygo2.object.grid import OneDimGrid, RoutingGrid
    >>> from laygo2.object.template import NativeInstanceTemplate
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
    >>> print(g.hextension)
    <laygo2.object.grid.CircularMapping object > 
        class: CircularMapping, 
        elements: [10, 10, 10]

    .. image:: ../assets/img/object_grid_RoutingGrid_hextension.png
           :height: 250 

    Notes
    -----
    **(Korean)** 수평 wire들의 extension.
    """

    vextension0 = None
    """CircularMapping: the array containing the extension of the zero-length wires on the vertical grid.
    
    Example
    --------
    >>> import laygo2
    >>> from laygo2.object.grid import CircularMapping as CM
    >>> from laygo2.object.grid import CircularMappingArray as CMA
    >>> from laygo2.object.grid import OneDimGrid, RoutingGrid
    >>> from laygo2.object.template import NativeInstanceTemplate
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
    >>> print(g.vextension0)
    <laygo2.object.grid.CircularMapping object > 
        class: CircularMapping, 
        elements: [15]
    """
    
    hextension0 = None
    """CircularMapping: the array containing the extension of the zero-length wires on the horizontal grid. 
    
    Example
    --------
    >>> import laygo2
    >>> from laygo2.object.grid import CircularMapping as CM
    >>> from laygo2.object.grid import CircularMappingArray as CMA
    >>> from laygo2.object.grid import OneDimGrid, RoutingGrid
    >>> from laygo2.object.template import NativeInstanceTemplate
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
    >>> print(g.hextension0)
    <laygo2.object.grid.CircularMapping object > 
        class: CircularMapping, 
        elements: [15, 15, 15]
    """
    
    vlayer = None
    """CircularMapping: Layer information of vertical wires.

    Example
    --------
    >>> import laygo2
    >>> from laygo2.object.grid import CircularMapping as CM
    >>> from laygo2.object.grid import CircularMappingArray as CMA
    >>> from laygo2.object.grid import OneDimGrid, RoutingGrid
    >>> from laygo2.object.template import NativeInstanceTemplate
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
    >>> print(g.vlayer)
    <laygo2.object.grid.CircularMapping object > 
        class: CircularMapping, 
        elements: [['M1', 'drawing']]

    .. image:: ../assets/img/object_grid_RoutingGrid_vlayer.png
           :height: 250

    Notes
    -----
    **(Korean)** 수직 wire들의 레이어 정보.
    """

    hlayer = None
    """CircularMapping: Layer information of horizontal wires.

    Example
    --------
    >>> import laygo2
    >>> from laygo2.object.grid import CircularMapping as CM
    >>> from laygo2.object.grid import CircularMappingArray as CMA
    >>> from laygo2.object.grid import OneDimGrid, RoutingGrid
    >>> from laygo2.object.template import NativeInstanceTemplate
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
    >>> print(g.hlayer)
    <laygo2.object.grid.CircularMapping object > 
        class: CircularMapping, 
        elements: [['M1', 'drawing'], ['M1', 'drawing'], ['M1', 'drawing']]

    .. image:: ../assets/img/object_grid_RoutingGrid_hlayer.png
           :height: 250

    Notes
    -----
    **(Korean)** 수평 wire들의 레이어정보.
    """

    pin_vlayer = None
    """CircularMapping: Layer information of vertical pin wires.

    Example
    --------
    >>> import laygo2
    >>> from laygo2.object.grid import CircularMapping as CM
    >>> from laygo2.object.grid import CircularMappingArray as CMA
    >>> from laygo2.object.grid import OneDimGrid, RoutingGrid
    >>> from laygo2.object.template import NativeInstanceTemplate
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
    >>> print(g.pin_vlayer)
    <laygo2.object.grid.CircularMapping object > 
        class: CircularMapping, 
        elements: [['M1', 'pin']]

    .. image:: ../assets/img/object_grid_RoutingGrid_pin_vlayer.png
           :height: 250

    Notes
    -----
    **(Korean)** 수직 pin wire들의 레이어 정보.
    """

    pin_hlayer = None
    """CircularMapping: Layer information of horizontal pine wires.

    Example
    --------
    >>> import laygo2
    >>> from laygo2.object.grid import CircularMapping as CM
    >>> from laygo2.object.grid import CircularMappingArray as CMA
    >>> from laygo2.object.grid import OneDimGrid, RoutingGrid
    >>> from laygo2.object.template import NativeInstanceTemplate
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
    >>> print(g.pin_hlayer)
    <laygo2.object.grid.CircularMapping object > 
        class: CircularMapping, 
        elements: [['M1', 'pin'], ['M1', 'pin'], ['M1', 'pin']]

    .. image:: ../assets/img/object_grid_RoutingGrid_pin_hlayer.png
           :height: 250

    Notes
    -----
    **(Korean)** 수평 pin wire 들의 레이어정보.
    """

    viamap = None
    """CircularMappingArray: Array containing Via objects positioned on grid crossing points.

    Example
    --------
    >>> import laygo2
    >>> from laygo2.object.grid import CircularMapping as CM
    >>> from laygo2.object.grid import CircularMappingArray as CMA
    >>> from laygo2.object.grid import OneDimGrid, RoutingGrid
    >>> from laygo2.object.template import NativeInstanceTemplate
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
    >>> print(g.viamap)
    <laygo2.object.grid.CircularMappingArray object at 0x000002217F15A530> 
    class: CircularMappingArray, 
    elements: [
        [<laygo2.object.template.NativeInstanceTemplate object at 0x000002217F15ADD0>
         <laygo2.object.template.NativeInstanceTemplate object at 0x000002217F15ADD0>
         <laygo2.object.template.NativeInstanceTemplate object at 0x000002217F15ADD0>]]

    .. image:: ../assets/img/object_grid_RoutingGrid_viamap.png
           :height: 250

    Notes
    -----
    **(Korean)** 그리드 교차점에 위치하는 via개채들을 담고있는배열.
    """

    primary_grid = "vertical"
    """str: The default direction of routing 
        (Direction of wire having length 0).

    Example
    --------
    >>> import laygo2
    >>> from laygo2.object.grid import CircularMapping as CM
    >>> from laygo2.object.grid import CircularMappingArray as CMA
    >>> from laygo2.object.grid import OneDimGrid, RoutingGrid
    >>> from laygo2.object.template import NativeInstanceTemplate
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
    >>> print(g.primary_grid) 
    “horizontal”

    .. image:: ../assets/img/object_grid_RoutingGrid_primary_grid.png
           :height: 250

    Notes
    -----
    **(Korean)** Routing의 기본 방향 (길이가 0인 wire방향).
    """

    xcolor = None
    """CircularMapping: Color of horizontal wires.

    Example
    --------
    >>> templates = tech.load_templates() 
    >>> grids = tech.load_grids(templates=templates) 
    >>> r23   = grids['routing_23_cmos’] 
    >>> print(r23.xcolor) 
    <laygo2.object.grid.CircularMapping object> class: CircularMapping, 
        elements: [[“colorA”], [“colorB”], [“colorA”], [“colorB”], [“colorA”], [“colorB”], [“colorA”], [“colorB”]]

    .. image:: ../assets/img/object_grid_RoutingGrid_xcolor.png
           :height: 250

    Notes
    -----
    **(Korean)** 수평 wire 들의 color.
    """

    ycolor = None
    """CircularMapping: Color of vertical wires.

    Example
    --------
    >>> templates = tech.load_templates() 
    >>> grids = tech.load_grids(templates=templates) 
    >>> r23   = grids['routing_23_cmos’]
    >>> print(r23.ycolor) 
    <laygo2.object.grid.CircularMapping object> class: CircularMapping, 
        elements: [[“colorA”]]

    .. image:: ../assets/img/object_grid_RoutingGrid_ycolor.png
           :height: 250

    Notes
    -----
    **(Korean)** 수직 wire들의 color.
    """

    def __init__(
        self,
        name,
        vgrid,
        hgrid,
        vwidth,
        hwidth,
        vextension,
        hextension,
        vlayer,
        hlayer,
        pin_vlayer,
        pin_hlayer,
        viamap,
        xcolor=None,
        ycolor=None,
        primary_grid="vertical",
        vextension0=None,
        hextension0=None,
    ):
        """
        Constructor function of RoutingGrid class.

        Parameters
        ----------
        name : str
            Routing object name
        vgrid : laygo2.OneDimGrid
            OneDimGrid of x-coordinate system
        hgrid : laygo2.OneDimGrid
            OneDimGrid of y-coordinate system
        vwidth : CircularMapping
            x-coordinate system width
        hwidth : CircularMapping
            y-coordinate system width
        vextension : CircularMapping
            x-coordinate system extension
        hextension : CircularMapping
            y-coordinate system extension
        vlayer : CircularMapping
            x-coordinate system layer
        hlayer : CircularMapping
            y-coordinate system layer
        pin_vlayer : CircularMapping
            layer of x-coordinate system pin
        pin_hlayer : CircularMapping
            layer of y-coordinate system pin
        xcolor : list
            x-coordinate system color
        ycolor : list
            y-coordinate system color
        viamap : CircularMappingArray
            Via map of Grid
        primary_grid : str
            direction of wire having length 0

        Returns
        -------
        laygo2.RoutingGrid

        Example
        --------
        >>> import laygo2
        >>> from laygo2.object.grid import CircularMapping as CM
        >>> from laygo2.object.grid import CircularMappingArray as CMA
        >>> from laygo2.object.grid import OneDimGrid, RoutingGrid
        >>> from laygo2.object.template import NativeInstanceTemplate
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
        >>> # Routing on grid 
        >>> mn_list = [[0, -2], [0, 1], [2, 1], [5,1] ]
        >>> route = g.route(mn=mn_list, via_tag=[True, None, True, True])
        >>> for r in route:
        >>>     print(r)
        <laygo2.object.physical.Instance object at 0x0000016939A23A90> 
            name: None,
            class: Instance,
            xy: [0, -60],
            params: None,
            size: [0, 0]
            shape: None
            pitch: [0, 0]
            transform: R0
            pins: {}
        <laygo2.object.physical.Rect object at 0x0000016939A23880>
            name: None,
            class: Rect,
            xy: [[0, -60], [0, 40]],
            params: None, , layer: ['M1' 'drawing'], netname: None
        <laygo2.object.physical.Rect object at 0x0000016939A21BA0>
            name: None,
            class: Rect,
            xy: [[0, 40], [100, 40]],
            params: None, , layer: ['M2' 'drawing'], netname: None
        <laygo2.object.physical.Instance object at 0x0000016939A21B70>
            name: None,
            class: Instance,
            xy: [100, 40],
            params: None,
            size: [0, 0]
            shape: None
            pitch: [0, 0]
            transform: R0
            pins: {}
        <laygo2.object.physical.Rect object at 0x0000016939A21D80>
            name: None,
            class: Rect,
            xy: [[100, 40], [250, 40]],
            params: None, , layer: ['M2' 'drawing'], netname: None
        <laygo2.object.physical.Instance object at 0x0000016939A22350>
            name: None,
            class: Instance,
            xy: [250, 40],
            params: None,
            size: [0, 0]
            shape: None
            pitch: [0, 0]
            transform: R0
            pins: {}
        
        .. image:: ../assets/img/object_grid_RoutingGrid_init.png
           :height: 250

        Notes
        -----
        **(Korean)**
        RoutingGrid 클래스의 생성자함수.
        파라미터
        name(str): Routing 객체의 이름
        vgrid(laygo2.OneDimGrid): x좌표계 OneDimGrid
        hgrid(laygo2.OneDimGrid): y좌표계 OneDimGrid
        vwidth(CircularMapping): x좌표계 Width
        hwidth(CircularMapping): y좌표계 Width
        vextension(CircularMapping): x좌표계의 extension
        hextension(CircularMapping): y좌표계의 extension
        vlayer(CircularMapping): x좌표계의 layer
        hlayer(CircularMapping): y좌표계의 layer
        pin_vlayer(CircularMapping): x좌표계 pin의 layer
        pin_hlayer(CircularMapping): y좌표계 pin의 layer
        xcolor(list): x좌표계 color
        ycolor(list): y좌표계 color
        viamap(CircularMappingArray): Grid의 Via map
        primary_grid(str): 길이가 0인 Wire방향
        반환값
        laygo2.RoutingGrid
        참조
        없음
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
        self.xcolor = xcolor
        self.ycolor = ycolor
        Grid.__init__(self, name=name, vgrid=vgrid, hgrid=hgrid)

    def route(self, mn, direction=None, via_tag=None):
        """
        Create wire object(s) for routing.

        Parameters
        ----------
        mn : list(numpy.ndarray)
            The list containing two or more mn coordinates to be connected.
        direction : str, optional.
            None or “vertical” or "horizontal". The direction of the routing object.
        via_tag : list(Boolean), optional.
            The list containing switches deciding whether to place via at the edges.

        Returns
        -------
        laygo2.object.physical.Rect or list :
            The generated routing object(s). Check the example code for details.

        Example
        --------
        >>> import laygo2
        >>> from laygo2.object.grid import CircularMapping as CM
        >>> from laygo2.object.grid import CircularMappingArray as CMA
        >>> from laygo2.object.grid import OneDimGrid, RoutingGrid
        >>> from laygo2.object.template import NativeInstanceTemplate
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
        >>> # Routing on grid 
        >>> mn_list = [[0, -2], [0, 1], [2, 1], [5,1] ]
        >>> route = g.route(mn=mn_list, via_tag=[True, None, True, True])
        >>> for r in route:
        >>>     print(r)
        <laygo2.object.physical.Instance object at 0x0000016939A23A90> 
            name: None,
            class: Instance,
            xy: [0, -60],
            params: None,
            size: [0, 0]
            shape: None
            pitch: [0, 0]
            transform: R0
            pins: {}
        <laygo2.object.physical.Rect object at 0x0000016939A23880>
            name: None,
            class: Rect,
            xy: [[0, -60], [0, 40]],
            params: None, , layer: ['M1' 'drawing'], netname: None
        <laygo2.object.physical.Rect object at 0x0000016939A21BA0>
            name: None,
            class: Rect,
            xy: [[0, 40], [100, 40]],
            params: None, , layer: ['M2' 'drawing'], netname: None
        <laygo2.object.physical.Instance object at 0x0000016939A21B70>
            name: None,
            class: Instance,
            xy: [100, 40],
            params: None,
            size: [0, 0]
            shape: None
            pitch: [0, 0]
            transform: R0
            pins: {}
        <laygo2.object.physical.Rect object at 0x0000016939A21D80>
            name: None,
            class: Rect,
            xy: [[100, 40], [250, 40]],
            params: None, , layer: ['M2' 'drawing'], netname: None
        <laygo2.object.physical.Instance object at 0x0000016939A22350>
            name: None,
            class: Instance,
            xy: [250, 40],
            params: None,
            size: [0, 0]
            shape: None
            pitch: [0, 0]
            transform: R0
            pins: {}

        .. image:: ../assets/img/object_grid_RoutingGrid_route.png
           :height: 250

        Notes
        -----
        **(Korean)**
        추상 좌표 위에 라우팅을 수행 하는 함수.
        파라미터
        mn(list(numpy.ndarray)): 배선을 수행할 2개 이상의 mn 좌표를 담고 있는 list.
        direction(str): None or “vertical”; path의 방향을 결정 (수평 or 수직) [optional].
        via_tag(list(Boolean)): Path에 via를 형성 할지를 결정하는 switch들을 담고 있는 list [optional].
        반환값
        list: 생성된 routing object들을 담고 있는 list.
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
            if np.all(
                xy0 == xy1
            ):  # if two points are identical, generate a metal stub on the bottom layer.
                if (direction == "vertical") or (
                    (direction is None) and (self.primary_grid == "vertical")
                ):
                    width = self.vwidth[__mn[0][0]]
                    hextension = int(width / 2)
                    vextension = self.vextension0[__mn[0][0]]
                    layer = self.vlayer[__mn[0][0]]
                    if self.xcolor is not None:
                        color = self.xcolor[
                            __mn[0][0] % self.xcolor.shape[0]
                        ]  # xcolor is determined by its grid layer.
                    else:
                        color = "not MPT"
                else:
                    width = self.hwidth[__mn[0][1]]
                    hextension = self.hextension0[__mn[0][1]]
                    vextension = int(width / 2)
                    layer = self.hlayer[__mn[0][1]]
                    if self.ycolor is not None:
                        color = self.ycolor[
                            __mn[0][1] % self.ycolor.shape[0]
                        ]  # ycolor is determined by its grid layer.
                    else:
                        color = "not MPT"
            else:
                if (xy0[0] == xy1[0]) or (direction == "vertical"):  # vertical routing
                    width = self.vwidth[__mn[0][0]]
                    hextension = int(width / 2)
                    vextension = self.vextension[__mn[0][0]]
                    layer = self.vlayer[__mn[0][0]]
                    if self.xcolor is not None:
                        color = self.xcolor[
                            __mn[0][0] % self.xcolor.shape[0]
                        ]  # xcolor is determined by its grid layer.
                    else:
                        color = "not MPT"
                else:  # horizontal routing
                    width = self.hwidth[__mn[0][1]]
                    hextension = self.hextension[__mn[0][1]]
                    vextension = int(width / 2)
                    layer = self.hlayer[__mn[0][1]]
                    if self.ycolor is not None:
                        color = self.ycolor[
                            __mn[0][1] % self.ycolor.shape[0]
                        ]  # ycolor is determined by its grid layer.
                    else:
                        color = "not MPT"
            p = laygo2.object.physical.Rect(
                xy=_xy,
                layer=layer,
                hextension=hextension,
                vextension=vextension,
                color=color,
            )
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
        >>> # Routing on grid 
        >>> mn_list = [[0, -2], [1, 0], [2, 5]]
        >>> via = mygrid.via(mn=mn_list)
        >>> print(via)
        [<laygo2.object.physical.VirtualInstance object>, 
         <laygo2.object.physical.VirtualInstance object>, 
         <laygo2.object.physical.VirtualInstance object>]

        .. image:: ../assets/img/object_grid_RoutingGrid_via.png
           :height: 250

        Notes
        -----
        **(Korean)** via 생성함수.

        파라미터
            - mn(list(numpy.ndarray)): via를 생성할 mn좌표. 복수 개 입력 가능.
        반환값
            - list(physical.PhysicalObject)): 생성된 via들을 담고 있는 list.
        """
        # If mn contains multiple coordinates (or objects), place iteratively.
        if isinstance(mn, list):
            if isinstance(
                mn[0], (int, np.integer)
            ):  # It's actually a single coordinate.
                return self.via(mn=np.asarray(mn), params=params)
            else:
                return [self.via(mn=_mn, params=params) for _mn in mn]
        elif isinstance(mn, np.ndarray):
            if isinstance(
                mn[0], (int, np.integer)
            ):  # It's actually a single coordinate.
                pass
            else:
                return np.array([self.via(mn=_mn, params=params) for _mn in mn])
        if not isinstance(mn, tuple):
            mn = tuple(mn)  # viamap (CircularMapping) works only with tuples
        tvia = self.viamap[mn]
        via = tvia.generate(params=params)
        via.xy = self[mn]
        return via

    def route_via_track(self, mn, track, via_tag=[None, True]):
        """
        Perform routing on the specified track with accessing wires to mn.

        Parameters
        ----------
        mn : list(numpy.ndarray)
            list containing coordinates of the points being connected through a track
        track : numpy.ndarray
            list containing coordinate values and direction of a track.
            Vertical tracks have [v, None] format, while horizontal tracks have [None, v] format
            (v is the coordinates of the track).

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
        >>> # Routing on grid 
        >>> mn_list = [[0, -2], [1, 0], [2, 5], [3, 4], [4, 5], [5, 5]]
        >>> track = g.route_via_track(mn=mn_list, track=[None, 0])
        >>> print(track)
        [[<laygo2.object.physical.Rect object>, 
          <laygo2.object.physical.VirtualInstance object>], 
          <laygo2.object.physical.VirtualInstance object>, 
         [<laygo2.object.physical.Rect object>, 
          <laygo2.object.physical.VirtualInstance object>], 
         [<laygo2.object.physical.Rect object>, 
          <laygo2.object.physical.VirtualInstance object>], 
         [<laygo2.object.physical.Rect object>, 
          <laygo2.object.physical.VirtualInstance object>], 
         [<laygo2.object.physical.Rect object>, 
          <laygo2.object.physical.VirtualInstance object>], 
          <laygo2.object.physical.Rect object>]

        .. image:: ../assets/img/object_grid_RoutingGrid_route_via_track.png
           :height: 250

        Notes
        -----
        **(Korean)** wire 라우팅 함수, track을 기준점으로 routing을 진행한다.
        
        파라미터
            - track(numpy.ndarray): track의 좌표값과 방향을 담고 있는 list. 
                수직 트랙일 경우 [v, None], 
                수평 트랙일 경우 [None, v]의 형태를 가지고 있다 (v는 track의 좌표값).
            - mn(list(numpy.ndarray)): track을 통해 연결될 지점들의 좌표를 담고 있는 list.
        반환값
            - list: 생성된 routing object들을 담고 있는 list. 
                마지막 object가 track위의 routing object에 해당.
        """
        mn = np.array(mn)
        route = list()

        if track[1] != None:  # x direction
            t = 0  # index of track axis
            p = 1  # index of perpendicular track
            mn_pivot = track[1]
        else:  # y direction
            t = 1
            p = 0
            mn_pivot = track[0]

        mn_b = np.array([[0, 0], [0, 0]])  # 1.branch
        min_t, max_t = mn[0][t], mn[0][t]

        for i in range(len(mn)):
            mn_b[0] = mn[i]
            mn_b[1][t] = mn_b[0][t]
            mn_b[1][p] = mn_pivot
            if np.array_equal(mn_b[0], mn_b[1]):  #### via only
                route.append(self.via(mn=mn_b[0], params=None))
            else:
                route.append(self.route(mn=[mn_b[0], mn_b[1]], via_tag=via_tag))

            center_t = mn[i][t]
            if center_t < min_t:
                min_t = center_t
            elif max_t < center_t:
                max_t = center_t

        mn_track = np.array([[0, 0], [0, 0]])  # 2.track
        mn_track[0][t], mn_track[0][p] = min_t, mn_pivot  # min
        mn_track[1][t], mn_track[1][p] = max_t, mn_pivot  # max

        if np.array_equal(mn_track[0], mn_track[1]):  # Skip
            route.append(None)
        else:
            route.append(self.route(mn=mn_track))

        return route

    def pin(self, name, mn, direction=None, netname=None, params=None):
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
        >>> mn = [[0, 0], [10, 10]]
        >>> pin = g.pin(name="pin", grid=g, mn=mn)
        >>> print(pin)
        <laygo2.object.physical.Pin object at 0x0000028DABE3AB90> 
            name: pin,
            class: Pin,
            xy: [[0, -10], [500, 350]],
            params: None, , layer: ['M2' 'pin'], netname: pin, shape: None, 
            master: None

        Notes
        -----
        **(Korean)** pin 생성함수.

        파라미터
            - name(str): Pin 이름
            - mn(numpy.ndarray): Pin을 생성할 abstract 좌표
            - direction(str): 방향 [optional]
            - netname(str): Pin의 net이름 [optional]
            - params(dict): Pin 속성 [optional]
        반환값
            - laygo2.physical.Pin: Pin object
        """
        xy0 = self.abs2phy[mn[0]]
        xy1 = self.abs2phy[mn[1]]
        # _xy = np.array([[xy0[0], xy0[1]], [xy1[0], xy1[1]]])
        if np.all(
            xy0 == xy1
        ):  # if two points are identical, generate a metal stub on the bottom layer.
            if (direction == "vertical") or (
                (direction is None) and (self.primary_grid == "vertical")
            ):
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
            if (xy0[0] == xy1[0]) or (direction == "vertical"):  # vertical routing
                width = self.vwidth[mn[0][0]]
                hextension = int(width / 2)
                vextension = 0
                layer = self.pin_vlayer[mn[0][0]]
            else:  # horizontal routing
                width = self.hwidth[mn[0][1]]
                hextension = 0
                vextension = int(width / 2)
                layer = self.pin_hlayer[mn[0][1]]
        # TODO: pin.xy differ from tech.py.
        _xy = np.array(
            [
                [xy0[0] - hextension, xy0[1] - vextension],
                [xy1[0] + hextension, xy1[1] + vextension],
            ]
        )  ## need to check
        p = laygo2.object.physical.Pin(
            name=name, xy=_xy, layer=layer, netname=netname, params=params
        )
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
