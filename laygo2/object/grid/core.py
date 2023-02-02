#!/usr/bin/python
########################################################################################################################
#
# Copyright (c) 2020, Nifty Chips Laboratory, Hanyang University
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
    Iterable: The extended index.

    Example
    -------
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
    -------
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
    -------
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
    -------
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


# External functions
def copy(obj):
    """Make a copy of the input grid object.

    Parameters
    ----------
    obj : laygo2.object.grid.Grid
        The input grid object to be copied.

    Returns
    -------
    laygo2.object.grid.Grid or derived: the copied grid object.

    Example
    -------
    >>> import laygo2
    >>> from laygo2.object.grid import OneDimGrid, Grid
    >>> g1_x = OneDimGrid(name='xgrid', scope=[0, 100], elements=[0, 10, 20, 40, 50 ])
    >>> g1_y = OneDimGrid(name='ygrid', scope=[0, 100], elements=[10, 20, 40, 50, 60 ])
    >>> g2   = Grid(name="test", vgrid = g1_x, hgrid = g1_y )
    >>> g2_copy = laygo2.object.grid.copy(g2)
    >>> print(g2)
    <laygo2.object.grid.core.Grid object at 0x000002002EBA67A0>
     name: test,
    class: Grid,
    scope: [[0, 0], [100, 100]],
    elements: [array([ 0, 10, 20, 40, 50]), array([10, 20, 40, 50, 60])],
    >>> print(g2_copy)
    <laygo2.object.grid.core.Grid object at 0x0000020040C35240>
     name: test,
     class: Grid,
     scope: [[0, 0], [100, 100]],
     elements: [array([ 0, 10, 20, 40, 50]), array([10, 20, 40, 50, 60])],
    """
    return obj.copy()


def vflip(obj):
    """Make a vertically-flipped copy of the input grid object.
    
    Parameters
    ----------
    obj : laygo2.object.grid.Grid
        The input grid object to be copied and flipped.

    Returns
    -------
    laygo2.object.grid.Grid or derived: the generated grid object.

    Example
    -------
    >>> import laygo2
    >>> from laygo2.object.grid import OneDimGrid, Grid
    >>> g1_x = OneDimGrid(name='xgrid', scope=[0, 100], elements=[0, 10, 20, 40, 50 ])
    >>> g1_y = OneDimGrid(name='ygrid', scope=[0, 100], elements=[10, 20, 40, 50, 60 ])
    >>> g2   = Grid(name="test", vgrid = g1_x, hgrid = g1_y )
    >>> g2_copy = laygo2.object.grid.vflip(g2)
    >>> print(g2)
    <laygo2.object.grid.core.Grid object at 0x000001EE82660BE0>
     name: test,
     class: Grid,
     scope: [[0, 0], [100, 100]],
     elements: [array([ 0, 10, 20, 40, 50]), array([10, 20, 40, 50, 60])],
    >>> print(g2_copy)
    <laygo2.object.grid.core.Grid object at 0x000001EE947152D0>
     name: test,
     class: Grid,
     scope: [[0, 0], [100, 100]],
     elements: [array([ 0, 10, 20, 40, 50]), array([40, 50, 60, 80, 90])],
    """
    return obj.vflip(copy=True)


def hflip(obj):
    """Make a horizontally-flipped copy of the input grid object.
    
    Parameters
    ----------
    obj : laygo2.object.grid.Grid
        The input grid object to be copied and flipped.

    Returns
    -------
    laygo2.object.grid.Grid: the generated grid object.

    Example
    -------
    >>> import laygo2
    >>> from laygo2.object.grid import OneDimGrid, Grid
    >>> g1_x = OneDimGrid(name='xgrid', scope=[0, 100], elements=[0, 10, 20, 40, 50 ])
    >>> g1_y = OneDimGrid(name='ygrid', scope=[0, 100], elements=[10, 20, 40, 50, 60 ])
    >>> g2   = Grid(name="test", vgrid = g1_x, hgrid = g1_y )
    >>> g2_copy = laygo2.object.grid.hflip(g2)
    >>> print(g2)
    <laygo2.object.grid.core.Grid object at 0x000001ACECC30BE0>
     name: test,
     class: Grid,
     scope: [[0, 0], [100, 100]],
     elements: [array([ 0, 10, 20, 40, 50]), array([10, 20, 40, 50, 60])],
    >>> print(g2_copy)
    <laygo2.object.grid.core.Grid object at 0x000001ACFED15300>
     name: test,
     class: Grid,
     scope: [[0, 0], [100, 100]],
     elements: [array([ 50,  60,  80,  90, 100]), array([10, 20, 40, 50, 60])],
    """
    return obj.hflip(copy=True)


def vstack(obj):
    """Stack grid(s) in vertical direction.
    
    Parameters
    ----------
    obj : list of laygo2.object.grid.Grid
        The list containing grid objects to be stacked.
    
    Returns
    -------
    laygo2.object.grid.Grid: the generated grid object.
    
    Example
    -------
    >>> import laygo2
    >>> from laygo2.object.grid import OneDimGrid, Grid
    >>> g1_x = OneDimGrid(name='xgrid', scope=[0, 100], elements=[0, 10, 20, 40, 50 ])
    >>> g1_y = OneDimGrid(name='ygrid', scope=[0, 100], elements=[10, 20, 40, 50, 60 ])
    >>> g2   = Grid(name="test", vgrid = g1_x, hgrid = g1_y )
    >>> g2_copy = laygo2.object.grid.copy(g2)
    >>> g2_stack = laygo2.object.grid.vstack([g2, g2_copy])
    >>> print(g2)
    <laygo2.object.grid.core.Grid object at 0x000001799FAA0BE0>
     name: test,
     class: Grid,
     scope: [[0, 0], [100, 100]],
     elements: [array([ 0, 10, 20, 40, 50]), array([10, 20, 40, 50, 60])],
    >>> print(g2_stack)
    <laygo2.object.grid.core.Grid object at 0x00000179B1B05870>
     name: test,
     class: Grid,
     scope: [[0, 0], [100, 200]],
     elements: [array([ 0, 10, 20, 40, 50]), array([ 10,  20,  40,  50,  60, 110, 120, 140, 150, 160])],
    """
    return obj[0].vstack(obj[1:], copy=True)


def hstack(obj):
    """Stack grid(s) in horizontal direction.

    Parameters
    ----------
    obj : list of laygo2.object.grid.Grid
        The list containing grid objects to be stacked.
    
    Returns
    -------
    laygo2.object.grid.Grid: the generated grid object.
    
    Example
    -------
    >>> import laygo2
    >>> from laygo2.object.grid import OneDimGrid, Grid
    >>> g1_x = OneDimGrid(name='xgrid', scope=[0, 100], elements=[0, 10, 20, 40, 50 ])
    >>> g1_y = OneDimGrid(name='ygrid', scope=[0, 100], elements=[10, 20, 40, 50, 60 ])
    >>> g2   = Grid(name="test", vgrid = g1_x, hgrid = g1_y )
    >>> g2_copy = laygo2.object.grid.copy(g2)
    >>> g2_stack = laygo2.object.grid.hstack([g2, g2_copy])
    >>> print(g2)
    <laygo2.object.grid.core.Grid object at 0x000001799FAA0BE0>
     name: test,
     class: Grid,
     scope: [[0, 0], [100, 100]],
     elements: [array([ 0, 10, 20, 40, 50]), array([10, 20, 40, 50, 60])],
    >>> print(g2_stack)
    <laygo2.object.grid.core.Grid object at 0x0000015BD8C85570>
     name: test,
     class: Grid,
     scope: [[0, 0], [200, 100]],
     elements: [array([  0,  10,  20,  40,  50, 100, 110, 120, 140, 150]), array([10, 20, 40, 50, 60])],
    """

    return obj[0].hstack(obj[1:], copy=True)


# Internal classes
class CircularMapping:
    """
    Basic circular mapping class (index number expands infinitely).

    Example
    -------
    >>> from laygo2.object.grid import Circularmapping
    >>> map = CircularMapping(elements=[100, 200, 300])
    >>> print(map[0])
    100
    >>> print(map[2])
    300
    >>> print(map[4])
    200
    >>> print(map[-3])
    100
    >>> print(map[[2, 3, -2])
    [300, 100, 200]
    >>> print(map[2:7])
    [300, 100, 200, 300, 100]

    Notes
    -----
    **(Korean)** 기본 순환 맵핑 (인덱싱 넘버가 무한히 확장) 클래스.
    """

    _elements = None
    """list: Array consisting of the elements of circular mapping.

    Example
    -------
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

    dtype = int
    """type: Data type of the circular mapping.

    Example
    -------
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
        -------
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

    def __init__(self, elements=np.array([0]), dtype=int):
        """
        Constructor function of CircularMapping class.

        Parameters
        ----------
        elements : list
            elements.
        dtype : type
            data type of elements.

        Example
        -------
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

    # Regular member functions
    def append(self, elem):
        """Append elements to the mapping."""
        if not isinstance(elem, list):
            elem = [elem]
        self.elements = np.array(self.elements.tolist() + elem)

    def flip(self):
        """Flip the elements of the object."""
        self.elements = np.flip(self.elements, axis=0)

    def copy(self):
        """Copy the object."""
        return CircularMapping(self.elements.copy(), dtype=self.dtype)

    def concatenate(self, obj):
        self.elements = np.concatenate((self.elements, obj.elements))
        # for e in elements:
        #    self.elements = np.concatenate((self.elements, obj.elements))
        # self.range[1] += obj.range[1] - obj.range[0]


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
        -------
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

    # Regular member functions
    def flip(self, axis):
        """Flip the elements of the object."""
        self.elements = np.flip(self.elements, axis=axis)

    def copy(self):
        """Copy the object."""
        return CircularMappingArray(self.elements.copy(), dtype=self.dtype)


class _AbsToPhyGridConverter:
    """
    An internal class that converts abstract coordinates into physical
    coordinates. Conversely, conditional operators convert physical
    coordinates into abstract coordinates.

    .. image:: ../assets/img/user_guide_abs2phy.png

    Notes
    -----
    **(Korean)** 추상 좌표를 물리 좌표로 변환하는 클래스, 조건부연산은 역변환을
    수행한다.
    """

    master = None
    """laygo2.Grid or laygo2.OneDimGrid: Coordinate system to which 
    _AbsToPhyGridConverter object belongs.

    Example
    -------
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
        -------
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
        -------
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
        -------
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
        -------
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
        -------
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
        -------
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
        -------
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
    Conversely, conditional operators convert abstract coordinates into
    physical coordinates.

    .. image:: ../assets/img/user_guide_phy2abs.png

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
    -------
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
        -------
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
        -------
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
        -------
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
        -------
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
        -------
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
        -------
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
        -------
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
        -------
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
        -------
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
        -------
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
        -------
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
        -------
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
        -------
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
        -------
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
        -------
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
        -------
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
        -------
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
    
    def left(self, obj):
        """
        Convert an object's physical left-center coordinate into abstract
        coordinate of the master grid.
        """
        return np.array([self.bottom_left(obj)[0], self.center(obj)[1]])
    
    def right(self, obj):
        """
        Convert an object's physical right-center coordinate into abstract
        coordinate of the master grid.
        """
        return np.array([self.bottom_right(obj)[0], self.center(obj)[1]])
    
    def top(self, obj):
        """
        Convert an object's physical upper-center coordinate into abstract
        coordinate of the master grid.
        """
        return np.array([self.center(obj)[0], self.top_left(obj)[1]])
    
    def bottom(self, obj):
        """
        Convert an object's physical lower-center coordinate into abstract
        coordinate of the master grid.
        """
        return np.array([self.center(obj)[0], self.bottom_left(obj)[1]])



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
    -------
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
    -------
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
    -------
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
    -------
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
        -------
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
        -------
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
        -------
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

    def flip(self):
        """Flip the elements of the object."""
        # self.elements = self.range[1]*np.ones(self.elements.shape) - np.flip(self.elements) + self.range[0]*np.ones(self.elements.shape)
        self.elements = np.flip(self.elements) * (-1) + self.range[1] + self.range[0]

    def copy(self):
        """Copy the object."""
        return OneDimGrid(self.name, self.range.copy(), elements=self.elements.copy())

    def concatenate(self, obj):
        objelem = obj.elements - obj.range[0] + self.range[1]
        self.elements = np.concatenate((self.elements, objelem))
        self.range[1] += obj.range[1] - obj.range[0]
        # for e in elements:
        #    self.elements = np.concatenate((self.elements, obj.elements))
        # self.range[1] += obj.range[1] - obj.range[0]


class Grid:
    """
    A base class having conversion operators and the mapping information (element)
    between two-dimensional physical coordinates and abstract coordinates.

    Examplar grid conversions between abstract and physical coordinates are
    summarized in the following figure.

    .. image:: ../assets/img/user_guide_grid_conversion.png


    Notes
    -----
    **(Korean)** 2차원 물리좌표와 추상좌표간 mapping 정보(element) 를 갖고 있으며
    해당 element를 활용하는 좌표 연산자를 가지고 있는 기본 클래스.
    """

    name = None
    """str: the name of the grid."""

    _xy = None
    """List[OneDimGrid]: the list contains the 1d-grid objects for x and y axes."""

    def _get_vgrid(self):
        return self._xy[0]

    def _set_vgrid(self, value):
        self._xy[0] = value

    vgrid = property(_get_vgrid, _set_vgrid)

    def _get_hgrid(self):
        return self._xy[1]

    def _set_hgrid(self, value):
        self._xy[1] = value

    hgrid = property(_get_hgrid, _set_hgrid)

    @property
    def elements(self):
        """numpy.ndarray: Two-dimensional element of a coordinate system.
            x elements, y elements

        Example
        -------
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
        -------
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
        -------
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
        -------
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
        -------
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
        -------
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
        -------
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
        -------
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
        -------
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
        -------
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
    -------
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
        -------
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
        -------
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
        -------
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
        -------
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
        -------
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
    
    def left(self, obj):
        """
        Return the abstract grid coordinates corresponding to the left
        point of obj.
        """
        return self.phy2abs.left(obj)

    def right(self, obj):
        """
        Return the abstract grid coordinates corresponding to the right
        point of obj.
        """
        return self.phy2abs.right(obj)

    def top(self, obj):
        """
        Return the abstract grid coordinates corresponding to the top
        point of obj.
        """
        return self.phy2abs.top(obj)

    def bottom(self, obj):
        """
        Return the abstract grid coordinates corresponding to the bottom
        point of obj.
        """
        return self.phy2abs.bottom(obj)

    def copy(self):
        """
        Make a copy of the current Grid object

        Returns
        -------
        laygo2.object.grid.Grid : the copied Grid object.
        
        See Also
        --------
        laygo2.object.grid.copy
        """
        name = self.name
        vgrid = self.vgrid.copy()
        hgrid = self.hgrid.copy()

        g = Grid(
            name=name,
            vgrid=vgrid,
            hgrid=hgrid,
        )
        return g

    def vflip(self, copy=True):
        """Flip the grid in vertical direction.

        Parameters
        ----------
        copy: optional, boolean
            If True, make a copy and flip the copied grid (default).
            If False, flip the current grid object.

        Returns
        --------
        laygo2.object.grid.Grid : the flipped Grid object.

        See Also
        --------
        laygo2.object.grid.vflip
        """
        if copy:
            g = self.copy()
        else:
            g = self
        g.hgrid.flip()  # Flip vertically means filpping the horizontal grid.
        return g

    def hflip(self, copy=True):
        """Flip the grid in horizontal direction.

        Parameters
        ----------
        copy: optional, boolean
            If True, make a copy and flip the copied grid (default).
            If False, flip the current grid object.

        Returns
        --------
        laygo2.object.grid.Grid : the flipped Grid object.

        See Also
        --------
        laygo2.object.grid.hflip
        """
        if copy:
            g = self.copy()
        else:
            g = self
        g.vgrid.flip()
        return g

    def vstack(self, obj, copy=True):
        """Stack grid(s) on top of the current grid in vertical direction.
        """
        if copy:
            g = self.copy()
        else:
            g = self
        if isinstance(obj, list):  # multiple stack
            obj_list = obj
        else:  # single stack
            obj_list = [obj]
        # compute the grid range first
        grid_ofst = g.hgrid.width
        for _obj in obj_list:
            g.hgrid.range[1] += _obj.hgrid.width
        # stack
        for _obj in obj_list:
            for i, h in enumerate(_obj.hgrid):
                # Check if the new grid element exist in the current grid already.
                val = (h - _obj.hgrid.range[0]) + grid_ofst
                val = val % (g.hgrid.width)  # modulo
                if not (val in g.hgrid):
                    # Unique element
                    g.hgrid.append(val + g.hgrid.range[0])
            grid_ofst += _obj.hgrid.width  # increse offset
        return g

    def hstack(self, obj, copy=True):
        """Stack grid(s) on top of the current grid in horizontal direction."""
        if copy:
            g = self.copy()
        else:
            g = self
        if isinstance(obj, list):  # Multiple stack.
            for o in obj:
                g = g.hstack(o, copy=copy)
            return g
        for i, v in enumerate(obj.vgrid):
            # Check if the new grid element exist in the current grid already.
            val = (v - obj.vgrid.range[0]) + g.vgrid.width  
            val = val % (g.vgrid.width + obj.vgrid.width)  # modulo
            if not (val in g.vgrid):
                # Unique element
                g.vgrid.append(v + g.vgrid.range[1])
        g.vgrid.range[1] += obj.vgrid.width
        return g

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
        -------
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
            self.__repr__()
            + " \n"
            + " name: "
            + self.name
            + ", \n"
            + " class: "
            + self.__class__.__name__
            + ", \n"
            + " scope: "
            + str(self.range.tolist())
            + ", \n"
            + " elements: "
            + str(self.elements)
            + ", \n"
        )


'''
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
'''
