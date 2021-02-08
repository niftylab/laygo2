import numpy as np
import pytest
import random

def _lt_1d(other, elements, width, shape):
    if isinstance(other, (int, np.integer)):
        quo = None
        mod = None
        for i, e in np.ndenumerate(elements):
            _quo = int(round(np.floor((other - e) / width)))
            if not (_quo * width + e == other):
                if quo is None:
                    quo = _quo
                    mod = i[0]
                elif _quo >= quo:  # update quo
                    quo = _quo
                    mod = i[0]
        return quo * shape + mod
    else:
       pass

def _le_1d(other, elements, width, shape):
    if isinstance(other, (int, np.integer)):
        quo = None
        mod = None
        for i, e in np.ndenumerate(elements):
            _quo = int(round(np.floor((other - e) / width)))
            if quo is None:
                quo = _quo
                mod = i[0]
            elif _quo >= quo:  # update quo
                quo = _quo
                mod = i[0]
        return quo * shape + mod
    else:
        pass

def _gt_1d(other, elements, width, shape):
    if isinstance(other, (int, np.integer)):
        quo = None
        mod = None
        for i, e in np.ndenumerate(elements):
            _quo = int(round(np.ceil((other - e) / width)))
            if not (_quo * width + e == other):
                if quo is None:
                    quo = _quo
                    mod = i[0]
                elif _quo < quo:  # update quo
                    quo = _quo
                    mod = i[0]
        return quo * shape + mod
    else:
       pass
def _ge_1d(other, elements, width, shape):
    if isinstance(other, (int, np.integer)):
        quo = None
        mod = None
        for i, e in np.ndenumerate(elements):
            _quo = int(round(np.ceil((other - e) / width)))
            if quo is None:
                quo = _quo
                mod = i[0]
            elif _quo < quo:  # update quo
                quo = _quo
                mod = i[0]
        return quo * shape + mod
    else:
        pass

def _phy2abs_operator(other, elements, width, shape, op ):

    if op == "<" :    ## max lesser
        comp = lambda e, r : e < r
        lsb_offset = 0

    elif op == "<=" : ## eq or max lesser eq
        comp = lambda e, r: e <= r
        lsb_offset = 0

    elif op == ">":    ## min greater
        comp = lambda e, r: e <=r
        lsb_offset = 1

    elif op == ">=": ## eq or min greater
        comp = lambda e, r: e < r
        lsb_offset = 1

    if isinstance(other, (int, np.integer)):
        if other > 0:
            quo_coarce = 0 + other // width
            msb_sub    = 1
        else:
            quo_coarce = 0 + other // width
            msb_sub    = 0

        remain = other % width            # positive
        msb    = quo_coarce * shape -1
        for i, e in np.ndenumerate(elements):
           # print("e: %d r:%d, m:%d, i:%d off:%d phy:%d " %(e, remain, msb + i[0], i[0], lsb_offset, quo_coarce*width + e   ))
           # print(comp( e , remain ))

            if comp( e , remain ) == True: # find maximum less then remain , e < r
                pass
            else:                          # when it is False, latest true index
                return msb + i[0] + lsb_offset

        return msb + shape + lsb_offset

def test_lt():
    #elements = [0, 35, 85, 130]
    elements = [0]
    width = 180
    shape = len(elements)
    # i = -17
    i = 360
    print( i, _lt_1d(i, elements, width, shape))

def test_phy2abs_operator():
    elements = [0, 10, 20, 40, 50]
    #elements = [0, 35, 45, 50]
    #elements = [0]
    width    = 100
    shape    = len(elements)
    #i = -17
    i  = -100
    #print("start")
    # 0 -180 -360 - 540

    print( i, _lt_1d(i, elements, width, shape), _phy2abs_operator(i, elements, width, shape, "<") )
    #print(i, _le_1d(i, elements, width, shape), _phy2abs_operator(i, elements, width, shape, "<="))
    #print(i, _gt_1d(i, elements, width, shape), _phy2abs_operator(i, elements, width, shape, ">"))

def test_phy2abs_operator_random():

    test_data = []
    test_set = [  [ [0,35,85,130], 180, 4 ],
                  [ [0,35,85,130], 140, 4 ],
                  [ [10, 35, 85, 130], 140, 4],
                  [ [0, 36, 86, 132], 140, 4],
                  ]
    #for _i in range(10000):
    #    test_data.append( random.randrange(-1000, 1000))
    test_data = range(-1000, 1000)
    for elements, width, shape in test_set:
        for i in test_data:
            assert( _lt_1d(i, elements, width, shape ) ==_phy2abs_operator(i,elements, width, shape, "<" ) ) , str(i) + str(elements) + str(width)
            assert (_le_1d(i, elements, width, shape) == _phy2abs_operator(i, elements, width, shape, "<=")) , str(i) + str(elements) + str(width)
            assert (_gt_1d(i, elements, width, shape) == _phy2abs_operator(i, elements, width, shape, ">")) , str(i) + str(elements) + str(width)
            assert (_ge_1d(i, elements, width, shape) == _phy2abs_operator(i, elements, width, shape, ">=")) , str(i) + str(elements) + str(width)

    test_set = [[ [0],180,1 ] ]
    for elements, width, shape in test_set:
        for i in test_data:
            #assert( _lt_1d(i, elements, width, shape ) ==_phy2abs_operator(i,elements, width, shape, "<" ) ) , str(i) + str(elements) + str(width)
            assert (_le_1d(i, elements, width, shape) == _phy2abs_operator(i, elements, width, shape, "<=")) , str(i) + str(elements) + str(width)
            #assert (_gt_1d(i, elements, width, shape) == _phy2abs_operator(i, elements, width, shape, ">")) , str(i) + str(elements) + str(width)
            assert (_ge_1d(i, elements, width, shape) == _phy2abs_operator(i, elements, width, shape, ">=")) , str(i) + str(elements) + str(width)
