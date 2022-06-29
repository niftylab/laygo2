import pytest
import numpy as np
import pprint
import laygo2

import test_tech as tech


def bb(a: int, b: int, c: int, d: int):
    return (np.asarray([[a, b], [c, d]]))

def test_grid_OneDimGrid():
    """
    DUT     : laygo2.object.grid.OneDimGrid
    Comment : test conversion operator
    Case    :  <, >, <=, >=
    """

    g1_o  = laygo2.object.grid.OneDimGrid(name='mygrid', scope=[0, 100], elements=[0, 10, 20, 40, 50])
    g1_no = laygo2.object.grid.OneDimGrid(name='mygrid', scope=[0, 200], elements=[10, 20, 25] )
    g1_e0 = laygo2.object.grid.OneDimGrid(name='mygrid', scope=[0, 180], elements=[0] )
    print("Testing 1d grid")

    test_set_index=[
        [ g1_o[:9],  np.array([0, 10, 20, 40, 50, 100, 110, 120, 140])],
        [ g1_o[1:] ,  np.array([10, 20, 40, 50]) ],
        [ g1_o[-6:10], np.array([-150, -100, -90, -80, -60, -50, 0, 10, 20, 40, 50, 100, 110, 120, 140, 150])],
        [ g1_o[ [3, 5, 7]], np.array([40, 100, 120]) ],

        [g1_no[:10], np.array([10, 20, 25, 210, 220, 225, 410, 420, 425,610])],
        [g1_no[1:], np.array([20, 25])],
        [g1_no[-6:10], np.array([
            -390, -380,-375,-190, -180, -175,
            10, 20, 25, 210, 220, 225, 410, 420, 425, 610
        ])],
        [g1_no[[3,5,7]], np.array([ 210,225, 420 ])],

        [g1_e0[:10], np.array([0, 180, 360, 540, 720, 900, 1080, 1260, 1440,1620])],
        [g1_e0[1:], np.array([180])],
        [g1_e0[-6:10], np.array([-1080, -900 ,-720 ,-540, -360,-180,0 , 180, 360, 540, 720, 900, 1080, 1260, 1440, 1620])],
        [g1_e0[ [3,5,7]], np.array([ 540, 900, 1260 ])],

    ]

    test_set_op_o=[
        # matched
        [g1_o == [-100, -80, -60, 0, 10, 40, 100, 110, 140], np.array([-5,-3,-2,0,1,3,5,6,8])],
        [g1_o < -60, -3], [g1_o < 0, -1], [g1_o < 120, 6],
        [g1_o < -100, -6], [g1_o < 100, 4],


        [g1_o <= -60, -2], [g1_o <= 0, 0], [g1_o <= 120, 7],
        [g1_o <= -100, -5], [g1_o <= 100, 5],

        [g1_o > -60, -1], [g1_o > 0, 1], [g1_o > 120, 8],
        [g1_o > -100, -4], [g1_o > 100, 6],

        [g1_o >= -60, -2], [g1_o >= 0, 0], [g1_o >= 120, 7],
        [g1_o >= -100,-5 ], [g1_o >= 100, 5],

        # between
        [g1_o == [-95, -75, 5, 15, 35, 105], np.array([None,None,None,None,None,None])],
        [g1_o < -95, -5], [g1_o < 45, 3], [g1_o < 105, 5],
        [g1_o <= -95, -5], [g1_o <= 45, 3], [g1_o <= 105, 5],
        [g1_o > -95, -4], [g1_o > 45, 4], [g1_o > 105, 6],
        [g1_o >= -95, -4], [g1_o >= 45, 4], [g1_o >= 105, 6],

        # upper out
        [g1_o == [-40, 60], np.array([None, None])],
        [g1_o < -40, -1], [g1_o < 60, 4],
        [g1_o <= -40, -1], [g1_o <= 60, 4],
        [g1_o > -40, 0], [g1_o > 60, 5],
        [g1_o >= -40, 0], [g1_o >= 60, 5],
        ]
    test_set_op_no = [
        # matched
        [g1_no == [-190, -180, -175, 10, 20, 25, 210, 220, 225], np.array([-3,-2,-1,0,1,2,3,4,5])],
        [g1_no < -190, -4], [g1_no < -175, -2], [g1_no < 10, -1], [g1_no < 25, 1],
        [g1_no < 210, 2], [g1_no < 225, 4],

        [g1_no <= -190, -3], [g1_no <= -175, -1], [g1_no <= 10, 0], [g1_no <= 25, 2],
        [g1_no <= 210, 3], [g1_no <= 225, 5],

        [g1_no > -190, -2], [g1_no > -175, 0], [g1_no > 10, 1], [g1_no > 25, 3],
        [g1_no > 210, 4], [g1_no > 225, 6],

        [g1_no >= -190, -3], [g1_no >= -175, -1], [g1_no >= 10, 0], [g1_no >= 25, 2],
        [g1_no >= 210, 3], [g1_no >= 225, 5],

        # between
        [g1_no == [-195, -185, -170, 15, 26, 30, 215], np.array([ None, None, None, None, None, None, None ])],
        [g1_no < -185, -3], [g1_no < 15, 0], [g1_no < 215, 3],
        [g1_no <= -185, -3], [g1_no <= 15, 0], [g1_no <= 215, 3],
        [g1_no > -185, -2], [g1_no > 15, 1], [g1_no > 215, 4],
        [g1_no >= -185, -2], [g1_no >= 15, 1], [g1_no >= 215, 4],

        # upper out
        [g1_no == [ -40,190], np.array([None, None])],
        [g1_no < -40, -1], [g1_no < 190, 2],
        [g1_no <= -40, -1], [g1_no <= 190, 2],
        [g1_no > -40, 0], [g1_no > 190, 3],
        [g1_no >= -40, 0], [g1_no >= 190, 3],

        # lower out
        [g1_no == [5, 205], np.array([None, None])],
        [g1_no < 5, -1], [g1_no < 205, 2],
        [g1_no <= 5, -1], [g1_no <= 205, 2],
        [g1_no > 5, 0], [g1_no > 205, 3],
        [g1_no >= 5, 0], [g1_no >= 205, 3],
    ]
    test_set_op_e0=[
        # matched
        [g1_e0 == [-180,0,180], np.array([ -1,0,1])],
        [g1_e0 <  -180,  -2], [g1_e0 <  0, -1], [g1_e0 < 180, 0],
        [g1_e0 <= -180, -1], [g1_e0 <= 0, 0],   [g1_e0 <= 180, 1],

        [g1_e0 >  -180,   0], [g1_e0 >  0, 1], [g1_e0 > 180, 2],
        [g1_e0 >= -180, -1], [g1_e0 >= 0, 0],   [g1_e0 >= 180, 1],

        [g1_e0 == [-170, 10, 170,190], np.array([None,None,None,None])],
        [g1_e0 < -170, -1], [g1_e0 < -10, -1], [g1_e0 < 170, 0], [g1_e0 < 190, 1],
        [g1_e0 <= -170, -1],[g1_e0 <= -10, -1], [g1_e0 <= 170, 0], [g1_e0 <= 190, 1],
        [g1_e0 > -170, 0],  [g1_e0 > -10, 0], [g1_e0 > 170, 1],[g1_e0 > 190, 2],
        [g1_e0 >= -170, 0], [g1_e0 >= -10, 0], [g1_e0 >= 170, 1],[g1_e0 >= 190, 2],

    ]


    i=0
    for q, a in test_set_index:
        i = i+1
        assert  np.all(q == a) == True , str("index ") + str(q) + str("  ") +  str(a) + str("  ") + str(i)
    i = 0
    for q, a in test_set_op_o:
        i = i + 1
        assert np.all(q == a) == True, str("ori ") + str(q) + str("  ") + str(a) + str("  ") + str(i)
    i = 0
    for q, a in test_set_op_no:
        i = i + 1
        assert np.all(q == a) == True, str("no ") + str(q) + str("  ") + str(a) + str("  ") + str(i)
    i = 0
    for q, a in test_set_op_e0:
        i = i + 1
        assert np.all(q == a) == True, str("e0 ") + str(q) + str("  ") + str(a) + str("  ") + str(i)

def test_grid_OneDimGrid_compare():
    """
    DUT     : laygo2.object.grid.OneDimGrid
    Comment : compare old conversion operator function and new
    Case    :  <, >, <=, >=
    """
    # compare older function vs new
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

    test_set = [[[0, 35, 85, 130], 180, 4],
                [[0, 35, 85, 130], 140, 4],
                [[10, 35, 85, 130], 140, 4],
                [[0, 36, 86, 132], 140, 4],
                ]


    test_data = range(-1000, 1000)
    for elements, width, shape in test_set:
        for i in test_data:
            assert (_lt_1d(i, elements, width, shape) == laygo2.object.grid._AbsToPhyGridConverter._phy2abs_operator(i, elements, width, shape, "<")), str(
                i) + str(elements) + str(width)
            assert (_le_1d(i, elements, width, shape) == laygo2.object.grid._AbsToPhyGridConverter._phy2abs_operator(i, elements, width, shape, "<=")), str(
                i) + str(elements) + str(width)
            assert (_gt_1d(i, elements, width, shape) == laygo2.object.grid._AbsToPhyGridConverter._phy2abs_operator(i, elements, width, shape, ">")), str(
                i) + str(elements) + str(width)
            assert (_ge_1d(i, elements, width, shape) == laygo2.object.grid._AbsToPhyGridConverter._phy2abs_operator(i, elements, width, shape, ">=")), str(
                i) + str(elements) + str(width)

    test_set = [[[0], 180, 1]]
    for elements, width, shape in test_set:
        for i in test_data:
            # assert( _lt_1d(i, elements, width, shape ) ==_phy2abs_operator(i,elements, width, shape, "<" ) ) , str(i) + str(elements) + str(width)
            assert (_le_1d(i, elements, width, shape) == laygo2.object.grid._AbsToPhyGridConverter._phy2abs_operator(i, elements, width, shape, "<=")), str(
                i) + str(elements) + str(width)
            # assert (_gt_1d(i, elements, width, shape) == _phy2abs_operator(i, elements, width, shape, ">")) , str(i) + str(elements) + str(width)
            assert (_ge_1d(i, elements, width, shape) == laygo2.object.grid._AbsToPhyGridConverter._phy2abs_operator(i, elements, width, shape, ">=")), str(
                i) + str(elements) + str(width)

def test_grid_Grid():
    """
    DUT     : laygo2.object.grid.Grid
    Comment : test Grid Class
    Case    :  <, >, <=, >=, property
    """

    print("Testing 2d grid")
    g1_o  = laygo2.object.grid.OneDimGrid(name='mygrid', scope=[0, 100], elements=[0, 10, 20, 40, 50])
    g1_no = laygo2.object.grid.OneDimGrid(name='mygrid', scope=[0, 200], elements=[10, 20, 25])
    g2    = laygo2.object.grid.Grid(name='mygrid', vgrid=g1_o, hgrid=g1_no)

    test_set_index = [
        [g2[2, 5], np.array([20, 225])],
        [g2[2, :], np.array([[20, 10], [20, 20], [20, 25]])],
        [g2[0:8, 1:3], np.array([[[0, 20], [0, 25]], [[10, 20], [10, 25]], [[20, 20], [20, 25]], [[40, 20], [40, 25]],
                                [[50, 20], [50, 25]], [[100, 20], [100, 25]], [[110, 20], [110, 25]],
                                [[120, 20], [120, 25]]])],

        [g2 == [20, 10], np.array([2, 0])],
        [g2 == [[20, 10], [40, 25]], np.array([[2, 0], [3, 2]])],
        [g2.phy2abs[[20, 40], [10, 25]], np.array([[[2, 0], [2, 2]], [[3, 0], [3, 2]]])]
    ]

    test_set_op = [
        [ g2 <  [ 20,10 ],  np.array( [1, -1 ] )  ],
        [ g2 <  [ 25, 15],  np.array( [2, 0  ] )  ],
        [ g2  <  [125, 215], np.array( [7, 3])],

        [g2 <= [20, 10], np.array([2, 0])],
        [g2 <= [25, 15], np.array([2, 0])],
        [g2 <= [125, 215], np.array([7, 3])],

        [g2 > [20, 10], np.array([3, 1])],
        [g2 > [25, 15], np.array([3, 1])],
        [g2 > [125, 215], np.array([8, 4])],

        [g2 >= [20, 10], np.array([2, 0])],
        [g2 >= [25, 15], np.array([3, 1])],
        [g2 >= [125, 215], np.array([8, 4])],
    ]

    test_set_listOp = [
        [ g2 <  [ [20,10], [25,15], [125,215] ] , [ [1,-1], [2,0], [7,3] ] ],
        [g2  <= [[20, 10], [25, 15], [125, 215]], [[2, 0], [2, 0], [7, 3]]],
        [g2  > [[20, 10], [25, 15], [125, 215]], [[3, 1], [3, 1], [8, 4]]],
        [g2 >= [[20, 10], [25, 15], [125, 215]], [[2, 0], [3, 1], [8, 4]]],
    ]

    test_set_property=[
        [ g2.elements, [ [0,10,20,40,50], [10,20,25] ] ],
        [ g2.width, 100],
        [ g2.height, 200]

    ]
    test_case_map = test_set_index + test_set_op + test_set_listOp

    for q, a  in test_case_map:
        assert  np.all( q == a)
    for q, a in test_set_property:
        assert q, a

def test_grid_RoutingGrid():
    """
    DUT     : laygo2.object.grid.RoutingGrid(Grid)
    Comment : TBD
    Case    : TBD
    """

    from laygo2.object.template import NativeInstanceTemplate
    g1a = laygo2.object.grid.OneDimGrid(name='mygrid', scope=[0, 100], elements=[0, 10, 20, 40, 50])
    g1b = laygo2.object.grid.OneDimGrid(name='mygrid', scope=[0, 200], elements=[10, 20, 25])

    vwidth     = laygo2.object.grid.CircularMapping(elements=[2, 2, 2, 5, 5])
    hwidth     = laygo2.object.grid.CircularMapping(elements=[4, 4, 4])
    vextension = laygo2.object.grid.CircularMapping(elements=[1, 1, 1, 1, 1])
    hextension = laygo2.object.grid.CircularMapping(elements=[1, 1, 1])
    vlayer     = laygo2.object.grid.CircularMapping(elements=[['M1', 'drawing']] * 5, dtype=object)
    hlayer     = laygo2.object.grid.CircularMapping(elements=[['M2', 'drawing']] * 3, dtype=object)
    pvlayer    = laygo2.object.grid.CircularMapping(elements=[['M1', 'drawing']] * 5, dtype=object)
    phlayer    = laygo2.object.grid.CircularMapping(elements=[['M2', 'drawing']] * 3, dtype=object)
    tvia0  = NativeInstanceTemplate( libname='mylib', cellname='mynattemplate', bbox=[[0, 0], [100, 100]] )
    viamap = laygo2.object.grid.CircularMappingArray(elements=np.array([ [tvia0] * 5] * 3), dtype=object)
    # ivia0 = tvia0.generate(name='IV0', xy=[0, 0], transform='R0')
    # via_inst = ivia0
    # viamap = CircularMapping(elements=np.array([[via_inst]*5]*3), dtype=object)
    # g3 = RoutingGrid(name='mygrid', x=g1a, y=g1b, widthmap=width, layermap=layer, viamap=viamap)
    g3 = laygo2.object.grid.RoutingGrid(name='mygrid', vgrid=g1a, hgrid=g1b, vwidth=vwidth, hwidth=hwidth,
                                        vextension=vextension,
                                        hextension=hextension, vlayer=vlayer, hlayer=hlayer, viamap=viamap,
                                        pin_vlayer=vlayer, pin_hlayer=hlayer,
                                        xcolor=["Not MPT"], ycolor=["Not MPT"]
                                        )
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
    #    #print(q, np.all(q == a))
        assert (np.all(q == a))

def test_grid_route_via_track():
    """
    DUT     : laygo2.object.grid.route_via_track()
    Comment : test RoutingGrid.rount_via_track()
    Case    :  Center-track, Upper-track, Lower-track, Overlap-track, disorder-track
    """

    tpmos_name, tnmos_name = 'pmos', 'nmos'
    pg_name = 'placement_cmos'
    libname = 'laygo2_test_1'
    templates = tech.load_templates()
    tpmos, tnmos = templates[tpmos_name], templates[tnmos_name]

    grids = tech.load_grids(templates=templates)
    pg, r12, r23 = grids['placement_cmos'], grids['routing_12_cmos'], grids['routing_23_cmos']

    lib = laygo2.object.database.Library(name=libname)
    dsn = laygo2.object.database.Design(name="test", libname=libname)
    lib.append(dsn)

    def _compare(dsn, grid, mn, track, route_golden, via_golden):
        route_golden = np.asarray(route_golden)
        via_golden = np.asarray(via_golden)

        dsn.route_via_track(grid=grid, track=track, mn=mn)
        route, via = [], []

        for name, i in dsn.rects.items():
            route.append(grid.mn(i.bbox))

        for name, i in dsn.virtual_instances.items():
            via.append(grid.mn(i.bbox))

        if np.array_equal(route_golden, np.asarray(route)) & np.array_equal(via_golden, np.asarray(via)) != True:
            print("log")
            print(route_golden)
            print(route)
            print(via_golden)
            print(via)

        return (np.array_equal(route_golden, np.asarray(route)) & np.array_equal(via_golden, np.asarray(via)))

    _mn, _track, _route_g, _via_g = [], [], [], []  ## input, input, expected, expected

    _mn.append([[0, 0], [0, 2], [2, 2]])  ## 1. Center track
    _track.append([None, 1])
    _route_g.append([bb(0, 0, 0, 1), bb(0, 1, 0, 2), bb(2, 1, 2, 2), bb(0, 1, 2, 1)])
    _via_g.append([bb(0, 1, 0, 1), bb(0, 1, 0, 1), bb(2, 1, 2, 1)])

    _mn.append([[0, 0], [0, 2], [2, 2]])  ## 2. Upper Track
    _track.append([None, 4])
    _route_g.append([bb(0, 0, 0, 4), bb(0, 2, 0, 4), bb(2, 2, 2, 4), bb(0, 4, 2, 4)])
    _via_g.append([bb(0, 4, 0, 4), bb(0, 4, 0, 4), bb(2, 4, 2, 4)])

    _mn.append([[0, 0], [0, 2], [2, 2]])  ## 3. Lower Track
    _track.append([None, -2])
    _route_g.append([bb(0, -2, 0, 0), bb(0, -2, 0, 2), bb(2, -2, 2, 2), bb(0, -2, 2, -2)])
    _via_g.append([bb(0, -2, 0, -2), bb(0, -2, 0, -2), bb(2, -2, 2, -2)])

    _mn.append([[0, 0], [0, 2], [2, 2]])  ## 4. overlap Track
    _track.append([None, 0])
    _route_g.append([bb(0, 0, 0, 2), bb(2, 0, 2, 2), bb(0, 0, 2, 0)])
    _via_g.append([bb(0, 0, 0, 0), bb(0, 0, 0, 0), bb(2, 0, 2, 0)])

    _mn.append([[2, 2], [0, 0], [0, 2]])  ## 5. sequence Track
    _track.append([None, 1])
    _route_g.append([bb(2, 1, 2, 2), bb(0, 0, 0, 1), bb(0, 1, 0, 2), bb(0, 1, 2, 1)])
    _via_g.append([bb(2, 1, 2, 1), bb(0, 1, 0, 1), bb(0, 1, 0, 1)])

    for i in range(len(_mn)):
        assert (_compare(dsn, r23, _mn[i], _track[i], _route_g[i], _via_g[i]))
        dsn.rects.clear()
        dsn.virtual_instances.clear()

def test_database_design_bbox():
    templates = tech.load_templates()
    grids     = tech.load_grids(templates)
    pg        = grids["placement_basic"]

    dsn   = laygo2.object.database.Design(name="test")
    inst0 = laygo2.object.physical.Instance(name="I0", xy=[0, 0], libname="mylib", cellname="mycell", unit_size=[100, 100], pins=None,
                                            transform='R0')
    inst1 = laygo2.object.physical.Instance(name="I1", xy=[0, 0], libname="mylib", cellname="mycell",
                                            unit_size=[100, 100], pins=None,
                                            transform='R0')
    dsn.place(grid=pg, inst= inst0, mn=[0,  0])
    dsn.place(grid=pg, inst= inst1, mn=[10, 0])

    print(dsn.bbox)
    assert( np.array_equal( dsn.bbox, [[0,0],[200,100]]))

def test_template_user():
    vinst0_pins = dict()
    pin0 = laygo2.object.physical.Pin(xy=[[0, 0], [10, 10]], layer=['M1', 'drawing'], netname='in')
    pin1 = laygo2.object.physical.Pin(xy=[[20, 20], [30, 30]], layer=['M1', 'drawing'], netname='in')
    pin2 = laygo2.object.physical.Pin(xy=[[40, 40], [50, 50]], layer=['M1', 'drawing'], netname='in')
    vinst0_pins['in'] = laygo2.object.physical.Pin(xy=[[0, 0], [10, 10]], layer=['M1', 'drawing'], netname='in', elements= [pin0,pin1,pin2])

    vinst0_native_elements = dict()
    vinst0_native_elements['R0'] = laygo2.object.physical.Rect(xy=[[0, 0], [10, 10]], layer=['M1', 'drawing'])
    #vinst0_native_elements['R1'] = laygo2.object.physical.Rect(xy=[[90, 90], [100, 100]], layer=['M1', 'drawing'])
    #vinst0_native_elements['R2'] = laygo2.object.physical.Rect(xy=[[0, 0], [100, 100]], layer=['prBoundary', 'drawing'])
    print("Create V Instance")
    vinst0 = laygo2.object.physical.VirtualInstance(name='I0', libname='mylib', cellname='myvcell', xy=[100, 100],
                                                    native_elements=vinst0_native_elements, unit_size=[100, 100], pins=vinst0_pins, transform='MX')

    print("END V Instance")
    print(vinst0.xy)
    vinst1 = laygo2.object.physical.VirtualInstance(name='I1', libname='mylib', cellname='myvcell', xy=[100, 100],
                                                    native_elements=vinst0_native_elements, unit_size=[100, 100],
                                                    pins=vinst0_pins, transform='R0')

    vinst0.xy=[500,500]
    print("compare")
    for i in range(3):
#        print(vinst1.pins["in"][i].xy)
        print(vinst0.pins["in"][i].xy)
    pass
"""
def test_database_design_test():

    g1_y     = laygo2.object.grid.OneDimGrid(name='mygrid', scope=[0, 50], elements=[0, 10, 20, 30, 40])
    g1_x_cut = laygo2.object.grid.OneDimGrid(name='mygrid', scope=[0, 5], elements=[0])
    g1_x     = laygo2.object.grid.OneDimGrid(name='mygrid', scope=[0, 10], elements=[0])

    templates = tech.load_templates()
    grids     = tech.load_grids(templates)
    pg        = grids["placement_cmos"]
    viamap    = 5*["via_r23_default"]
    vmap_mapped =[]
    for via_name in viamap:
        vmap_mapped.append( templates[ via_name ] )
    vmap_mapped = np.asarray([vmap_mapped])
    vmap_mapped2 = laygo2.object.grid.CircularMappingArray(elements=vmap_mapped, dtype=object)

    grid_cut = laygo2.object.grid.RoutingGrid(name='test', vgrid=g1_x_cut, hgrid=g1_y, viamap=vmap_mapped2
                                              ,vwidth=np.zeros(4), hwidth=[0],
                                              vlayer=None, hlayer=None, pin_vlayer=None, pin_hlayer=None,
                                              vextension=None, hextension=None )

    grid     = laygo2.object.grid.RoutingGrid(name='test', vgrid=g1_x, hgrid=g1_y, viamap=vmap_mapped2
                                              ,vwidth=np.zeros(4), hwidth=[0],
                                              vlayer=None, hlayer=None, pin_vlayer=None, pin_hlayer=None,
                                              vextension=None, hextension=None )

    dsn = laygo2.object.database.Design_test(name="test")
    r     = grid
    r_cut = grid_cut

    bbox_top  = np.array( [ [0,0], [300,290] ] ) ## instance bbox union


    rect    = [0]*7
    rect_er = [0]*4
    rect[0] = laygo2.object.physical.Rect(xy=[ [0, 10], [30, 10]],   layer=['M0', 'drawing'], netname='net0', hextension = 5, vextension = 5)
    rect[1] = laygo2.object.physical.Rect(xy=[ [0, 10], [30, 10]],   layer=['M0', 'pin'],     netname='net0', hextension = 5, vextension = 5)

    # center space error
    rect[2]    = laygo2.object.physical.Rect(xy=[[60,  40],  [90, 40]], layer=['M0', 'drawing'], netname='net0',hextension = 5, vextension = 5)
    rect_er[0] = laygo2.object.physical.Rect(xy=[[120, 40],  [150, 40]], layer=['M0', 'drawing'], netname='er0',hextension = 5, vextension = 5)
    rect[3]    = laygo2.object.physical.Rect(xy=[[210,  40], [240, 40]], layer=['M0', 'drawing'], netname='net0',hextension = 5, vextension = 5)

    # center & edge space error
    rect[4]    = laygo2.object.physical.Rect(xy=[[60, 60], [90, 60]], layer=['M0', 'drawing'], netname='net0',hextension = 5, vextension = 5)
    rect_er[1] = laygo2.object.physical.Rect(xy=[[120, 60], [150, 60]], layer=['M0', 'drawing'], netname='er1',hextension = 5, vextension = 5 )
    rect_er[2] = laygo2.object.physical.Rect(xy=[[180, 60], [270, 60]], layer=['M0', 'drawing'], netname='er2',hextension = 5, vextension = 5)

    # edge with pin & overlap
    rect[5]    = laygo2.object.physical.Rect(xy=[[210, 90], [240, 90]], layer=['M0', 'drawing'], netname='net0',hextension = 5, vextension = 5)
    rect[6] = laygo2.object.physical.Rect(xy=[[210, 90], [270, 90]], layer=['M0', 'pin'], netname='net0',hextension = 5, vextension = 5)



    vinst0_native_elements = dict()
    vinst0_native_elements['R0'] = laygo2.object.physical.Rect(xy=[[ 200, 100], [220, 100]], layer=['M0', 'drawing'])

    vinstr0 = laygo2.object.physical.VirtualInstance(name='I0', libname='mylib', cellname='myvcell', xy=[0, 0],
                                                     native_elements=vinst0_native_elements,
                                                     unit_size=[300, 300], pins=None, transform='R0')

    vinstmx = laygo2.object.physical.VirtualInstance(name='I1', libname='mylib', cellname='myvcell', xy=[0, 0],
                                                     native_elements=vinst0_native_elements,
                                                     unit_size=[300, 300], pins=None, transform='MX')

    rect_er[3] = laygo2.object.physical.Rect(xy=[[100, 100], [180, 100]], layer=['M0', 'drawing'], netname='er2',
                                             hextension=5, vextension=5)

    dsn.place(grid=pg, inst=[ vinstr0, vinstmx] , mn=[0, 0])

    for i, obj in enumerate( rect + rect_er):
        dsn.append(obj)

    print("start!")

    dsn.rect_space("M0", xy = bbox_top, grid=r, grid_cut = r_cut, space_min = 50 )
    result=[]

    golden=[[105,40],[105,60],[165,60],[190,100],[300,60]]
    for name, obj in dsn.virtual_instances.items():
        if "contact" in obj.native_elements:
            result.append(obj.xy)

    result=np.unique( np.array(result), axis=0 )

    assert(np.array_equal(result, golden))
"""
def test_physical_virtualInstace_get_element_position():
    dsn = laygo2.object.database.Design(name="test")
    templates = tech.load_templates()
    grids = tech.load_grids(templates=templates)
    pg = grids["placement_cmos"]
    vinst0_pins = dict()
    vinst0_pins['in'] = laygo2.object.physical.Pin(xy=[[0, 100], [100, 100]], layer=['M1', 'drawing'], netname='in')
    vinst0_native_elements = dict()
    vinst0_native_elements['R0'] = laygo2.object.physical.Rect(xy=[[ 50, 100], [400 , 100]], layer=['M1', 'drawing'])
    vinstr0 = laygo2.object.physical.VirtualInstance(name='I0', libname='mylib', cellname='myvcell', xy=[0, 0],
                                                    native_elements=vinst0_native_elements,
                                                    unit_size=[500, 500], pins=vinst0_pins, transform='R0')
    vinstmx = laygo2.object.physical.VirtualInstance(name='I1', libname='mylib', cellname='myvcell', xy=[0, 0],
                                                    native_elements=vinst0_native_elements,
                                                    unit_size=[500, 500], pins=vinst0_pins, transform='MX')
    vinstmy = laygo2.object.physical.VirtualInstance(name='I2', libname='mylib', cellname='myvcell', xy=[0, 0],
                                                    native_elements=vinst0_native_elements,
                                                    unit_size=[500, 500], pins=vinst0_pins, transform='MY')
    vinstr90 = laygo2.object.physical.VirtualInstance(name='I3', libname='mylib', cellname='myvcell', xy=[0, 0],
                                                    native_elements=vinst0_native_elements,
                                                    unit_size=[500, 500], pins=vinst0_pins, transform='R90')

    insts = [vinstr0, vinstmx, vinstmy, vinstr90 ]
    dsn.place(grid=pg, inst= insts, mn=[0,0] )

    print("start")
    for inst in insts:
        print(inst.xy, inst.transform)
    setr0 = np.array( [ [50,  100], [400, 100 ] ])
    setmx= np.array(  [ [550,  400], [900, 400 ] ]) # y
    setmy = np.array( [ [1100, 100], [1450, 100 ] ])
    setr90 = np.array( [[1600, 400], [1950, 400 ] ]) # y


    assert( np.array_equal( setr0, vinstr0.get_element_position( vinstr0.native_elements["R0"] ) ) )
    assert (np.array_equal( setmx, vinstmx.get_element_position(vinstmx.native_elements["R0"])))
    assert (np.array_equal( setmy, vinstmy.get_element_position(vinstmy.native_elements["R0"])))
    #assert (np.array_equal( setr90, dsn.get_xy_virtualInstance(vinstr90, vinstr90.native_elements["R0"])))






