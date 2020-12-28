import pytest
import numpy as np
import pprint
import laygo2

import test_tech as tech


def bb(a: int, b: int, c: int, d: int):
    return (np.asarray([[a, b], [c, d]]))

@pytest.fixture
def setup():
    tpmos_name, tnmos_name = 'pmos', 'nmos'
    pg_name = 'placement_cmos'
    libname = 'laygo2_test_1'

    templates    = tech.load_templates()
    tpmos, tnmos = templates[tpmos_name], templates[tnmos_name]

    grids = tech.load_grids(templates=templates)
    pg, r12, r23 = grids['placement_cmos'], grids['routing_12_cmos'], grids['routing_23_cmos']

    lib = laygo2.object.database.Library(name=libname)
    dsn = laygo2.object.database.Design(name="test", libname=libname)
    lib.append(dsn)
    r_list={ "tpmos":tpmos, "tnmos":tnmos, "pg":pg, "r23":r23, "dsn":dsn}
    return(r_list)


def test_grid_route_via_track(setup):
    """ test laygo2.object.grid.route_via_track()
    """
    def _compare(dsn, grid, mn, track, route_golden, via_golden):
        route_golden = np.asarray(route_golden)
        via_golden   = np.asarray(via_golden)

        dsn.route_via_track(grid= grid, track= track, mn= mn)
        route, via = [], []

        for name, i in dsn.rects.items():
            route.append(grid.mn(i.bbox))

        for name, i in dsn.virtual_instances.items():
            via.append(grid.mn(i.bbox))

        if np.array_equal( route_golden, np.asarray(route)) & np.array_equal( via_golden, np.asarray(via) ) != True:
            print("log")
            print(route_golden)
            print(route)
            print(via_golden)
            print(via)

        return( np.array_equal( route_golden, np.asarray(route)) & np.array_equal( via_golden, np.asarray(via) ) )

    ### Test Case: 1.Center -> 2.Upper -> 3.Lower -> 4.overlap -> 5.sequence
    r23 = setup["r23"]
    dsn = setup["dsn"]
    _mn, _track, _route_g, _via_g = [], [], [], [] ## input, input, expected, expected

    _mn.append(      [ [0, 0], [0, 2], [2, 2] ])   ## 1. Center track
    _track.append(   [ None, 1 ])
    _route_g.append( [ bb(0, 0, 0, 1), bb(0, 1, 0, 2), bb(2, 1, 2, 2), bb(0, 1, 2, 1) ])
    _via_g.append(   [ bb(0, 1, 0, 1), bb(0, 1, 0, 1), bb(2, 1, 2, 1) ] )

    _mn.append(      [ [0, 0], [0, 2], [2, 2] ])   ## 2. Upper Track
    _track.append(   [ None, 4 ])
    _route_g.append( [ bb(0, 0, 0, 4), bb(0, 2, 0, 4), bb(2, 2, 2, 4), bb(0, 4, 2, 4)] )
    _via_g.append(   [ bb(0, 4, 0, 4), bb(0, 4, 0, 4), bb(2, 4, 2, 4)])

    _mn.append(      [ [0, 0], [0, 2], [2, 2]])    ## 3. Lower Track
    _track.append(   [None, -2])
    _route_g.append( [bb(0, -2, 0, 0), bb(0, -2, 0, 2), bb(2, -2, 2, 2), bb(0, -2, 2, -2)])
    _via_g.append(   [bb(0, -2, 0, -2), bb(0, -2, 0, -2), bb(2, -2, 2, -2)])

    _mn.append(      [ [0, 0], [0, 2], [2, 2]])    ## 4. overlap Track
    _track.append(   [ None, 0])
    _route_g.append( [bb(0, 0, 0, 2), bb(2, 0, 2, 2), bb(0, 0, 2, 0)])
    _via_g.append(   [bb(0, 0, 0, 0), bb(0, 0, 0, 0), bb(2, 0, 2, 0)])

    _mn.append([     [2, 2], [0, 0], [0, 2]])           ## 5. sequence Track
    _track.append(   [None, 1])
    _route_g.append( [bb(2, 1, 2, 2), bb(0, 0, 0, 1), bb(0, 1, 0, 2), bb(0,1,2,1) ])
    _via_g.append(   [bb(2, 1, 2, 1), bb(0, 1, 0, 1), bb(0, 1, 0, 1)])

    for i in range( len(_mn) ):
        assert( _compare( dsn, r23, _mn[i], _track[i], _route_g[i], _via_g[i]) )
        dsn.rects.clear()
        dsn.virtual_instances.clear()





