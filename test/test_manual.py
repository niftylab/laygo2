import pytest
import numpy as np
import pprint
import laygo2

import test_tech as tech

def test_grid_manual_convertion_table():
    g1   = laygo2.object.grid.OneDimGrid(name='xgrid', scope=[0, 100], elements=[0, 10, 20, 40, 50])
    g1_y = laygo2.object.grid.OneDimGrid(name='ygrid', scope=[0, 100], elements=[10, 20, 40, 50, 60])
    g2 = laygo2.object.grid.Grid(name='test', vgrid=g1, hgrid=g1_y)

    set_op1 = ("[]", "()", "==", "<", "<=", ">", ">=")

    g1ap, g1pa = g1.abs2phy, g1.phy2abs

    set_g1         = ( g1[10], "dummy" , g1 == 10 , g1 < 10 , g1 <=10 , g1 > 10 , g1 >= 10  )
    set_g1_abs2phy = ( g1ap[10], g1ap(10), g1ap ==10, g1ap < 10 , g1ap <=10, g1ap >10, g1ap >=10 )
    set_g1_phy2abs = ( g1pa[10], g1pa(10), g1pa ==10, g1pa < 10 , g1pa <=10, g1pa >10, g1pa >=10 )

    set_g1_title = ("g1", "g1.abs2phy", "g1.phy2abs")
    set_g1_loop  = (set_g1, set_g1_abs2phy, set_g1_phy2abs)

    g2x, g2y, g2xy, g2m, g2n, g2mn  = g2.x, g2.y, g2.xy, g2.m, g2.n, g2.mn

    set_g2    = ( g2[[10,10]]   , "dummy" , g2 == [ 10,10 ], g2 < [10,10 ] , g2 <= [10,10 ], g2 > [10,10 ], g2 >= [10,10 ] )
    set_g2_x  = ( g2x[10], g2x(10), g2x ==10 , g2x < 10 , g2x <= 10, g2x > 10 , g2x >= 10 )
    set_g2_y  = ( g2y[10], g2y(10), g2y ==10 , g2y < 10 , g2y <= 10, g2y > 10 , g2y >= 10 )
    set_g2_xy = ( g2xy[[10,10]] , g2xy( [10,10] )   , g2xy == [ 10,10 ], g2xy < [10,10 ] , g2xy <= [10,10 ], g2xy > [10,10 ], g2xy >= [10,10 ] )

    set_g2_m  = (g2m[10], g2m(10), g2m ==10 , g2m < 10 , g2m <= 10, g2m > 10 , g2m >= 10)
    set_g2_n  = (g2n[10], g2n(10), g2n ==10 , g2n < 10 , g2n <= 10, g2n > 10 , g2n >= 10)
    set_g2_mn = (g2mn[[10,10]] , g2mn( [10,10] )   , g2mn == [ 10,10 ], g2mn < [10,10 ] , g2mn <= [10,10 ], g2mn > [10,10 ], g2mn >= [10,10 ])

    set_loop  = ( set_g2, set_g2_x, set_g2_y, set_g2_xy, set_g2_m, set_g2_n, set_g2_mn, )
    set_title = (   "g2","g2_x","g2_y","g2_xy","g2_m", "g2_n", "g2_mn" )

    # g1
    for title, plist in zip(set_g1_title, set_g1_loop):
        print()
        print(title)
        print(set_op1)
        print(plist)
    # g2
    for title, plist in zip( set_title, set_loop ) :
        print( )
        print(title)
        print(set_op1)
        print(plist)

def test_grid_manual_Grid_conversion_operators():
    ## abs2phy  ##
    g1_x = laygo2.object.grid.OneDimGrid(name='xgrid', scope=[0, 180], elements=[0, 35, 85, 130])
    g1_y = laygo2.object.grid.OneDimGrid(name='ygrid', scope=[0, 30], elements=[0])
    g2   = laygo2.object.grid.Grid(name='test', vgrid = g1_x, hgrid = g1_y)
    print("start")

    print("phy 2 abs ")
    test_title_phy2abs=[
        "__eq__", "__gt__", "__lt__", "__getitem__", "__call__", "__eq__", "__gt__", "__lt__"
    ]

    test_upper_phy2abs=[
        g2 == [35,0], g2 > [40, 0], g2 < [40, 0], g2.mn[35, 0], g2.mn([35, 0]), g2.xy == [35,0], g2.xy > [35,0], g2.xy < [35,0] ]

    test_lower_phy2abs = [
        g2 == [-95, 0], g2 > [-95, 0], g2 < [-95, 0], g2.mn[-95, 0], g2.mn([-95, 0]), g2.xy == [-95, 0], g2.xy > [-95, 0],
        g2.xy < [-95, 0]]


    test_title_abs2phy=[
       "__getitem__",  "__eq__", "__gt__", "__lt__", "__getitem__", "__call__"
    ]

    test_upper_abs2phy = [
        g2[1, 0], g2.mn == [1, 0], g2.mn > [1, 0], g2.mn < [1, 0],
        g2.xy[1, 0], g2.xy([1, 0]) ]

    test_lower_abs2phy = [
        g2[-1, 0], g2.mn == [-1, 0], g2.mn > [-1, 0], g2.mn < [-1, 0],
        g2.xy[-1, 0], g2.xy([-1, 0])]

    #for t, mn in zip( test_title, test_upper_phy2abs) :
    #    print( t, mn )

    #for t, mn in zip( test_title_phy2abs, test_lower_phy2abs) :
    #    print( t, mn )
    #for t, mn in zip( test_title_abs2phy, test_upper_abs2phy) :
    #    print( t, mn )

    for t, mn in zip( test_title_abs2phy, test_lower_abs2phy) :
        print( t, mn )

def test_grid_manual_CircularMapping():
    elements = [ 0, 35, 85, 130]
    cm = laygo2.object.CircularMapping( elements = elements)
    test_set = [
        [cm.dtype, "dtype"],
        [cm.elements, "elements"],
        [cm.shape, "shape"]
    ]

    for v, t in test_set:
        print(t, v, type(v))
    print(cm[5])
    print(cm[0:10])

def test_grid_manual_CircularMappingArray():
    elementx = [0, 35, 85, 130 ]
    elementy = [0 , 35]
    elements  = [[0,0], [35,0], [85,0], [130,0],
                 [35,35], [85, 35], [130,35 ]  ]
    cm = laygo2.object.CircularMapping(elements=elements)
    test_set = [
        [cm.dtype, "dtype"],
        [cm.elements, "elements"],
        [cm.shape, "shape"]
    ]
    print(elements)
    for v, t in test_set:
        print(t, v, type(v))
    print( cm[1] )
    print(cm[[0, 1]])

def test_grid_manual_AbsToPhyGridConverter():
    g1_x = laygo2.object.grid.OneDimGrid(name='xgrid', scope=[0, 180], elements=[0, 35, 85, 130, 50])
    abs2phy = laygo2.object.grid._AbsToPhyGridConverter( master = g1_x)
    print( abs2phy)
    test_set=[
        [abs2phy.master, "master"],
        [abs2phy[0], "__getItem__ "],
        [abs2phy(0), " call"],
        [abs2phy == 35, " equal"],
        [abs2phy < 35, "lt"],
        [abs2phy <= 35, "te"],
        [abs2phy > 35, "gt"],
        [abs2phy >= 35, "ge"]
    ]

    for v, t in test_set:
        print( t, v, type(v) )

def test_grid_manual_PhyToAbsGridConverter():
    g1_x = laygo2.object.grid.OneDimGrid(name='xgrid', scope=[0, 180], elements=[0, 35, 85, 130, 50])
    phy2abs = laygo2.object.grid._PhyToAbsGridConverter(master=g1_x)
    print(" ")
    print(phy2abs)
    test_set = [
        [phy2abs.master, "master"],
        [phy2abs[35], "__getItem__ "],
        [phy2abs(35), " call"],
        [phy2abs == 1, " equal"],
        [phy2abs < 1, "lt"],
        [phy2abs <= 1, "te"],
        [phy2abs > 1, "gt"],
        [phy2abs >= 1, "ge"],
    ]
    for v, t in test_set:
        print(t, v, type(v))

    g1_x = laygo2.object.grid.OneDimGrid(name='xgrid', scope=[0, 100], elements=[10, 20, 40, 50, 60])
    g1_y = laygo2.object.grid.OneDimGrid(name='ygrid', scope=[0, 100], elements=[10, 20, 40, 50, 60])
    g2 = laygo2.object.grid.Grid(name='test', vgrid=g1_x, hgrid=g1_y)
    phy2abs = laygo2.object.grid._PhyToAbsGridConverter(master=g2)

    rect0 = laygo2.object.physical.Rect(xy=[[0, 0], [100, 100]], layer=['M1', 'drawing'], netname='net0')
    test_set2=[
      [phy2abs.bbox(rect0), "bbox"],
      [phy2abs.bottom_left(rect0), "bl"],
      [phy2abs.bottom_right(rect0), "br"],
      [phy2abs.top_left(rect0), "tl"],
      [phy2abs.top_right(rect0), "tr"],
      [phy2abs.center(rect0), "center"],
      [phy2abs.width(rect0), "width"],
      [phy2abs.height(rect0), "height"],
      [phy2abs.size(rect0), "size"],
    ]
    for t,v in test_set2:
        print(t,v, type(v))

    g1_y = laygo2.object.grid.OneDimGrid(name='xgrid', scope=[0, 120], elements=[0, 20, 40, 60, 80, 100, 120])
    g1_x = laygo2.object.grid.OneDimGrid(name='ygrid', scope=[0, 10], elements=[0])
    g2 = laygo2.object.grid.Grid(name='test', vgrid=g1_x, hgrid=g1_y)
    phy2abs=laygo2.object.grid._PhyToAbsGridConverter(master=g2)
    rect0 = laygo2.object.physical.Rect(xy=[[0, 0], [60, 90]], layer=['M1', 'drawing'], netname='net0',
                                        params={'maxI': 0.005})
    rect1 = laygo2.object.physical.Rect(xy=[[30, 30], [120, 120]], layer=['M1', 'drawing'], netname='net0',
                                        params={'maxI': 0.005})
    rect2 = laygo2.object.physical.Rect(xy=[[30, 30], [120, 120]], layer=['M1', 'drawing'], netname='net0',
                                        params={'maxI': 0.005})

    print("start")

    set_bbox2t = ("overlap", "crossing", "union ")
    set_bbox2 = (phy2abs.overlap(rect0, rect1), phy2abs.crossing(rect0, rect1), phy2abs.union(rect0, rect1))

    for t, v in zip(set_bbox2t, set_bbox2):
        print(t, v, type(v))

def test_grid_manual_OneDimGrid():
    g1_x = laygo2.object.grid.OneDimGrid(name='xgrid', scope=[0, 180], elements=[0, 35, 85, 130, 50])
    print("  ")
    test_set=[
        [g1_x.name, "name"],
        [g1_x.range, "range"],
        [g1_x.phy2abs, "phy2abs"],
        [g1_x.abs2phy, "abs2phy"],
        [g1_x.width, "width"],
        [ g1_x.export_to_dict(), " export "]
    ]

    for v, t in test_set:
        print(t + "   ", v, type(v) )
    print(g1_x)

def test_grid_manual_Grid_xy():
    ## abs2phy  ##
    g1_x = laygo2.object.grid.OneDimGrid(name='xgrid', scope=[0, 100], elements=[0, 10, 20, 40, 50])
    g1_y = laygo2.object.grid.OneDimGrid(name='ygrid', scope=[0, 100], elements=[10, 20, 40, 50, 60])
    g2   = laygo2.object.grid.Grid(name='test', vgrid = g1_x, hgrid = g1_y)

    set_op1 =( "[]", "()", "<", "<=", ">", ">=" )

    set_xy=(
        g2.xy[10, 10], g2.xy([10, 10]), g2.xy < [10,10], g2.xy <= [10, 10],
        g2.xy > [10,10], g2.xy >= [10, 10] )

    set_x = ( g2.x[10], g2.x([10]), g2.x < 10, g2.x <= 10, g2.x > 10, g2.x >= 10 )
    set_y = ( g2.y[10], g2.y([10]), g2.y < 10, g2.y <= 10, g2.y > 10, g2.y >= 10 )

    set_op2 = ("[]", "<", "<=", ">", ">=")
    set_g1x = ( g1_x[10],  g1_x < 10, g1_x <=10, g1_x >10 , g1_x >=10 )

    print("start")
    print(" 2-dimension, xy ")
    for token, value in zip(set_op1, set_xy):
        print(token + "  : ", value)


    print(" 2-dimension, x access ")
    for token, value in zip( set_op1, set_x):
        print( token + "  : ", value)

    print(" 1-dimension ")
    for token, value in zip( set_op2, set_g1x):
        print( token + "  : ", value)

def test_grid_manual_Grid_mn():
    g1_x = laygo2.object.grid.OneDimGrid(name='xgrid', scope=[0, 100], elements=[0, 10, 20, 40, 50])
    g1_y = laygo2.object.grid.OneDimGrid(name='ygrid', scope=[0, 100], elements=[10, 20, 40, 50, 60])
    g2 = laygo2.object.grid.Grid(name='test', vgrid=g1_x, hgrid=g1_y)

    set_op1 = ("[]", "()", "<", "<=", ">", ">=")
    set_mn  = ( g2.mn[40, 40], g2.mn([40,40]), g2.mn < [40,40] , g2.mn <= [40, 40] , g2.mn > [40, 40] , g2.mn >= [40, 40]  )

    set_n = (g2.n[40], g2.n(40), g2.n < 40, g2.n <= 40, g2.n > 40, g2.n >= 40)
    print("start!")
    for token, value in zip( set_op1, set_mn):
        print(token + "  : ", value, "   :", type(value))

    print("n")
    for token, value in zip(set_op1, set_n):
        print(token + "  : ", value, "   :", type(value))

def test_grid_manual_Grid_property():
    g1_x = laygo2.object.grid.OneDimGrid(name='xgrid', scope=[0, 100], elements=[0, 10, 20, 40, 50])
    g1_y = laygo2.object.grid.OneDimGrid(name='ygrid', scope=[0, 100], elements=[10, 20, 40, 50, 60])
    g2 = laygo2.object.grid.Grid(name='test', vgrid=g1_x, hgrid=g1_y)

    set_op1 = ("shape", "width", "height", "height_vec", "width_vec", "range")
    set_p  = ( g2.shape, g2.width, g2.height, g2.height_vec, g2.width_vec, g2.range  )

    for token, value in zip(set_op1, set_p):
        print(token + "  : ", value, "   :", type(value))

    print(g2.summarize())

def test_grid_manual_Grid_bboxHandler():
    g1_x = laygo2.object.grid.OneDimGrid(name='xgrid', scope=[0, 100], elements=[10, 20, 40, 50, 60])
    g1_y = laygo2.object.grid.OneDimGrid(name='ygrid', scope=[0, 100], elements=[10, 20, 40, 50, 60])
    g2   = laygo2.object.grid.Grid(name='test', vgrid=g1_x, hgrid=g1_y)

    rect0 = laygo2.object.physical.Rect(xy=[[0, 0], [100, 100]], layer=['M1', 'drawing'], netname='net0', params={'maxI': 0.005})
    rect1 = laygo2.object.physical.Rect(xy=[[50, 50], [150, 150]], layer=['M1', 'drawing'], netname='net0', params={'maxI': 0.005})
    rect2 = laygo2.object.physical.Rect(xy=[[70, 70], [150, 150]], layer=['M1', 'drawing'], netname='net0', params={'maxI': 0.005})

    pin0  = laygo2.object.physical.Pin(xy =[[0, 0], [100, 100]], layer=['M1', 'drawing'], netname='net0', master=rect0,
               params={'direction': 'input'})

    #g2.bbox(pin0)
    set_bboxt= ( "bbox", "bl", "br", "tl", "tr")
    set_bbox = ( g2.bbox(pin0), g2.bottom_left(pin0), g2.bottom_right(pin0), g2.top_left(pin0), g2.top_right(pin0) )

    for t,v in zip(set_bboxt, set_bbox):
        print(t,v, type(v))

def test_grid_manual_Grid_bboxHandler2():
    g1_y = laygo2.object.grid.OneDimGrid(name='xgrid', scope=[0, 120], elements=[0,20,40,60,80,100,120])
    g1_x = laygo2.object.grid.OneDimGrid(name='ygrid', scope=[0, 10], elements=[0])
    g2 = laygo2.object.grid.Grid(name='test', vgrid=g1_x, hgrid=g1_y)

    rect0 = laygo2.object.physical.Rect(xy=[[0, 0], [ 60 , 90 ]], layer=['M1', 'drawing'], netname='net0',
                                        params={'maxI': 0.005})
    rect1 = laygo2.object.physical.Rect(xy=[[30, 30], [120, 120]], layer=['M1', 'drawing'], netname='net0',
                                        params={'maxI': 0.005})
    rect2 = laygo2.object.physical.Rect(xy=[[30, 30], [120, 120]], layer=['M1', 'drawing'], netname='net0',
                                        params={'maxI': 0.005})

    print("start")



    set_rectt   = ("rect0", "rect1", "rect2")
    set_rect_mn = (g2.mn( rect0.bbox ), g2.mn(rect1.bbox)  , g2.mn(rect2) )
    set_rect_xy = (rect0.bbox, rect1.bbox, rect2.bbox )

    for t, v, v2 in zip(set_rectt, set_rect_mn, set_rect_xy):
        print("\n", t, "\n",  v, "\n", v2)


    set_bbox2t = ("overlap", "crossing", "union ")
    set_bbox2 = (g2.overlap(rect0, rect1), g2.crossing(rect0, rect1), g2.union(rect0, rect1))

    for t, v in zip(set_bbox2t, set_bbox2):
        print(t, v, type(v))

def test_grid_manual_PlacementGrid():
    g1_x = laygo2.object.grid.OneDimGrid(name='xgrid', scope=[0, 20], elements=[0])
    g1_y = laygo2.object.grid.OneDimGrid(name='ygrid', scope=[0, 100], elements=[0])
    g2 = laygo2.object.grid.PlacementGrid(name='test', vgrid=g1_x, hgrid=g1_y)
    inst0_pins = dict()
    inst0_pins['in'] = laygo2.object.physical.Pin(xy=[[0, 0], [10, 10]], layer=['M1', 'drawing'], netname='in')
    inst0_pins['out'] = laygo2.object.physical.Pin(xy=[[90, 90], [100, 100]], layer=['M1', 'drawing'], netname='out')
    inst0 = laygo2.object.physical.Instance(name="I0", xy=[100, 100], libname="mylib", cellname="mycell"
                                            , shape=[3, 2], pitch=[200, 200], unit_size=[100, 100], pins=inst0_pins,
                                            transform='R0')

    print("start")
    print(inst0.xy)

    g2.place(inst=inst0, mn=[10,10])
    print(inst0.xy)

def test_grid_manual_RoutingGrid():
    templates = tech.load_templates()
    grids = tech.load_grids(templates=templates)
    pg, r12, r23 = grids['placement_cmos'], grids['routing_12_cmos'], grids['routing_23_cmos']


    print("start")
    print(r23.elements)
    test_list=[
        ("type"  , r23.type),
        ("vwidth", r23.vwidth),
        ("hwidth", r23.hwidth),
        ("vextension", r23.vextension),
        ("hextension", r23.hextension),
        ("vlayer", r23.vlayer),
        ("hlayer", r23.hlayer),
        ("pin_vlayer", r23.pin_vlayer),
        ("pin_hlayer", r23.pin_hlayer),
        ("viamap", r23.viamap),
        ("primary_grid", r23.primary_grid)
    ]
    for t, v in test_list:
        print(t, v, end="  ")
        print(type(v))

    mn_list = [[0, -2], [0, 1], [2, 1], [5, 1]]
    route = r23.route(mn=mn_list, via_tag=[True, None, True, True])
    print("route:")
    print(route)
    mn_list = [[0, -2], [1, 0], [2, 5]]
    via = r23.via(mn=mn_list)
    print("via:")
    print(via)
    mn_list = [[0, -2], [1, 0], [2, 5], [3, 4], [4, 5], [5, 5]]
    track = r23.route_via_track(mn=mn_list, track=[None, 0])
    print("track:")
    print(track)
    mn_list = [ [0, 0], [10, 10]]
    pin = r23.pin(mn=mn_list, name="pin")
    print("pin:")
    print(pin)

def test_physical_manual_PhysicalObject():

    physical = laygo2.object.physical.PhysicalObject( xy = [[0, 0], [200, 200]], name="test", params={'maxI': 0.005})
    print("start")
    set_titile1= ("name", "xy", "bbox", "master", "params")
    set_print=(physical.name, physical.xy, physical.bbox, physical.master, physical.params)


    set_title2  =("pointers", "left", "right", "top", "bottom", "center", "bottom_left", "bottom_right", "top_left", "top_right")
    set_pointer=(physical.pointers, physical.left, physical.right, physical.top, physical.bottom, physical.center, physical.bottom_left, physical.bottom_right,
                 physical.top_left, physical.top_right
                 )
    for t, x in zip(set_titile1, set_print):
        print( t  + "  " +  str(x)  + "  " + str(type(x))     )

    for t, x in zip(set_title2, set_pointer):
        print( t  + "  " +  str(x)  + "  " + str(type(x))     )

    print(physical)

def test_physical_manual_IterablePhysicalObject():
    phy0  = laygo2.object.physical.IterablePhysicalObject( xy=[[0, 0], [100, 100]], name="test" )
    phy1 = laygo2.object.physical.IterablePhysicalObject( xy=[[0, 0], [200, 200]], name="test")
    phy2 = laygo2.object.physical.IterablePhysicalObject( xy=[[0, 0], [300, 300]], name="test")
    element = [phy0, phy1, phy2]
    iphy0 = laygo2.object.physical.IterablePhysicalObject( xy=[[0, 0], [300, 300]], name="test", elements = element)

    test_set=[
        iphy0.elements,
        iphy0.shape,
        iphy0.ndenumerate()
    ]
    print(" ")
    for v in test_set:
        print(v)
        print(type(v))
    print(iphy0)

def test_physical_manual_instance():
    import numpy as np
    inst0_pins = dict()
    inst0_pins['in']  = laygo2.object.physical.Pin( xy = [[0, 0], [10,10]], layer = ['M1', 'drawing'], netname = 'in')
    inst0_pins['out'] = laygo2.object.physical.Pin( xy = [[90, 90], [100, 100]], layer=['M1', 'drawing'], netname='out')
    inst0 = laygo2.object.physical.Instance( name = "I0", xy = [ 100,100], libname="mylib", cellname="mycell"
    , shape = [3,2], pitch = [200,200], unit_size = [100, 100], pins = inst0_pins, transform =  'R0')


    test_value= [
        [inst0.libname,  "libname"],
        [inst0.cellname, "cellname"],
        [inst0.unit_size, "unit_size"],
        [inst0.transform, "transform"],
        [inst0.pins, "pins"],
        [inst0.xy0, "xy0"],
        [inst0.xy1, "xy1"],
        [inst0.size, "size"],
        [inst0.pitch, "pitch"],
        [inst0.spacing, " spacing "],
        [inst0.height, " height"],
        [inst0.width, "width"],
        [inst0.xy, "xy"],
        [inst0.bbox, "bbox"]
    ]

    for v,t  in test_value:
        print(t, v, end="  ")
        print(type(v))


    for idx, it in inst0.ndenumerate():
        print( "   ", idx, it)

    print("pins")
    print(inst0.pins["in"].shape)
    for idx, it in inst0.pins["in"].ndenumerate():
        print( "   ", idx, it)

    print( inst0[1,0].xy0 )
    print( inst0.pins["in"][1,1].xy)

def test_physical_manual_Rect():
    rect0 = laygo2.object.physical.Rect(xy=[[0, 0], [100, 100]], layer=['M1', 'drawing'], netname='net0', hextension=20, vextension=20)
    test_set=[
        [ rect0.layer, " layer"],
        [ rect0.netname, " netname"],
        [ rect0.hextension,  " hextension"],
        [ rect0.vextension,  " vextension"],
        [ rect0.height,  " height"],
        [ rect0.width,  " width"],
        [ rect0.size, " size"]
    ]
    for v, t in test_set:
        print(t, v, end="  ")
        print( type(v) )
    print(rect0)

def test_physical_manual_Path():
    path0 = laygo2.object.physical.Path(xy=[[0, 0], [0, 100]], width=10, extension=5, layer=['M1', 'drawing'], netname='net0')
    test_set = [
        [path0.layer, " layer"],
        [path0.netname, " netname"],
        [path0.extension, " extension"],
        [path0.width, " width"],
        ]
    for v,t  in test_set:
        print(t, v, end="  ")
        print(type(v))
    print(path0)

def test_physical_manual_Pin():
    pin0 = laygo2.object.physical.Pin( xy=[[0, 0], [100, 100]], layer=['M1', 'drawing'], netname='net0',
               params={'direction': 'input'})
    test_set=[
    [pin0.layer, " layer"],
    [pin0.netname, " netname"],
    [pin0.height, " height"],
    [pin0.width, " width"],
    [pin0.size, " size"],
    [pin0.bbox, " bbox"],
    [pin0.xy, " xy"]

    ]
    print(" ")
    for v, t in test_set:
        print(t, v, end="  ")
        print(type(v))

    print( pin0 )

def test_physical_manual_Text():
    text0 = laygo2.object.physical.Text(xy=[[ 0, 0], [100,100 ]], layer=['text', 'drawing'], text='test', params=None)
    test_set = [
        [text0.layer, " layer"],
        [text0.text, " text"],
    ]
    print(" ")
    for v, t in test_set:
        print(t, v, end="  ")
        print(type(v))

def test_physical_manual_VirtuialInstance():
    vinst0_pins = dict()
    vinst0_pins['in']  = laygo2.object.physical.Pin(xy=[[0, 0], [10, 10]], layer=['M1', 'drawing'], netname='in')
    vinst0_pins['out'] = laygo2.object.physical.Pin(xy=[[90, 90], [100, 100]], layer=['M1', 'drawing'], netname='out')
    vinst0_native_elements = dict()
    vinst0_native_elements['R0'] = laygo2.object.physical.Rect(xy=[[0, 0], [10, 10]], layer=['M1', 'drawing'])
    vinst0_native_elements['R1'] = laygo2.object.physical.Rect(xy=[[90, 90], [100, 100]], layer=['M1', 'drawing'])
    vinst0_native_elements['R2'] = laygo2.object.physical.Rect(xy=[[0, 0], [100, 100]], layer=['prBoundary', 'drawing'])
    vinst0 = laygo2.object.physical.VirtualInstance(name='I0', libname='mylib', cellname='myvcell', xy=[500, 500],
                            native_elements=vinst0_native_elements, shape=[3, 2], pitch=[100, 100],
                            unit_size=[100, 100], pins=vinst0_pins, transform='R0')

    test_set=[
        [vinst0.native_elements, "native_elements"]
    ]
    print("  ")
    for v, t in test_set:
        print(t, v, end="  ")
        print(type(v))

    print("  ")
    print("  ", vinst0)
    for idx, it in vinst0.ndenumerate():
        print("  ", idx, it)


    print(" pins")
    for idx, it in vinst0.pins['in'].ndenumerate():
        print("  ", idx, it)

def test_template_manual_NativeInstanceTemplate():
    print("NativeInstanceTemplate test")
    # define pins
    nat_temp_pins = dict()
    nat_temp_pins['in']  = laygo2.object.Pin(xy=[[0, 0], [10, 10]], layer=['M1', 'drawing'], netname='in')
    nat_temp_pins['out'] = laygo2.object.Pin(xy=[[90, 90], [100, 100]], layer=['M1', 'drawing'], netname='out')
    # create a template
    nat_temp = laygo2.object.NativeInstanceTemplate(libname='mylib', cellname='mynattemplate', bbox=[[0, 0], [100, 100]],
                                      pins=nat_temp_pins)
    nat_inst = nat_temp.generate(name='mynatinst', shape=[2, 2], pitch=[100, 100], transform='R0')

    test_set = [
        [nat_temp.name,     " name"],
        [nat_temp.libname,  " libname"],
        [nat_temp.cellname, " cellname"],
        [nat_temp.height(), " heightname"],
        [nat_temp.width(),  " width"],
        [nat_temp.size(),   " size"],
        [nat_temp.bbox(), " bbox"],
        [nat_temp.pins(), " pins"],
        [nat_temp.generate(), " generatos"],
        [nat_temp.export_to_dict(), " ex"]
    ]

    for target, text in test_set:
        print(text, end= ":   ")
        print(target, type(target))

def test_template_manual_ParameterizedInstanceTemplate():
    print("ParameterizedInstanceTemplate test")
    pcell_temp_pins = dict()

    def pcell_bbox_func(params):
        if params==None:
            params={}
            params['W'] = 1
        return np.array([[0, 0], [100 , 100* params['W']]])

    # define the pin generation function.
    def pcell_pins_func(params):
        if params==None:
            params={"W":1}
        i = params['W']
        template_pins = dict()
        pin_in  = laygo2.object.Pin(xy =[ [ 0, 0], [100 , 0 ] ],      layer=['M1', 'pin'], netname='in')
        pin_out = laygo2.object.Pin(xy =[ [ 0, 100], [100 , 100* i]], layer=['M1', 'pin'], netname='out')
        template_pins['in' ] = pin_in
        template_pins['out'] = pin_out
        return template_pins

    # create a template
    pcell_temp = laygo2.object.ParameterizedInstanceTemplate(libname='mylib', cellname='mynattemplate', bbox_func=pcell_bbox_func,
                                      pins_func=pcell_pins_func)
    pcell_inst1 = pcell_temp.generate(name='I1', params={ "W":2, "L":1} )
    pcell_inst2 = pcell_temp.generate(name='I2', params={"W": 1, "L": 1})

    test_set = [
        [pcell_temp.name,     " name"],
        [pcell_temp.libname,  " libname"],
        [pcell_temp.cellname, " cellname"],
        [pcell_temp.height(), " heightname"],
        [pcell_temp.width(),  " width"],
        [pcell_temp.size(),   " size"],
        [pcell_temp.bbox(), " bbox"],
        [pcell_temp.pins(), " pins"],
        [pcell_inst1, " generatos"],
    ]

    for target, text in test_set:
        print(text, end= ":   ")
        print(target, type(target))

def test_template_manual_UserDefinedTemplate():
    print("UserDefinedTemplate test")

    # define the bbox computation function.
    def user_bbox_func(params):
        if params==None:
            params={}
            params['multi'] = 1
        return np.array([[0, 0], [100 * params['multi'], 100]])

    # define the pin generation function.
    def user_pins_func(params):
        if params==None:
            params={}
            params['multi'] = 1
        i = params['multi']
        template_pins = dict()
        pin_in  = laygo2.object.Pin(xy =[ [ 0, 0], [100 * i, 0 ] ],   layer=['M1', 'pin'], netname='in')
        pin_out = laygo2.object.Pin(xy=[ [ 0, 100], [100 * i, 100]], layer=['M1', 'pin'], netname='out')
        template_pins['in' ] = pin_in
        template_pins['out'] = pin_out
        return template_pins

    # define the instance generation function.
    def user_generate_func(name=None, shape=None, pitch=np.array([0, 0]), transform='R0', params=None):
        if params==None:
            params={}
            params['multi'] = 1
        m = params['multi']
        shape = np.array([1, 1])
        inst_pins = user_pins_func(params)
        inst_native_elements = dict()

        inst_native_elements['left'] = laygo2.object.Rect(xy=[ [0, 0], [0,100]], layer=['M1', 'drawing'])
        ofst = np.array([100, 0])
        for i in range(m):
            bl  = np.array([0,0])
            tr  = np.array([100,100])
            inst_native_elements['center'+str(i)] = laygo2.object.Rect(xy=[ i*ofst + bl , i*ofst + tr ], layer=['M3', 'drawing'])

        inst_native_elements['right'] = laygo2.object.Rect(xy=[ m*ofst + [0, 0], m*ofst + [ 0 , 100]], layer=['M1', 'drawing'] )
        inst = laygo2.object.VirtualInstance(name=name, libname='mylib', cellname='myvinst', xy=np.array([0, 0]),
                                             native_elements=inst_native_elements, shape=shape,
                                             pitch=pitch, unit_size=[m * 100, 100], pins=inst_pins,
                                             transform=transform, params=params)
        return inst

    user_temp = laygo2.object.UserDefinedTemplate(name='myusertemplate', bbox_func=user_bbox_func, pins_func=user_pins_func,
                                    generate_func=user_generate_func)
    #user_inst = user_temp.generate(name='myuserinst',  params={'multi': 5})
    print(user_temp)
    test_set = [
        [user_temp.name, " name"],
        [user_temp.height(), " height"],
        [user_temp.width(), " width"],
        [user_temp.size(), " size"],
        [user_temp.bbox(), " bbox"],
        [user_temp.pins(), " pins"],
        [user_temp.generate(), " generatos"],

    ]

    for target, text in test_set:
        print(text, end=":   ")
        print(target, type(target))

    #print(user_inst)
    #for n,i in user_inst.native_elements.items():
    #    print(i)

def test_database_manual_BaseDatabase():
    base = laygo2.object.BaseDatabase(name='mycell')

    rect0 = laygo2.object.Rect(xy=[[0, 0], [100, 100]], layer=['M1', 'drawing'], name='R0', netname='net0', params={'maxI': 0.005})
    rect1 = laygo2.object.Rect(xy=[[200, 0], [300, 100]], layer=['M1', 'drawing'], netname='net0', params={'maxI': 0.005})
    pin0 = laygo2.object.Pin(xy=[[0, 0], [100, 100]], layer=['M1', 'pin'], netname='n0', master=rect0, params={'direction': 'input'})

    inst0_pins = dict()
    inst0_pins['in'] = laygo2.object.Pin(xy=[[0, 0], [10, 10]], layer=['M1', 'drawing'], netname='in')
    inst0_pins['out'] = laygo2.object.Pin(xy=[[90, 90], [100, 100]], layer=['M1', 'drawing'], netname='out')
    inst0 = laygo2.object.Instance(name='I0', xy=[100, 100], libname='mylib', cellname='mycell', shape=[3, 2], pitch=[100, 100],
                     unit_size=[100, 100], pins=inst0_pins, transform='R0')

    base.append(rect0)
    base.append(rect1)
    base.append(pin0)
    base.append(inst0)

    p_name = [
    ( "name",base.name ),
    ("params", base.params),
    ("noname_index", base.noname_index),
    ("keys", base.keys()),
    ("elements", base.elements) ]

    print("start")
    for t, v in p_name:
        print(t, v, end="  ")
        print(type(v))

    print("items()")
    print(base.items())
    print(type(base.items()))

    print("__getItem__()")
    print(base["R0"])
    print("__setItem__()")
    rect2 = laygo2.object.Rect(xy=[[0, 0], [100, 100]], layer=['M1', 'drawing'], name='R2', netname='net0')
    base["R2"] = rect2
    print("__iter__()")
    print(base.__iter__())
    print(type(base.__iter__()))
    for i in base:
        print(i)
    print("init")
    print(base)

def test_database_manual_library():
    lib = laygo2.object.Library(name='mylib')
    print(lib.name)
    print(lib)

def test_database_manual_design():
    dsn = laygo2.object.Design(name='dsn', libname="testlib")
    print("dsn:")
    print(dsn)
    rect0 = laygo2.object.Rect(xy=[[0, 0], [100, 100]], layer=['M1', 'drawing'], name='R0', netname='net0', params={'maxI': 0.005})
    pin0 = laygo2.object.Pin(xy=[[0, 0], [100, 100]], layer=['M1', 'pin'], netname='n0', master=rect0, params={'direction': 'input'})

    inst0_pins = dict()
    inst0_pins['in'] = laygo2.object.Pin(xy=[[0, 0], [10, 10]], layer=['M1', 'drawing'], netname='in')
    inst0_pins['out'] = laygo2.object.Pin(xy=[[90, 90], [100, 100]], layer=['M1', 'drawing'], netname='out')
    inst0 = laygo2.object.Instance(name='I0', xy=[100, 100], libname='mylib', cellname='mycell', shape=[3, 2], pitch=[100, 100],
                     unit_size=[100, 100], pins=inst0_pins, transform='R0')

    vinst0_pins = dict()
    vinst0_pins['in'] = laygo2.object.physical.Pin(xy=[[0, 0], [10, 10]], layer=['M1', 'drawing'], netname='in')
    vinst0_pins['out'] = laygo2.object.physical.Pin(xy=[[90, 90], [100, 100]], layer=['M1', 'drawing'], netname='out')
    vinst0_native_elements = dict()
    vinst0_native_elements['R0'] = laygo2.object.physical.Rect(xy=[[0, 0], [10, 10]], layer=['M1', 'drawing'])
    vinst0 = laygo2.object.physical.VirtualInstance(name='VI0', libname='mylib', cellname='myvcell', xy=[500, 500],
                                                    native_elements=vinst0_native_elements, shape=[3, 2],
                                                    pitch=[100, 100],
                                                    unit_size=[100, 100], pins=vinst0_pins, transform='R0')

    text0 = laygo2.object.physical.Text(xy=[[ 0, 0], [100,100 ]], layer=['text', 'drawing'], text='test', params=None)

    dsn.append(rect0)
    dsn.append(pin0)
    dsn.append(inst0)
    dsn.append(vinst0)
    dsn.append(text0)


    p_name = [
        ("libname",dsn.libname ),
        ("cellname", dsn.cellname),
        ("rects", dsn.rects),
        ("pins", dsn.pins),
        ("texts", dsn.texts),
        ("instnaces", dsn.instances),
        ("virtual_instnaces", dsn.virtual_instances)
        ]
    print("start")

    for t, v in p_name:
        print(t, v, end="  ")
        print(type(v))

    print("get by layer")
    print( dsn.get_matchedrects_by_layer(['M1', 'drawing']))
    print("export to template")
    print(dsn.export_to_template())
    ## route, route_via_track,
