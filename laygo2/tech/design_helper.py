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

import numpy as np
import laygo2.object.template
import laygo2.object.physical
import laygo2.object.database


def generate_cut_layer(dsn, grids, tlib, templates):
    """
    generate cut layer for space violation
    """

    r23     = grids["routing_23_cmos"]
    r23_cut = grids["routing_23_cmos_cut"] 
    dsn.rect_space("M0",r23,r23_cut,150)

def generate_tap(dsn, grids, tlib, templates, type_iter='nppn', type_extra=None, transform_iter='0X0X', transform_extra=None, side='both'): 
    pass

def generate_gbnd(dsn, grids, tlib, templates, gbnds: dict):
    # TBD

    pg = grids["placement_basic"]
    # unpack
    gbndl = gbnds.get("gbndl" , False)
    gbndr = gbnds.get("gbndr", False)
    

    mn_dsn = pg.mn(dsn.bbox)

    width    = mn_dsn[1][0]
    height   = mn_dsn[1][1]

    if gbndl:

        tfs     = ["R0", "MX", "R0", "MX"]
        names   = ["gbndl", "gbndr"]
        start_l = lambda x : mn_dsn[0][0] - x
        start_r = lambda x : mn_dsn[1][0] + 0
        starts  = (start_l, start_r)

        for k, gbnd_sub in enumerate( (gbndl, gbndr) ):
            
            start_cal = starts[k]
            n_max     = height
            n_cur     = 0
            i_loop    = 0
            print(k)

            while n_cur  < n_max:
                t_left = gbnd_sub[i_loop % len(gbnd_sub) ]
                print(t_left)
                tf     = tfs[  i_loop % len(gbnd_sub) ]
               
                igbnd    = templates[t_left].generate( name = f"{names[k]}_{i_loop}", transform = tf )
                n_height = pg.n( igbnd.height )
                m_width  = pg.m( igbnd.width  )
                n_cur    = n_height * i_loop
                
                if n_cur == n_max:
                    break

                start_m  = start_cal(m_width)
                dsn.place( grid = pg, inst = [igbnd,], mn = [ start_m, n_cur  ] )

                i_loop = i_loop + 1

    # right

def generate_gbnd_edge(dsn, grids, tlib, templates, t_gbnds: list, pattern_dopping: list, pattern_tf: list):
    """
        t_gbnds: [t_nmos_gbnd, t_pmos_gbnd]
        pattern_dopping: the sequence of 0 or 1, 0 = nmos, 1 = pmos.
        patter_tf      : the seqeunce of pattern
    """

    pg = grids["placement_basic"]
    mn_dsn  = pg.mn(dsn.bbox)

    width    = mn_dsn[1][0]
    height   = mn_dsn[1][1]

    n_max     = height
    n_cur     = 0
    i_loop    = 0
    i_pattern = len(pattern_tf)

    while n_cur  < n_max:
        tf       = pattern_tf[      i_loop % i_pattern ]
        tf2      = 0

        if tf == "R0":
            tf2 = "MY"
        elif tf == "MX":
            tf2 = "R180"

        i_dop    = pattern_dopping[ i_loop % i_pattern ]
        t_dop    = t_gbnds[i_dop]
        
        igbnd_l  = templates[t_dop].generate( name = f"gbndl_{i_loop}", transform = tf  )
        igbnd_r  = templates[t_dop].generate( name = f"gbndr_{i_loop}", transform = tf2 )
        
        n_height = pg.n( igbnd_l.height )
        m_width  = pg.m( igbnd_l.width  )
        n_cur    = n_height * i_loop
           
        if n_cur == n_max:
            break

        dsn.place( grid = pg, inst = [igbnd_l], mn = [ mn_dsn[0][0] - m_width, n_cur  ] )
        dsn.place( grid = pg, inst = [igbnd_r], mn = [ mn_dsn[1][0]          , n_cur  ] )

        i_loop = i_loop + 1


def generate_pwrail(dsn, grid, vss_name='VSS', vdd_name='VDD'):

    r23      = grid
    y_height = r23.height * 0.5
    n_height = r23.n(y_height)
    grid_cnt = np.int(dsn.bbox[1,1] / ( y_height )  ) + 1 # Calculate the number of power rails in the design

    # Calculate the number of iterations of each power net
    if grid_cnt % 2 == 0:
        iter_vdd = grid_cnt//2
        iter_vss = grid_cnt//2 + 1
    else:
        iter_vdd = (grid_cnt + 1 )//2
        iter_vss = (grid_cnt + 1 )//2

    rvss = []
    rvdd = []
   
    p_space = 32
    names   = [vss_name, vdd_name]
    iters   = [iter_vss, iter_vdd]
    _mn     = [r23.mn.bottom_left( dsn.bbox ), r23.mn.bottom_right(dsn.bbox)]
    
    for i in range(grid_cnt ):    
        name      = names[ i%2]
        _mn[0][1] = n_height *  i   
        _mn[1][1] = n_height *  i 
        route     = dsn.route(grid=r23, mn=_mn)
                
        #dsn.pin(name = name+"center" +str(i), grid=r23, mn=r23.mn.bbox(route), netname= name + ':')
        dsn.pin(name = name+"left" +str(i), grid=r23, mn= [ _mn[0], _mn[0] + [1,0]]  , netname= name + ':')
        dsn.pin(name = name+"right" +str(i), grid=r23, mn= [ _mn[1]-[1,0], _mn[1]]  , netname= name + ':')

        for k in range( ( (_mn[1][0] - _mn[0][0]) // p_space ) - 1  ):
            _mn1 = [ p_space * k     +  _mn[0][0], _mn[0][1] ]
            _mn2 = [ p_space * (k+1) +  _mn[0][0], _mn[1][1] ]
            dsn.pin(name = f"{name}_{i}_{k}", grid = r23, mn = [ _mn1, _mn2 ]  , netname= name + ':')

def fill_by_instance(dsn, grids, tlib, templates, inst_name:str, iter_type=("R0","MX"), mn_bndr = [None, None], offsets = [None, None] ):
    """ fill empty layout space by given instances  
        iter_type = y-direction transforms 
        mn_bndr   = boundary instances or boundary to be filled
        
        
    """
    pg      = grids["placement_basic"]
    dsnbbox = pg.mn(dsn.bbox)
    offset  = dsnbbox[0]
    width   = dsnbbox[1][0] - 0
    height  = dsnbbox[1][1] - dsnbbox[0][1]

    canvas = np.ones((height, width), dtype=int)
    xy_bl  = pg.mn.bottom_left(dsn.bbox)
    xy_tr  = pg.mn.top_right(dsn.bbox)
        
    if isinstance( mn_bndr[0], laygo2.object.physical.Instance) :
        xy_bl = pg.mn( mn_bndr[0].bottom_left )
        xy_tr = pg.mn( mn_bndr[1].top_right   )
        #canvas[ xy_bl[1] : xy_tr[1], xy_bl[0]: xy_tr[0] ] = 0
    
    if isinstance( mn_bndr[0], (list, np.ndarray) ):
        xy_bl = mn_bndr[0]

    if isinstance( mn_bndr[1], (list, np.ndarray) ):
        xy_tr = mn_bndr[1]
    
    canvas[ xy_bl[1] : xy_tr[1], xy_bl[0]: xy_tr[0] ] = 0
    
    def check_occupied(canvas, physical, index):
        bbox = pg.mn(physical.bbox)
        x0 = bbox[0][0]
        x1 = bbox[1][0]
        y0 = bbox[0][1]
        y1 = bbox[1][1]
        if x0 == x1 and y0 == y1:
            return
        canvas[y0:y1, x0:x1] = index

    index = 1
    for n, inst in dsn.instances.items():
        check_occupied(canvas, inst, index)
        index = index + 1
    for n, vinst in dsn.virtual_instances.items():
        check_occupied(canvas, vinst, index)
        index = index + 1

    filler   = templates[inst_name].generate(name="filler", transform="R0")
    f_height = pg.mn(filler)[1][1]
    f_width  = pg.mn(filler)[1][0]
        
    if offsets[0] != None:
        f_width = f_width + offsets[0]

    if offsets[1] != None:
        f_height = f_height + offsets[1]
        
    n_mod    = int(height / f_height)

    for y in range(n_mod):
        buffers = []
        for x in range(width):
            if canvas[f_height * y, x] == 0:
                buffers.append(True)
            else:
                buffers.append(False)

            if np.array_equal(buffers, [True] * f_width):
                tf = iter_type[int(y % len(iter_type))]
                _mn = np.asarray([x - f_width + 1, y * f_height])
                if tf == "MX":
                    _mn = _mn + [0, f_height]

                dsn.place(grid=pg, inst=templates[inst_name].generate(name="filler" + f"{x}_{y * f_height}", transform=tf), mn=_mn)
                buffers = []
            
            elif buffers[-1] == False:
                buffers = []

    return canvas

############################################


