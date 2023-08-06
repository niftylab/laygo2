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

from laygo2.object import *
from laygo2.object.technology import BaseTechnology
from laygo2.object.template.tile import TileMOSTemplate, TileTapTemplate

class NiftyTechnology(BaseTechnology):
    """
    Class for defining technology parameters and objects in "Niftylab-style".
    This greatly streamlines the setup of templates and grids for new technologies,
    and generators constructed based-on the "Niftylab-style" templates cna be reused 
    with minimal modifications.

    However, this comes at the expense of supporting diverse layout styles, particularly 
    those with unique polygon shapes.
    """
    
    def load_tech_templates(self, libname=None):
        """
        Load templates and construct a template library object.

        Parameters
        ----------
            libname: optional, str
                The name of library to be loaded. 
                By default, the first library in tech_params['templates'] is used.
        """
        # Library name
        if libname is None:
            ln = list(self.tech_params['templates'].keys())[0]
        else:
            ln = libname

        # Native templates and grids
        ntemplates = self.tech_params['templates'][ln]
        ngrids     = self.tech_params['grids'][ln]
         
        # Template library
        tlib    = laygo2.object.database.TemplateLibrary(name = ln)
    
        # 1. Load native templates
        for tn, tdict in ntemplates.items():
            # bounding box
            bbox = np.array(tdict['xy'])
            # pins
            pins = None
            if 'pins' in tdict:
                pins = dict()
                for pn, _pdict in tdict['pins'].items():
                    pins[pn] = laygo2.object.Pin(xy=_pdict['xy'], layer=_pdict['layer'], netname=pn)
        
            t = laygo2.object.NativeInstanceTemplate(libname=libname, cellname=tn, bbox=bbox, pins=pins)
            tlib.append(t)
    

        # 2. Construct UserDefinedTemplate objects derived from native templates
        glib = self.load_tech_grids(templates=tlib)

        # Mapping parameters for tile templates
        placement_pattern = ["gbndl", "bndl", "dmyl",  "core", "dmyr", "bndr", "gbndr"]
        transform_pattern = dict( gbndl = "R0", dmyl = "R0", bndl  = "R0", core  = "R0",
                                  dmyr = "MY", bndr  = "R0", gbndr = "R0" )
        routing_map       = dict( G = 3, S = 1, D = 2, G_extension0_x = [None,None], 
                                  S_extension0_m = [None, None], D_extension0_m = [None, None])
    
        nmos_ulvt = dict(
            core  = 'nmos4_fast_center_nf2',
            dmyr  = 'nmos4_fast_dmy_nf2',
            dmyl  = 'nmos4_fast_dmy_nf2',
            bndr  = 'nmos4_fast_boundary',
            bndl  = 'nmos4_fast_boundary',
            gbndr = 'nmos4_fast_boundary',
            gbndl = 'nmos4_fast_boundary',
            grid  = "routing_12_mos",
        )

        pmos_ulvt = dict(
            core  = 'pmos4_fast_center_nf2',
            dmyr  = 'pmos4_fast_dmy_nf2',
            dmyl  = 'pmos4_fast_dmy_nf2',
            bndr  = 'pmos4_fast_boundary',
            bndl  = 'pmos4_fast_boundary',
            gbndr = 'pmos4_fast_boundary',
            gbndl = 'pmos4_fast_boundary',
            grid  = "routing_12_mos",
        )

        ptap_ulvt = dict(
            core  = 'ntap_fast_center_nf2_v2',
            dmyr  = 'nmos4_fast_dmy_nf2',
            dmyl  = 'nmos4_fast_dmy_nf2',
            bndr  = 'ntap_fast_boundary',
            bndl  = 'ntap_fast_boundary',
            gbndr = 'ntap_fast_boundary',
            gbndl = 'ntap_fast_boundary',
            grid  = "routing_12_mos",
        )

        ntap_ulvt = dict(
            core  = 'ptap_fast_center_nf2_v2',
            dmyr  = 'pmos4_fast_dmy_nf2',
            dmyl  = 'pmos4_fast_dmy_nf2',
            bndr  = 'ptap_fast_boundary',
            bndl  = 'ptap_fast_boundary',
            gbndr = 'ptap_fast_boundary',
            gbndl = 'ptap_fast_boundary',

            grid  = "routing_12_mos",
        )
  
        # Generate tile templates 
        gen_list = [["nmos", nmos_ulvt], ["pmos", pmos_ulvt]]
        for name, placement_map in gen_list:
            grid_name  = placement_map["grid"]

            temp = TileMOSTemplate( tlib, glib, grid_name, routing_map, placement_map, placement_pattern, transform_pattern, name)
            tlib.append(temp)
    
        gen_list = [["ptap", ntap_ulvt], ["ntap", ptap_ulvt] ]
        for name, placement_map in gen_list:
            grid_name  = placement_map["grid"]
        
            temp       = TileTapTemplate( tlib, glib, grid_name, routing_map, placement_map, placement_pattern, transform_pattern, name)
            tlib.append(temp)

        return tlib

    # Helper functions. To be customized for target technologies
    def _iterate_for_generate_tap(iter_num, type_iter):
        """ Internal function for generate_tap()."""
        ltap_list = []
        rtap_list = []
        ltapbnd_list = []
        rtapbnd_list = []
        for idx in range(iter_num): # number of iteration
            i=0
            for celltype in type_iter: # in each iteration
                ltap_list.append(templates[celltype+'mos4_fast_tap'].generate(name='LTAP'+str(idx)+str(i), transform='R0' if transform_iter[i]=='0' else 'MX'))
                rtap_list.append(templates[celltype+'mos4_fast_tap'].generate(name='RTAP'+str(idx)+str(i), transform='R0' if transform_iter[i]=='0' else 'MX'))
                ltapbnd_name = 'ptap_fast_left' if celltype == 'n' else 'ntap_fast_left'
                rtapbnd_name = 'ptap_fast_right' if celltype == 'n' else 'ntap_fast_right'
                ltapbnd_list.append(templates[ltapbnd_name].generate(name='LTAPBND'+str(idx)+str(i), transform='R0' if transform_iter[i]=='0' else 'MX'))
                rtapbnd_list.append(templates[rtapbnd_name].generate(name='RTAPBND'+str(idx)+str(i), transform='R0' if transform_iter[i]=='0' else 'MX'))
                i+=1
        return ltap_list, rtap_list, ltapbnd_list, rtapbnd_list

    def generate_tap(self, dsn, grids, tlib, templates, type_iter='nppn', type_extra=None, transform_iter='0X0X', transform_extra=None, side='both'): 
        """ 
        Generates vertical tap stack(s) on the left and/or right sides of the design.

        Parameters
        ----------
        dsn: laygo2.object.database.Design
            The design object to generate tap(s).
        grids: laygo2.object.database.GridLibrary
            The library that contains grid information.
        type_iter : str
            list of transistor types for main taps, which is iterated over the entire row.
            For example, for two NMOS rows and one PMOS row, its value should be ['n', 'n', 'p'].
            ptaps will be placed for NMOS rows, and ntaps will be placed for PMOS row.
            It should have the identical dimension (length) with transform_iter. 
        type_extra : str
            list of transistor types for residual taps, for the leftover rows after iterations.
        transform_iter : str
            list of transform types for main taps.
        transform_extra : str
            list of transform types of residual taps.
        side: str
            The boundary side of layout to generate taps. The value should be one of the following: both, left, right.

        """
        pg         = grids["placement_basic"]           # Load basic placement grid for placing taps.
        height_tap = grids["routing_23_cmos"].height//2 # Calculate the height of tap which is generally the half of the CMOS height.
    
        bbox             = dsn.bbox                         # The bbox of the design.
        height_dsn       = bbox[1][1]                       # The height of the design.
        total_num_of_tap = int(height_dsn // height_tap) # Total number of taps. 8 taps are needed if there are 4 CMOS grids in the design. 5 taps if 2 CMOS grids and 1 half-CMOS.
        iter_len         = len(type_iter)                   # length of iteration
        print('======== TAP GENERATION START ========')
        print('Total number of taps on each side: ' + str(total_num_of_tap))
        print('Iteration tap type: {0}. Transform: {1}'.format(type_iter, transform_iter))
        print('Extra tap type: {0}. Transform: {1}'.format(type_extra, transform_extra))
 
        if total_num_of_tap%iter_len == 0: # full iteration
            ltap_list, rtap_list, ltapbnd_list, rtapbnd_list = _iterate_for_generate_tap(iter_num = total_num_of_tap//iter_len, type_iter = type_iter)
 
        else: # iteration + extra taps
            ltap_list, rtap_list, ltapbnd_list, rtapbnd_list = _iterate_for_generate_tap(iter_num = (total_num_of_tap-len(type_extra))//iter_len, type_iter = type_iter) # Iteration
            i=0
            for celltype in type_extra: # Extra taps
                ltap_list.append(templates[celltype+'mos4_fast_tap'].generate(name='LTAPEND'+celltype+str(i), transform='R0' if transform_extra[i]=='0' else 'MX'))
                rtap_list.append(templates[celltype+'mos4_fast_tap'].generate(name='RTAPEND'+celltype+str(i), transform='R0' if transform_extra[i]=='0' else 'MX'))
                ltapbnd_name = 'ptap_fast_left' if celltype == 'n' else 'ntap_fast_left'
                rtapbnd_name = 'ptap_fast_right' if celltype == 'n' else 'ntap_fast_right'
                ltapbnd_list.append(templates[ltapbnd_name].generate(name='LTAPBND'+celltype+str(i), transform='R0' if transform_iter[i]=='0' else 'MX'))
                rtapbnd_list.append(templates[rtapbnd_name].generate(name='RTAPBND'+celltype+str(i), transform='R0' if transform_iter[i]=='0' else 'MX'))
                i+=1
        
        # Place TAPs on the design.
        if side == 'both':
            dsn.place(grid=pg, inst=np.array(rtap_list   ).reshape(len(rtap_list   ),1), mn=pg.mn.bottom_right(bbox))
            dsn.place(grid=pg, inst=np.array(rtapbnd_list).reshape(len(rtapbnd_list),1), mn=pg.mn.bottom_right(rtap_list[0]))
            dsn.place(grid=pg, inst=np.array(ltap_list   ).reshape(len(ltap_list   ),1), mn=pg.mn.bottom_left(bbox)         - pg.mn.width_vec(ltap_list[0]))
            dsn.place(grid=pg, inst=np.array(ltapbnd_list).reshape(len(ltapbnd_list),1), mn=pg.mn.bottom_left(ltap_list[0]) - pg.mn.width_vec(ltapbnd_list[0]))
        elif side == 'left':
            dsn.place(grid=pg, inst=np.array(ltap_list   ).reshape(len(ltap_list   ),1), mn=pg.mn.bottom_left(bbox)         - pg.mn.width_vec(ltap_list[0]))
            dsn.place(grid=pg, inst=np.array(ltapbnd_list).reshape(len(ltapbnd_list),1), mn=pg.mn.bottom_left(ltap_list[0]) - pg.mn.width_vec(ltapbnd_list[0]))
        elif side == 'right':
            dsn.place(grid=pg, inst=np.array(rtap_list   ).reshape(len(rtap_list   ),1), mn=pg.mn.bottom_right(bbox))
            dsn.place(grid=pg, inst=np.array(rtapbnd_list).reshape(len(rtapbnd_list),1), mn=pg.mn.bottom_right(rtap_list[0])) 
 
        bbox = dsn.bbox
 
        if bbox[0][0] != 0:
            for _dsn in dsn:
                dsn.elements[_dsn].xy -= [bbox[0][0],0]
        print('========= TAP GENERATION END =========')

    def generate_gbnd(self, dsn, grids, templates):
        """ 
        Generates global boundary structures around the design.
        (Check the name of gbnd cells since those are different by each template library)
        """
 
        # Call placement grid and calculate the bounding box of the design.
        pg = grids["placement_basic"]
        bbox_xy = dsn.bbox
        bbox_mn = pg.mn(dsn.bbox)
        
        # Call each dummy GBND cell from template library to calculate the height and width of each cell.
        gbnd_vertical_dmy   = templates["boundary_topleft"].generate(name="gbnd_vertical_dmy"  )
        gbnd_horizontal_dmy = templates["boundary_top"    ].generate(name="gbnd_horizontal_dmy")
        gbnd_corner_dmy     = templates["boundary_topleft"].generate(name="gbnd_corner_dmy"    )
        
        # Calculate the number of mosaic and generate GBND cells to be placed. 
        num_horizontal = bbox_mn[1][0]-bbox_mn[0][0]
        itop_gb   = templates["boundary_top"].generate( name="gbnd_top", transform='MX', shape=[num_horizontal, 1] )
        ibot_gb   = templates["boundary_top"].generate( name="gbnd_bot", transform='R0', shape=[num_horizontal, 1] ) 
 
        num_vertical = bbox_mn[1][1]//pg.mn.height(gbnd_vertical_dmy)
        ileft_gb  = templates["boundary_topleft"].generate( name="gbnd_left",  transform='R0', shape=[1, num_vertical] )
        iright_gb = templates["boundary_topleft"].generate( name="gbnd_right", transform='MY', shape=[1, num_vertical] )  
 
        ibl_gb    = templates["boundary_topleft"].generate( name="gbnd_bl", transform='R0'   )  
        ibr_gb    = templates["boundary_topleft"].generate( name="gbnd_br", transform='MY'   )  
        itr_gb    = templates["boundary_topleft"].generate( name="gbnd_tr", transform='R180' )  
        itl_gb    = templates["boundary_topleft"].generate( name="gbnd_tl", transform='MX'   )  
 
        # Place GBND cells on the design.    
        dsn.place(grid=pg, inst=itop_gb,   mn=pg.mn.top_left(    bbox_xy) + pg.mn.height_vec(gbnd_horizontal_dmy)) # TOP
        dsn.place(grid=pg, inst=ibot_gb,   mn=pg.mn.bottom_left( bbox_xy) - pg.mn.height_vec(gbnd_horizontal_dmy)) # BOTTOM
 
        dsn.place(grid=pg, inst=ileft_gb,  mn=pg.mn.bottom_left( bbox_xy) - pg.mn.width_vec( gbnd_vertical_dmy  )) # LEFT
        dsn.place(grid=pg, inst=iright_gb, mn=pg.mn.bottom_right(bbox_xy) + pg.mn.width_vec( gbnd_vertical_dmy  )) # RIGHT
  
        dsn.place(grid=pg, inst=ibl_gb,    mn=pg.mn.bottom_left( ibot_gb) - pg.mn.width_vec( gbnd_corner_dmy    )) # BOTTOM LEFT CORNER
        dsn.place(grid=pg, inst=ibr_gb,    mn=pg.mn.bottom_right(ibot_gb) + pg.mn.width_vec( gbnd_corner_dmy    )) # BOTTOM RIGHT CORNER
        dsn.place(grid=pg, inst=itl_gb,    mn=pg.mn.top_left(   ileft_gb) + pg.mn.height_vec(gbnd_corner_dmy    )) # TOP LEFT CORNER
        dsn.place(grid=pg, inst=itr_gb,    mn=pg.mn.top_right( iright_gb) + pg.mn.height_vec(gbnd_corner_dmy    )) # TOP RIGHT CORNER
 
        bbox_xy = dsn.bbox
 
        if bbox_xy[0][0] != 0:
            for _dsn in dsn:
                dsn.elements[_dsn].xy -= [bbox_xy[0][0],0]

    def generate_pwr_rail(self, dsn, grids, tlib=None, templates=None, route_type='cmos', netname=None, vss_name='VSS', vdd_name='VDD', rail_swap=False, vertical=False, pin_num=0, pin_pitch=0):
        """ 
        Generates thick wire rails for supplies (VDD, VSS)
            
        Parameters
        ----------
        route_type : str
            The type of routing style ('cmos', 'mos', 'cmos_flipped')
        netname : str or list
            The name of nets
            ex) 'VDD', ['VDD', 'VSS'], ['VDD', 'VSS', ['VDD', -1]]
        vss_name : str
            The name of GROUND net (will be deprecated)
        vdd_name : str
            the name of POWER net (will be deprecated)
        rail_swap : boolean
            Determine the bottom rail is GND or POWER net. 0 for GND 1 for POWER (will be deprecated).
        vertical : boolean
            whether generate vertical wires for connecting each horizontal rail
        pin_num : int
            the number of pins
        pin_pitch : int
            the pitch between pins in the abstract coordinate
        """
 
        print('=========== SUPPLY RAIL GENERATION START ===========')
        # 1. Load grids
        r23 = grids['routing_23_{0}'.format(route_type)]                                        # "CMOS grid" to calculate the number of power rails
        if route_type != 'mos': r23t = grids['routing_23_{0}_thick'.format(route_type)]         # "CMOS grid" to make M3 vertical rails.
        else: r23t = grids['routing_23_{0}'.format(route_type)]                                 # "MOS grid" to calculate the number of power rails
        
 
        # 2. Calculate the number of power rails in the design
        bbox = dsn.bbox
        grid_cnt = bbox[1,1] // r23.height
        bottom_rail = [r23.mn.bottom_left(bbox), r23.mn.bottom_right(bbox)]                     # bbox for M2 bottom rail
 
        if netname == None:
            """
                This block is for old-version users.
                It will be removed later. :D
            """
 
            print("\n[WARNING] generate_pwr_rail with vss_name/vdd_name arguments will be deprecated.\n")
            if grid_cnt%2 == 0:
                iter_vdd = grid_cnt//2
                iter_vss = grid_cnt//2 + 1
 
            else:
                iter_vdd = (grid_cnt+1)//2
                iter_vss = (grid_cnt+1)//2
 
            pw_len = 2
 
            rvss = []
            rvdd = []
            vss_set = [iter_vss, rvss, vss_name]
            vdd_set = [iter_vdd, rvdd, vdd_name]
 
            pw_set = np.array([vss_set, vdd_set])
 
            if rail_swap: 
                pw_set[0][2] = vdd_name
                pw_set[1][2] = vss_name
            
            pw_iter = grid_cnt + 1 
            rail_count = {}
            rail_change = {}
        
        else:
            # 3. Generate a power rail list.
            pw_iter = grid_cnt + 1 
 
            # Check the type of netname.
            if type(netname) == str:
                netname = netname.split()
                rail_change = {}
 
            else:
                rail_change = {}
                for _name in netname:
                    if type(_name) == list:
                        try: 
                            if _name[1] < 0: _name[1] += pw_iter
                            rail_change[_name[0]].append(_name[1])
                        except: 
                            rail_change[_name[0]]= [_name[1]]
            
            # Create a power rail list.
            netname = list(filter(lambda x: type(x)==str, netname))
            pw_len = len(netname)
            pw_set = np.empty(shape=(pw_len, 3), dtype=object)
 
            # Calculate the number of iterations of each power net.      
            pw_set[:pw_len,0] = pw_iter // pw_len
            pw_set[0:(pw_iter % pw_len), 0] += 1
 
            for i in range(pw_len): 
                pw_set[i, 1] = []
                pw_set[i, 2] = netname[i]
            
            # Rail swap for Iterated rails (Optional).
            if rail_swap: pw_set[:pw_len,2] = np.flip(pw_set[:pw_len,2])
 
            # Revise the list.
            for _name, _num in rail_change.items():
                for idx in _num:
                    pw_set[(idx%pw_len), 0] -= 1
                
                if _name in netname:
                    pin_ex = pw_set[:,2].tolist().index(_name)
                    pw_set[pin_ex, 0] += len(_num)
                
                else:
                    pw_extra = [len(_num), [], _name]
                    pw_set = np.append(pw_set, [pw_extra], axis=0)
 
            # Count the number of each rail.
            rail_count = {}
            for _name, _num in rail_change.items():
                for idx in _num:
                    rail_count[idx] = _name
            
            # Remove zero-rails.
            zero_list = list(np.where(pw_set[:,0]==0))[0]
            pw_set = np.delete(pw_set, zero_list, 0)
            pw_len -= len(zero_list)
 
        # 4. Generate iterated power rails
        pin_name = []
 
        for idx in range(pw_iter):
            pin = idx % pw_len
            iter = idx // pw_len
 
            # Generate a horizontal rail
            _mn = bottom_rail
            _mn[0][1] = r23.n(r23.height) * (pw_len*iter+pin)
            _mn[1][1] = r23.n(r23.height) * (pw_len*iter+pin)
 
            route = dsn.route(grid=r23, mn=_mn)
            
            # Check the netname.
            if idx in rail_count.keys():
                _netname = rail_count[idx]
                pin = pw_set[:,2].tolist().index(_netname)
                pw_set[pin][1].append(route)
                pin_name.append("{0}:".format(_netname))
            
            else: 
                pw_set[pin][1].append(route)
                pin_name.append(pw_set[pin][2]+':')
 
            # Generate the vertical vias (Optional).
            if vertical & (pw_set[pin][0] != 1):
                if pin % 2 == 0: dsn.via(grid=r23t, mn=r23t.mn.bottom_left(pw_set[pin][1][-1])-[pin,0])
                else: dsn.via(grid=r23t, mn=r23t.mn.bottom_right(pw_set[pin][1][-1])+[pin-1,0])
        
        bbox = dsn.bbox
 
        for idx in range(len(pw_set)):
            # Generate the vertical rails (Optional).
            if vertical & (pw_set[idx][0] != 1) :
                if idx % 2 == 0:
                    _mn = [r23t.mn.bottom_left(pw_set[idx][1][0])-[idx,0], r23t.mn.bottom_left(pw_set[idx][1][-1])-[idx,0]]
                else:
                    _mn = [r23t.mn.bottom_right(pw_set[idx][1][0])+[(idx-1),0], r23t.mn.bottom_right(pw_set[idx][1][-1])+[(idx-1),0]]
                dsn.route(grid=r23t, mn=_mn)
            
        
            # Rail extension (Optional).
            if vertical & any(1 < i for i in pw_set[:,0]):
                for x in range(len(pw_set[idx][1])):
                    pw_set[idx][1][x].xy[0][0] = bbox[0][0]
                    pw_set[idx][1][x].xy[1][0] = bbox[1][0]
 
        # 5. Check whether two variables (pin_num and pin_pitch) are entered properly.
        bottom_rail = [r23.mn.bottom_left(bbox), r23.mn.bottom_right(bbox)]
 
        if (pin_num == 0) & (pin_pitch == 0) : pin_num = 1 
        elif (pin_num < 0) | (pin_pitch < 0) :
            pin_num = 1
            pin_pitch = 0
            print('\n[WARNING] You entered negative number.\n')
        elif (pin_num != 0) & (pin_pitch != 0) : 
            pin_pitch = 0
            print('\n[WARNING] You have to choose between pin_num or pin_pitch.\nLAYGO2 follows <pin_num> this time.\n')
 
        pwidth = bottom_rail[1][0] - bottom_rail[0][0]
 
        # 6. Compare between pwidth and (pin_num / pin_pitch).
        if pin_num != 0: 
            if pin_num > pwidth:
                pin_num = pwidth
                pin_pitch = 1
                print('"You want too many pins. x_x')
                print('The maximum number of pins : {0}"\n'.format(pwidth))
            else: pin_pitch = pwidth//pin_num
        elif pin_pitch != 0: 
            if pin_pitch > pwidth: 
                pin_pitch = pwidth
                print('"You want too wide pitch. x_x')
                print("The minimum number of pin : 1")
                print('Thus, one pin is generated each."\n')
            pin_num = pwidth // pin_pitch
 
        # 7. Generate iterated power rails.
        for idx in range(pw_iter):
            pin = idx % pw_len
            iter = idx // pw_len
 
            _mn = bottom_rail
            _mn[0][1] = r23.n(r23.height) * (pw_len*iter+pin)
            _mn[1][1] = r23.n(r23.height) * (pw_len*iter+pin)
 
            pmin = _mn[0]
            pmax = _mn[1]
            pp = np.array([pmin, pmin]) + np.array([[(pwidth%pin_num)//2, 0], [(pwidth%pin_num)//2, 0]])
 
            for pn in range(0, pin_num):
                pp[1] += [pin_pitch, 0]
                dsn.pin(name="net{0}_{1}".format(idx, pn), grid=r23, mn=pp, netname=pin_name[idx])
                pp[0] += [pin_pitch, 0]
 
        # 8. Set X coordinate as zero.
        if bbox[0][0] != 0:
            for _dsn in dsn:
                dsn.elements[_dsn].xy -= [bbox[0][0],0]
 
        # 9. Print messages about the results.
        print('\nThe number of rails of your design is {0}.\n'.format(grid_cnt) + "The number of pins of your design is {0}.\n".format(len(pw_set)) + '\nName of Iterated net:')
 
        for i in range(pw_len): print('{0}. "{1}"'.format(i, pw_set[i][2]))
        if rail_change:
            print('\nName of Changed net:')
            for _num, _name in sorted(rail_count.items()): print('Rail #{0} :  "{1}"'.format(_num, _name))    
        print('\n============ SUPPLY RAIL GENERATION END ============')

    def extend_wire(self, dsn, layer='M4', target=500):
        """
        Extend routing wires to meet the design rules about area or width of wires. This function is executed as follows:
        1. Find matched rects with the given layer in design.
        2. Check the direction of rect (horizontal or vertical). Do nothing if the rect is zero sized.
        3. Calculate total width/height of the rect and check whether the rect violates the design rule.
        4. Calculate delta (amount of extension) and create rect with new extension but maintaining bbox.
        5. Append new rects to the design and delete old rects.
        
        Parameters
        ----------
        dsn : laygo2.object.database.Design
            Design to be implemented.
        layer : str
            Name of layer for extension.
        target : int
            Target width/height of wires to be extended.
        """
 
        rect_list = dsn.get_matchedrects_by_layer([layer, 'drawing'])    
        for rect in rect_list:
            if rect.height == 0 and rect.width == 0:
                direction = 'horizontal' if layer in ['M2','M4','M6'] else 'vertical'
                check = rect.width + 2*rect.hextension if direction == 'horizontal' else rect.height + 2*rect.vextension
                
                if check < target:
                    delta = (target - check)//2
                    if direction == 'horizontal':
                        p = laygo2.object.physical.Rect(xy=rect.bbox, layer=rect.layer, hextension=rect.hextension0+delta, 
                        vextension=rect.vextension0, color=rect.color)
                    else:
                        p = laygo2.object.physical.Rect(xy=rect.bbox, layer=rect.layer, hextension=rect.hextension0, 
                        vextension=rect.vextension0+delta, color=rect.color)
 
                    dsn.append(p)
 
                    for key, value in list(dsn.items()):
                        if value == rect:
                            del dsn.elements[key]
                        else:
                            pass
                else:
                    pass
 
            else:
                direction = 'horizontal' if rect.height == 0 else 'vertical'
                check = rect.width + 2*rect.hextension if direction == 'horizontal' else rect.height + 2*rect.vextension
 
                if check < target:
                    delta = (target - check)//2
                    if direction == 'horizontal':
                        hextension = round(rect.hextension+delta, -1)
                        p = laygo2.object.physical.Rect(xy=rect.bbox, layer=rect.layer, hextension=hextension, 
                        vextension=rect.vextension, color=rect.color)
                    else:
                        vextension = round(rect.vextension+delta, -1)
                        p = laygo2.object.physical.Rect(xy=rect.bbox, layer=rect.layer, hextension=rect.hextension, 
                        vextension=vextension, color=rect.color)
 
                    dsn.append(p)
 
                    for key, value in list(dsn.items()):
                        if value == rect:
                            del dsn.elements[key]
                        else:
                            pass
                else:
                    pass

    def fill_by_instance(self, dsn, grids, tlib, templates, inst_name:tuple, canvas_area="full", shape=[1,1], iter_type=("R0","MX"), pattern_direction='v', fill_sort='filler'):
        """ Fill empty layout space by given instances.
            
            Parameters
            ----------
            inst_name : tuple
                the name of the instance to fill empty layout space.
            canvas_area : "full" or list
                the range of the space to be filled.
            shape : list
                the shape of the given instances.
            iter_type : tuple
                Transform types of iterating instances.
            pattern_direction : str
                Determine the direction of iterating instances. ('v' : vertical, 'h' : horizontal)
            fill_sort : str
                the name of the created iterating instances.
        """
 
        print('\n=========== FILLING EMPTY LAYOUT SPACE START ===========')
 
        pg = grids["placement_basic"]
 
        # 1. Check the canvas.
        dsnbbox = pg.mn(dsn.bbox)
 
        offset  = dsnbbox[0]
        width   = dsnbbox[1][0] - 0
        height  = dsnbbox[1][1] - dsnbbox[0][1]
 
        canvas = np.zeros((height, width), dtype=int)
 
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
 
        if canvas_area != "full":
            bbox_l0 = pg.mn.bottom_left(canvas_area[0])
            bbox_l1 = pg.mn.bottom_right(canvas_area[0])
            bbox_r0 = pg.mn.top_left(canvas_area[1])
            bbox_r1 = pg.mn.top_right(canvas_area[1])
 
            width   = abs(bbox_r1[0] - bbox_l0[0])
            height  = abs(bbox_r1[1] - bbox_l0[1])
            offset  = bbox_l0
 
            canvas = canvas[bbox_l0[1]:bbox_r1[1],bbox_l0[0]:bbox_r1[0]]
        else:
            bbox_l0 = dsnbbox[0]
            bbox_l1 = dsnbbox[0] + [dsnbbox[1][0],0]
            bbox_r0 = dsnbbox[1] - [dsnbbox[1][0],0]
            bbox_r1 = dsnbbox[1]
 
        def check_name(inst_name, boundary_name):
            if type(inst_name) == str:
                inst_name = inst_name.split()
            else:
                for _name in inst_name:
                    if type(_name) == list:
                        try:
                            boundary_name[_name[0]].append(_name[1])
                        except:
                            boundary_name[_name[0]] = [_name[1]]
 
            inst_name = list(filter(lambda x: type(x)==str, inst_name))
            return inst_name, boundary_name
 
        # 2. Fill the empty space.
        boundary_name = {}
        inst_name, boundary_name = check_name(inst_name, boundary_name)
 
        filler   = templates[inst_name[0]].generate(name=fill_sort, transform="R0", shape=shape)
 
        f_height = pg.mn(filler)[1][1]
        f_width  = pg.mn(filler)[1][0]
        n_mod    = int(height / f_height)
        print(boundary_name)
        for _name, _num in boundary_name.items():
            for i in range(len(_num)):
                if _num[i] < 0: 
                    if pattern_direction == 'v': boundary_name[_name][i] += ((bbox_r0[1]-bbox_l1[1]) // f_height)
                    elif pattern_direction == 'h': boundary_name[_name][i] += ((bbox_r0[0]-bbox_l1[0]) // f_width)
 
        boundary_count = {}
        for _name, _num in boundary_name.items():
            for idx in _num:
                boundary_count[idx] = _name
 
        pattern_name = 0
        for y in range(n_mod):
            buffers = []
 
            if pattern_direction == 'v':
                if y in boundary_count.keys(): it = boundary_count[y]
                else: 
                    it = inst_name[pattern_name]
                    pattern_name = (pattern_name + 1) % len(inst_name)
            elif pattern_direction == 'h': pattern_name = 0
 
            for x in range(width):
                if canvas[f_height * y, x] == 0:
                    buffers.append(True)
 
                    if np.array_equal(buffers, [True] * f_width):
                        if pattern_direction == 'h':
                            h_num = int((x - (bbox_l1[0] - bbox_l0[0])) / f_width)
                            if h_num in boundary_count.keys(): 
                                it = boundary_count[h_num]
                            else: 
                                it = inst_name[pattern_name]
                                if h_num != 0: pattern_name = (pattern_name + 1) % len(inst_name)
                            
                        tf = iter_type[int(y % len(iter_type))]
                        _mn = np.asarray([x-f_width+1, y * f_height])+offset
                        if tf == "MX": _mn = _mn + [0, f_height]
                        dsn.place(grid=pg, inst=templates[it].generate(name=fill_sort + f"{x}_{y * f_height}", transform=tf, shape=shape), mn=_mn)
                        buffers = []
                else:
                    buffers = []
        print('============ FILLING EMPTY LAYOUT SPACE END ============')
        return canvas

    def generate_cut_layer(self, dsn, grid, grid_cut, layer: str, space_min: float, flip=0 ):
        """ place the instance(cut-via) when layer space violation occurs """
        from collections import defaultdict
        rects  = dsn.rects
        insts  = dsn.instances
        vinsts = dsn.virtual_instances
        xy     = dsn.bbox
        pins   = dsn.pins
 
        def get_ebbox(obj):
            """ refine and get bbox from the object  """
            ebbox = np.zeros( (5,2), dtype=np.int64 )
 
            if isinstance(obj, laygo2.object.physical.Rect): # apply hextensions
                ebbox[0] = obj.bbox[0] - np.array([obj.hextension, 0])
                ebbox[1] = obj.bbox[1] + np.array([obj.hextension, 0])
                ebbox[2] = ebbox[2] + np.array([ obj.hextension, 0 ])
                ebbox[3] = ebbox[3] + np.array([ 0, obj.vextension])
            else:
                ebbox[0:2] = obj.bbox
 
            if obj.bbox[0][1] != obj.bbox[1][1]: # obj has no-zero height
                h_avr = int( 0.5 * ( obj.bbox[1][1] +  obj.bbox[0][1] ) )
                ebbox[0][1] = h_avr
                ebbox[1][1] = h_avr
 
 
            return ebbox
 
        def place( xy_w, xy_e, obj_w, obj_e, grid_cut ):
            """ place the cut-via """
 
            mn_w = grid_cut.mn(xy_w)
            mn_e = grid_cut.mn(xy_e)
            #print("convert")
            #print(grid_cut.name)
            #print( grid_cut.range )
            #print(mn_w)
            #print(mn_e)
            mn_w = mn_w.astype(int)
            mn_e = mn_e.astype(int)
            #print(mn_w)
            #print(mn_e)
            mn_c = ( 0.5*(mn_w + mn_e) ).astype(int)
            dsn.via( grid=grid_cut, mn= mn_c )
 
        def check_space_ok( xw:float, xe:float, space:float ):
            """ check space is enough  """
            delta = xe - xw
            print("check space", end=" ")
            print(f"{xw}, {xe}, delta:{delta}, space:{space}")
 
            if 0 < delta < space: # error
                print("##cut")
                return False
            else:               # pass  or overlap
                #print("no cut")
                return True
 
        space_min_edge = space_min  ## for space at edge,
 
        objs_to_check = dsn.get_matched_rects_by_layer( [layer, "drawing"]   )
        objs_to_check.extend( dsn.get_matched_rects_by_layer( [layer, "pin"] ))
 
        objs_toppin = []
        for rname, rect in rects.items():
            if np.array_equal(rect.layer, [layer,"pin" ] ):
                objs_toppin.append(rect)
 
        for rname, pin in pins.items():
            if np.array_equal(pin.layer, [layer,"pin" ] ):
                objs_toppin.append(pin)
 
        ebboxs_to_check= []
        ebboxs_toppin  = []
 
        objset=( objs_to_check,  objs_toppin   )
        bboxs =( ebboxs_to_check, ebboxs_toppin  )
        for n, objs in enumerate( objset):
            _bboxs = bboxs[n]
            for i, obj in enumerate(objs):
                ebbox = get_ebbox(obj) # bl, tr ,hextension, vextension
                ebbox[4,:] = [ i,i ]   # object index
                _bboxs.append( ebbox )
        
        ebboxs_to_check = np.unique(ebboxs_to_check, axis=0)  ## sort by bottom-left-x and remove duplicated-xy
        
        y_box     = defaultdict(list)
        y_box_pin = defaultdict(list)
        for ebbox in ebboxs_to_check: # packed by y-axis , assuming rect has 0 height
            y_box[ ebbox[0][1] ].append( ebbox )
        
        for ebbox in ebboxs_toppin: # packed by y-axis , assuming rect has 0 height
            y_box_pin[ ebbox[0][1] ].append( ebbox )
 
        y_keys = y_box.keys()
 
        for key in y_keys: # scan start
            ebbox_list  = y_box[key]
            ebboxs_pin  = y_box_pin.get(key, 0 )
            i_last      = len(ebbox_list) - 1
 
            i_edges= (0, i_last)
            #print("##i_edge###")
            #print(i_edges)
            #print(key)
            
            ## check edge, pin-layer is on edge of design
            for n, i in enumerate( i_edges ):
                ebbox_edge = ebbox_list[i]
                flag_skip  = 0
                #print(ebbox_edge)
            
                if ebboxs_pin == 0 :
                    flag_skip  = 0
                else:
 
                    for ebbox_pin in ebboxs_toppin:
                        bl_pin, tr_pin   = ebbox_pin[0], ebbox_pin[1]
                        bl_edge, tr_edge = ebbox_edge[0], ebbox_edge[1]
                        tol = 0.1
                        # edge pin   e p e p
                        # pin edge   p e p e
                        #            e e p p
                        #            p p e e  
                        if  ( (bl_pin[0] <= tr_edge[0]) and ( tr_pin[0] >= bl_edge[0] )   or ( ( tr_pin[0]  >= tr_edge[0]  ) and bl_pin[0] <= tr_edge[0] )):
                            flag_skip  = 1
                            break
                        ## the route on the edge is a pin! skip
                
                if flag_skip ==1:
                    continue
                else:
                    #print( f" there is no pin on edge {n}")
                    if n == 0:
                    #     print("left")
                         flag = check_space_ok( xy[0][0], ebbox_edge[0][0], space_min_edge) # check left
                    else:
                    #    print("right")
                        flag = check_space_ok( ebbox_edge[1][0], xy[1][0], space_min_edge) # check right
                    if flag == False: # not ok
                        _xy = np.array([xy[n][0], ebbox_edge[0][1]]) # edge-xy of design
                        place( _xy, _xy, dsn, objs_to_check[ebbox_edge[4][0]], grid_cut)
                 
            #print("check middle")
            ## check middle
            if i_last != 0:
                iw_ebbox = ebbox_list[0]    # checkking bottom-right
                for i in range(i_last + 1): # from leftmost-right to rightmost-left
                    # ie : reference
                    # iw : target
                    new_ebbox = ebbox_list[i]
                    if new_ebbox[1][0] <= iw_ebbox[1][0] : # bottom-right vs bottom-right
                        continue # skip, it is overlapped
                    else: # evaluation
                        ie_ebbox = new_ebbox
                        flag     = check_space_ok( iw_ebbox[1][0],  ie_ebbox[0][0], space_min )
                        if flag == False :
                            print("###################### compensation")
                            _xy_w =  iw_ebbox[1] - iw_ebbox[2] # compensate extensions for mn converting
                            _xy_e =  ie_ebbox[0] + ie_ebbox[2] # compensate extensions for mn converting
 
                            place( _xy_w, _xy_e , objs_to_check[ iw_ebbox[4][0]], objs_to_check[ie_ebbox[4][0]], grid_cut )
                        iw_ebbox = ie_ebbox # update






    def post_process(self, dsn, grids, tlib, templates ):
        pass
        #generate_cut_layer(dsn, grids, tlib, templates)  

# Tests
if __name__ == '__main__':
    # Create templates.
    print("Create templates")
    _templates = load_templates()
    for tn, t in _templates.items():
        print(t)
