####################################################################################################################
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

from typing import Set, Tuple, List, Dict
from abc import *

from laygo2.object.template import UserDefinedTemplate
import laygo2.object.database

import numpy as np


class TileTemplate(UserDefinedTemplate, metaclass = ABCMeta):
    """ 
    The class for basic parametric cell. 
    The sub-elements will be placed by placement_pattern.

    generation flow:
        1. insert parameters by __init__()
        2. use self.generate_func, self.generete_pin, self.bbox for creating UserDefindTemplate
    """

    glib         = None  
    """The main grid libarary for routing."""
    
    tlib         = None  
    """The main template library for calling nativeTemplates."""
    
    placement_map: Dict[str, str]  = None  
    """The mapping of the name of sub-elements and the name of native template.
       the name of sub_elements = {"core", "bndr", "bndl", "gbndr", "gbndl"}.
    """

    placement_pattern: List[str] = None
    """The sequence of sub-elements placement."""
      
    transform_pattern: List[str] = None            
    """The sequencye of sub-elements transform: {"R0", "MX", "MY", R180"}."""
    
    routing_map: Dict[str, int]  = None  
    """The mapping of Mosfet terminal and r2 vertical track and other options for tracks.
       Mosfet terminal = {"G", "D", "S"},
       
    """
    
    routing_gname: str         = None  
    """The name of main routing grid."""

    nf_core: int               = None  
    """The core template number of finger."""

    libname:str                = None 
    """The name of Native template library."""

    def __init__(self, tlib, glib, routing_gname: str, routing_map: dict, placement_map: dict, placement_pattern: list, transform_pattern: list, name: str):
        
        self.tlib              = tlib
        self.glib              = glib
        self.routing_gname     = routing_gname     # mosfet grid name
        self.routing_map       = routing_map       # gate, drain, source trakcs
        self.placement_map     = placement_map     # native template name dictionary
        self.placement_pattern = placement_pattern # placement pattern
        self.transform_pattern = transform_pattern # transform pattern
        
        self.libname = tlib.libname
        self.nf_core   = 2
        
        super().__init__(name = name, bbox_func = self.bbox_func, pins_func = self.pins_func, generate_func = self.generate_func)

    def _route_pattern_via_track(self, grid, mn_list, n_track, name_via ) -> list:
        """ Route with multi instance via

        Parameters
        ----------
        mn_list: np.array 
            sorted and unique array
        n_track: int
            track number of n
        name_via: str
            the name of via
        """
        len_via = mn_list.shape[0]
        m_left  = mn_list[0][0] # left-most m
        tvia    = grid.viamap[m_left, n_track]
            
        if len_via == 1:
            via     = tvia.generate(name = name_via, shape = ( len_via, 1), pitch = [1, 1] )
            track   = None
        else:
            m_left2   = mn_list[1][0]
            pitch_m   = m_left2 - m_left
            pitch_x   = grid.x(pitch_m)

            via       = tvia.generate(name = name_via, shape = ( len_via, 1), pitch = [pitch_x, 0])
            track     = grid.route( mn = ( [mn_list[0][0], n_track ] , [mn_list[-1][0], n_track ]   ) )
            
        via.xy = grid.xy( [m_left, n_track ])

        return via, track

    def _update_params(self, params_in) -> dict:
        """ Update sub-elements parameters
        """

        params = {}

        if "switch" in params_in:
            nfs               = dict(core = "nf", dmyl = "nfdmyl", dmyr = "nfdmyr")
            flags             = params_in["switch"]
            sizes             = params_in["size"]
            placement_pattern = self.placement_pattern

            for i in len(flags):
                tname = placement_pattern[i]
                size  = sizes[i]
                flag  = flags[i] 
                if flag == True:
                    params_in[tname] = True
                    if tname in nfs:
                        params_in[ nfs[tname] ] = size   
                else:
                    params_in[tname] = False

        params["nf"]        = params_in.get("nf"     , self.nf_core )
        params["nfdmyl"]    = params_in.get("nfdmyl" , False )
        params["nfdmyr"]    = params_in.get("nfdmyr" , False )
        
        params["dmyl"] = True
        params["dmyr"] = True

        if params["nfdmyl"] == False:
            params["dmyl"]      = False

        if params["nfdmyr"] == False:
            params["dmyr"]      = False

        params["core"]      = params_in["nf"]

        params["bndl" ]     = params_in.get("bndl"   , True )
        params["bndr" ]     = params_in.get("bndr"   , True )
        
        params["gbndl" ]     = params_in.get("gbndl" , False)
        params["gbndr" ]     = params_in.get("gbndr" , False)
        
        params = self._update_params_sub(params_in, params)
        return params
    
    @abstractmethod
    def _update_params_sub(self, params_in, params):
        """ update custom flags """
        pass

    @abstractmethod
    def _mos_route(self, params):
        """ internal routing method """
        pass

    @abstractmethod
    def pins_func(self, params):
        """ pin generation method """
        pass

    def _generate_subinstance(self, params) -> dict:
        """The method of generating native_elements.
        """

        tlib            = self.tlib
        placement_map   = self.placement_map
        params    = self._update_params(params)
        nf_core   = self.nf_core
        tfs       = self.transform_pattern
        
        iparams   = dict()  # mapping sub-elements and generated instance.
        nf        = int( params["nf"] / nf_core) 
        
        if params["nfdmyl"] == False:
            nfdmyl = 1
        else:
            nfdmyl = params["nfdmyl"]

        if params["nfdmyr"] == False:
            nfdmyr = 1
        else:
            nfdmyr = params["nfdmyr"]

        tf_set = ("R0", "MY")

        if nf_core == 1: # not used, source and drain swap continously
            icore_sub = [0]* nf
            nelements_core = {}
            pins_core      = {}
            for i in range( nf):
                tf_core   = tf_set[ i%2]
                icore_sub = tlib[ placement_map["core"] ].generate( name=f'IM{i}'   , shape = [1, 1], transform = tf_core )
                nelements_core[f"IM{i}"] = icore_sub
                pins_core[f"D{i}"]   = icore_sub.pins["D"]
                pins_core[f"S{i}"]   = icore_sub.pins["S"]
                pins_core[f"G{i}"]   = icore_sub.pins["G"]
            icore  = laygo2.object.physical.VirtualInstance( xy = [0,0], libname = "dummy", cellname="dummy",
                                                             pins = pins_core,
                                                             name = "IM0", native_elements = nelements_core, transform = "R0"
                                                            )
        else:
            icore  = tlib[ placement_map["core"] ].generate( name='ICORE0', shape = [nf, 1] , transform = tfs["core"] )
        
        ibndl  = tlib[ placement_map["bndl"] ].generate( name='IBNDL0', shape = [1, 1] , transform = tfs["bndl"] )
        ibndr  = tlib[ placement_map["bndr"] ].generate( name='IBNDR0', shape = [1, 1] , transform = tfs["bndr"] )
        
        idmyl  = tlib[ placement_map["dmyl"] ].generate( name='IDMYL0',  shape = [max( int(0.5 * nfdmyl ),1)  , 1], transform = tfs["dmyl"] )
        idmyr  = tlib[ placement_map["dmyr"] ].generate( name='IDMYR0',  shape = [max( int(0.5 * nfdmyr) ,1)  , 1], transform = tfs["dmyr"] )

        igbndl = tlib[ placement_map["gbndl"] ].generate( name='IGDMYL0', shape = [1, 1], transform = tfs["gbndl"] )
        igbndr = tlib[ placement_map["gbndr"] ].generate( name='IGDMYR0', shape = [1, 1], transform = tfs["gbndr"] )

        iparams["gbndl"] = igbndl
        iparams["gbndr"] = igbndr
        
        iparams["dmyl"]  = idmyl
        iparams["dmyr"]  = idmyr

        iparams["bndl"]  = ibndl
        iparams["bndr"]  = ibndr
        
        iparams["core"]  = icore

        return iparams

    def _mos_place(self, params) -> dict:
        """ The method of placing native elements.
            It will modify instances.xy
        """

        params        = self._update_params( params )
        iparams       = self._generate_subinstance( params )
        
        placement_pattern = self.placement_pattern
        tfs               = self.transform_pattern

        # placement
        cursor = [0, 0]
        for i, t_inst in enumerate( placement_pattern ):
            if params[t_inst] != False:
                inst     = iparams[t_inst]
            
                if tfs[t_inst] == "MY":
                    xy_bl    = inst.bbox[0]  # this is on the left side
                    xy_tr    = inst.bbox[1]
                    inst.xy  = np.asarray( cursor ) + [ -xy_bl[0], 0 ]
                else:
                    inst.xy  = cursor
            
                xy_bl    = inst.bbox[0]
                xy_tr    = inst.bbox[1]
                cursor   = [ xy_tr[0], xy_bl[1] ] 
            else:
                iparams.pop(t_inst) # remove from dictionary
        
        return iparams

   
    def bbox_func(self, params) -> np.ndarray:
        params  = self._update_params(params)
        
        iparams = self._mos_place( params )

        xy_all = np.asarray([0,0])
        for t_inst, inst in iparams.items():
            xy = inst.bbox
            xy_bl     = xy[0]
            xy_tr     = xy[1]

            xy_all[0] = max( xy_all[0], xy_tr[0])
            xy_all[1] = max( xy_all[1], xy_tr[1])
        
        return xy_all
    
    def generate_func(self, name = None, shape = None, pitch = None, transform = 'R0', params = None):
        """ generate Virtualinstances and with routing
        """

        nelements = dict()
        params    = self._update_params( params )
        placement_map   = self.placement_map
        libname   = self.libname
        iparams   = self._mos_place( params )
        nelements.update( iparams )            
        
        routes    = self._mos_route( params )
        nelements.update( routes )            

        pins      = self.pins_func(params)
        nelements.update(pins)

        bbox      = self.bbox_func(params)

        # Generate and return the final instance
        inst = laygo2.object.VirtualInstance( name = name, xy = np.array([0, 0]) , libname = libname, cellname = f'myvcell_{name}' ,
                                              native_elements = nelements, shape = shape, pitch = pitch,
                                              transform       = transform, unit_size = bbox, pins = pins)
        
        return inst

class TileMOSTemplate(TileTemplate):

    def _update_params_sub(self, params_in, params):
        # about routing
        params["sdswap"]    = params_in.get("sdswap"   , False)  
        params["trackswap"] = params_in.get("trackswap", False)  
        params["tie"      ] = params_in.get("tie"      , False)  
        
        # new function
        params["ntrackswap"] = params_in.get("ntrackswap", False)  
        params["sdswap"]     = params_in.get("sdswap"    , False)  
        params["rail"     ]  = params_in.get("rail", True )  
        
        return params

    def pins_func(self, params):
        """The method of creating pin."""
        params  = self._update_params(params)
        pins    = dict()
        
        # generate a virtual routing structure for reference
        route_obj = self._mos_route( params )

        if 'RG0' in route_obj:  # gate
            g_obj = route_obj['RG0']
            pins['G'] = laygo2.object.Pin(xy=g_obj.xy, layer=g_obj.layer, netname='G')
        if 'RD0' in route_obj:  # drain
            d_obj = route_obj['RD0']
            pins['D'] = laygo2.object.Pin(xy=d_obj.xy, layer=d_obj.layer, netname='D')
        if 'RS0' in route_obj:  # source
            s_obj = route_obj['RS0']
            pins['S'] = laygo2.object.Pin(xy=s_obj.xy, layer=s_obj.layer, netname='S')
        if 'RRAIL0' in route_obj:  # rail
            r_obj = route_obj['RRAIL0']
            pins['RAIL'] = laygo2.object.Pin(xy=r_obj.xy, layer=r_obj.layer, netname='RAIL')
        return pins

    def _mos_route(self, params) -> dict:
        """The method of routing mosfet"""

        params            = self._update_params(params)
        glib              = self.glib
        routing_gname     = self.routing_gname
        routing_map       = self.routing_map
        nf_core           = self.nf_core
        iparams           = self._mos_place(params)
        placement_pattern = self.placement_pattern

        r12 = glib[routing_gname]
        n_g = routing_map["G"]
        n_d = routing_map["D"]
        n_s = routing_map["S"]
        n_r = routing_map.get( "RAIL", 0 )

        nelements = {}
        nelements.update(iparams)
        
        icore   = iparams["core"]
        nf      = params["nf"]

        G, S, D = "G", "S", "D"
        
        if params["trackswap"]: #backward compatable
            n_d, n_s = n_s, n_d
            S, D     = D, S
        
        if params["ntrackswap"]: 
            n_d, n_s = n_s, n_d
        
        if params["sdswap"]: 
            S, D     = D, S

        # Extrack mn from core mosfet
        mn_list_g = [] 
        mn_list_d = [] 
        mn_list_s = []
        i_iter = int( nf / nf_core) 

        for i in range( i_iter ):
            if i_iter == 1:
                icore_sub =  icore
            else:
                icore_sub =  icore[i][0] # it is 2-dimensional array
                
            for pin_name in icore_sub.pins.keys():

                if G in pin_name:
                    mn_list_g.append( r12.mn.center( icore_sub.pins[pin_name] ) )
                elif D in pin_name:
                    mn_list_d.append( r12.mn.center( icore_sub.pins[pin_name] ) )
                elif S in pin_name:
                    mn_list_s.append( r12.mn.center( icore_sub.pins[pin_name] ) )
        

        mn_list_g = np.asarray( mn_list_g)
        mn_list_g = mn_list_g[  mn_list_g[:,0].argsort()] # sort list by bottom_left
        mn_list_g = np.unique(  mn_list_g, axis = 0 )

        mn_list_d = np.asarray( mn_list_d)
        mn_list_d = mn_list_d[  mn_list_d[:,0].argsort()] 
        mn_list_d = np.unique(  mn_list_d, axis = 0 )
        
        mn_list_s = np.asarray( mn_list_s)
        mn_list_s = mn_list_s[  mn_list_s[:,0].argsort()] 
        mn_list_s = np.unique(  mn_list_s, axis = 0 )

        # Create via and track
        r_g = dict( via = 0 , track = 0) 
        r_d = dict( via = 0 , track = 0) 
        r_s = dict( via = 0 , track = 0) 

        for mn_list, n_track, terminal, r_dict in zip( (mn_list_g, mn_list_d, mn_list_s), (n_g, n_d, n_s), (G, D, S), (r_g, r_d, r_s) ):
            
            via, track = self._route_pattern_via_track( r12, mn_list, n_track, terminal)
            r_dict["via"]   = via
            r_dict["track"] = track 

        # Modify track
        if r_g["track"] == None: 
            mn_sub       = mn_list_g[0]
            r_g["track"] = r12.route( mn = [ mn_sub + [0,0], mn_sub + [0,0] ], via_tag =[None, None] )  # new track
            
            x_extl        = routing_map["G_extension0_x"][0]
            x_extr        = routing_map["G_extension0_x"][1]
            if x_extl != None:
                xy_sub     = r12.xy(mn_sub)
                r_g["track"].xy = [ xy_sub + [ x_extl , 0], xy_sub + [x_extr, 0] ] # new track location

        if r_d["track"] == None: 
            mn_sub    = mn_list_d[0]
            mn_sub[1] = n_d
            
            m_ext = routing_map["D_extension0_m"]
            m_extl = 0
            m_extr = 0
            if m_ext[0] != None:
                m_extl = m_ext[0]
                m_extr = m_ext[1]

            r_d["track"]   = r12.route( mn = [ mn_sub + [m_extl,0], mn_sub + [m_extr,0] ], via_tag =[None, None] )

        if r_s["track"] == None: 
            mn_sub    = mn_list_s[0]
            mn_sub[1] = n_s
            
            m_ext = routing_map["S_extension0_m"]
            m_extl = 0
            m_extr = 0
            if m_ext[0] != None:
                m_extl = m_ext[0]
                m_extr = m_ext[1]
            r_s["track"]   = r12.route( mn = [ mn_sub + [m_extl,0], mn_sub + [m_extr,0] ], via_tag =[None, None] )

        nelements["VIA_G"] = r_g["via"]
        nelements["VIA_D"] = r_d["via"]
        nelements["VIA_S"] = r_s["via"]

        nelements["RG0"] = r_g["track"]
        nelements["RD0"] = r_d["track"]
        nelements["RS0"] = r_s["track"]

        # Rail routing
        if params["rail"]:

            tinst_l = 0 # left most sub-element name
            tinst_r = 0 # right most sub-element name

            for i, tinst in enumerate(placement_pattern):
                name = placement_pattern[i]
                if name in iparams:
                    if name != "gbndl":
                        tinst_l = name
                        break   
            
            for i, tinst in enumerate(placement_pattern):
                name = placement_pattern[-1 - i]
                if name in iparams:
                    if name != "gbndr":
                        tinst_r = name
                        break   
                    
            inst_l = iparams[tinst_l]
            inst_r = iparams[tinst_r]

            xy_bl = inst_l.bbox[0]
            xy_tr = inst_r.bbox[1]
            xy_br = [xy_tr[0], xy_bl[1] ]

            mn_bl = r12.xy >= xy_bl 
            mn_br = r12.xy <= xy_br 

            r_track = r12.route( mn = [mn_bl, mn_br], via_tag = [None, None] )
            nelements["RRAIL0"] = r_track

        # TIE
        if params["tie"] != False:

            if params["tie"] == "S" or params["tie"] == True:
                del nelements["RS0"]
                del nelements["VIA_S"]
                
                via, track = self._route_pattern_via_track(r12, mn_list_s, n_r, "VIA_TIE")
                r_t        = r12.route_via_track( mn = mn_list_s, via_tag = [ False, False], track = [None, n_r])
            
            elif params["tie"] == "D":
                del nelements["RD0"]
                del nelements["VIA_D"]
                
                via, track = self._route_pattern_via_track(r12, mn_list_d, n_r, "VIA_TIE")
                r_t        = r12.route_via_track( mn = mn_list_d, via_tag = [ False, False], track = [None, n_r])

            else:
                raise Exception(" Value error for tie")
            
            nelements["VIA_TIE"] = via
            
            if r_t[-1] == None:
                del r_t[1]
            for i, r in enumerate(r_t): # source or drain to rail
                nelements[f"RECT_TIE{i}"] = r

        # dmy routing

        names_dmy = ["dmyl", "dmyr"]
        for i, tdmy_nf in enumerate( ( "nfdmyl", "nfdmyr" )):
            if params[tdmy_nf] == 0:
                continue
            name_dmy  = names_dmy[i]
            icore     = iparams[ name_dmy ]
            mn_list_g = [] 
            mn_list_d = [] 
            mn_list_s = []
            i_iter = int( params[tdmy_nf] / nf_core) 

            for i in range( i_iter ):
                if i_iter == 1:
                    icore_sub =  icore
                else:
                    icore_sub =  icore[i][0] # it is 2-dimensional array
                
                for pin_name in icore_sub.pins.keys():

                    if G in pin_name:
                        mn_list_g.append( r12.mn.center( icore_sub.pins[pin_name] ) )
                    elif D in pin_name:
                        mn_list_d.append( r12.mn.center( icore_sub.pins[pin_name] ) )
                    elif S in pin_name:
                        mn_list_s.append( r12.mn.center( icore_sub.pins[pin_name] ) )

                mn_list_a = mn_list_g + mn_list_s + mn_list_d
                mn_list_a = np.asarray( mn_list_a)
                mn_list_a = mn_list_a[  mn_list_a[:,0].argsort()] # sort list by bottom_left
                mn_list_a = np.unique(  mn_list_a, axis = 0 )

                r_t        = r12.route_via_track( mn = mn_list_a, via_tag = [ False, False], track = [None, n_r])
                
                mn_list_a[:,1] = n_r
                mn_list_a = np.unique(  mn_list_a, axis = 0 )
                via, track = self._route_pattern_via_track(r12, mn_list_a, n_r, f"VIA_DUMMY_{icore.name}")
                
                nelements[f"VIA_TIE_{name_dmy}"] = via
                
                if r_t[-1] == None:
                    del r_t[1]
                for i, r in enumerate(r_t): # source or drain to rail
                    nelements[f"RECT_DMY_TIE_{name_dmy}_{i}"] = r

        return nelements
  

class TileTapTemplate(TileTemplate):
    """The class of TileTap."""

    def _update_params_sub(self, params_in, params):
        # about routing
        params["sdswap"]    = params_in.get("sdswap"   , False)  
        params["trackswap"] = params_in.get("trackswap", False)  
        params["tie"      ] = params_in.get("tie"      , False)  
        
        # new function
        params["ntrackswap"] = params_in.get("ntrackswap", False)  
        params["sdswap"]     = params_in.get("sdswap"    , False)  
        params["rail"     ]  = params_in.get("rail", True )  
        
        return params

    def pins_func(self, params):
        """The method of creating pin."""
        params  = self._update_params(params)
        pins    = dict()
        
        # generate a virtual routing structure for reference
        route_obj = self._mos_route( params )

        if 'RG0' in route_obj:  # gate
            g_obj = route_obj['RG0']
            pins['G'] = laygo2.object.Pin(xy=g_obj.xy, layer=g_obj.layer, netname='G')
        if 'RD0' in route_obj:  # drain
            d_obj = route_obj['RD0']
            pins['D'] = laygo2.object.Pin(xy=d_obj.xy, layer=d_obj.layer, netname='D')
        if 'RS0' in route_obj:  # source
            s_obj = route_obj['RS0']
            pins['S'] = laygo2.object.Pin(xy=s_obj.xy, layer=s_obj.layer, netname='S')
        if 'RRAIL0' in route_obj:  # rail
            r_obj = route_obj['RRAIL0']
            pins['RAIL'] = laygo2.object.Pin(xy=r_obj.xy, layer=r_obj.layer, netname='RAIL')
        return pins


    def _mos_route(self, params):
        params    = self._update_params(params)
        glib      = self.glib
        routing_gname = self.routing_gname
        routing_map = self.routing_map
        nf_core   = self.nf_core
        iparams   = self._mos_place(params)

        r12 = glib[routing_gname]
        n_g = routing_map["G"]
        n_d = routing_map["D"]
        n_s = routing_map["S"]
        n_r = routing_map.get( "RAIL", 0 )

        nelements = {}
        nelements.update(iparams)
        
        icore   = iparams["core"]
        nf      = params["nf"]

        G, S, D = "G", "S", "D"
        
        if params["trackswap"]: #backward compatable
            n_d, n_s = n_s, n_d
            S, D     = D, S
        
        if params["ntrackswap"]: 
            n_d, n_s = n_s, n_d
        
        if params["sdswap"]: 
            S, D     = D, S

        mn_list_s = [] # TAP0
        mn_list_d = [] # TAP1

        i_iter = int( nf / nf_core) 
        for i in range( i_iter ):
            if i_iter == 1:
                icore_sub =  icore
            else:
                icore_sub =  icore[i][0] # it is 2-dimensional array
                
            for pin_name in icore_sub.pins.keys():
                if "TAP0" in pin_name:
                    mn_list_s.append( r12.mn.center( icore_sub.pins[pin_name] ) )
                elif "TAP1" in pin_name:
                    mn_list_d.append( r12.mn.center( icore_sub.pins[pin_name] ) )
        
        mn_list_s = np.asarray( mn_list_s)
        mn_list_s = mn_list_s[  mn_list_s[:,0].argsort()] 
        mn_list_s = np.unique(  mn_list_s, axis = 0 )
        
        mn_list_d = np.asarray( mn_list_d)
        mn_list_d = mn_list_s[  mn_list_d[:,0].argsort()] 
        mn_list_d = np.unique(  mn_list_d, axis = 0 )

        r_s = dict( via = 0 , track = 0) 
        r_d = dict( via = 0 , track = 0) 

        for mn_list, n_track, terminal, r_dict in zip( (mn_list_s,mn_list_d), (n_s, n_d), ( D, S), (r_d, r_s) ):
            via, track = self._route_pattern_via_track( r12, mn_list, n_track, terminal)
            r_dict["via"]   = via
            r_dict["track"] = track 

        # Modify track
        if r_d["track"] == None: 
            mn_sub    = mn_list_d[0]
            mn_sub[1] = n_d
            
            m_ext = routing_map["D_extension0_m"]
            m_extl = 0
            m_extr = 0
            if m_ext[0] != None:
                m_extl = m_ext[0]
                m_extr = m_ext[1]

            r_d["track"]   = r12.route( mn = [ mn_sub + [m_extl,0], mn_sub + [m_extr,0] ], via_tag =[None, None] )

        if r_s["track"] == None: 
            mn_sub    = mn_list_s[0]
            mn_sub[1] = n_s
            
            m_ext = routing_map["S_extension0_m"]
            m_extl = 0
            m_extr = 0
            if m_ext[0] != None:
                m_extl = m_ext[0]
                m_extr = m_ext[1]
            r_s["track"]   = r12.route( mn = [ mn_sub + [m_extl,0], mn_sub + [m_extr,0] ], via_tag =[None, None] )

        nelements["VIA_D"] = r_d["via"]
        nelements["VIA_S"] = r_s["via"]

        nelements["RD0"] = r_d["track"]
        nelements["RS0"] = r_s["track"]

        # Rail routing
        if params["rail"]:
            xy    = icore.bbox
            xy_bl = xy[0]
            xy_tr = xy[1]
            xy_br = [xy_tr[0], xy_bl[1] ]
            mn_bl = r12.xy >= xy_bl 
            mn_br = r12.xy <= xy_br 

            r_track = r12.route( mn = [mn_bl, mn_br], via_tag = [None, None] )
            nelements["RRAIL0"] = r_track

        # TIE
        if params["tie"] != False:

            if params["tie"] == "TAP0" or params["tie"] == True:
                del nelements["RS0"]
                del nelements["VIA_S"]
                
                via, track = self._route_pattern_via_track(r12, mn_list_s, n_r, "VIA_TIE")
                r_t        = r12.route_via_track( mn = mn_list_s, via_tag = [ False, False], track = [None, n_r])
            
            elif params["tie"] == "TAP1":
                del nelements["RD0"]
                del nelements["VIA_D"]
                
                via, track = self._route_pattern_via_track(r12, mn_list_d, n_r, "VIA_TIE")
                r_t        = r12.route_via_track( mn = mn_list_d, via_tag = [ False, False], track = [None, n_r])

            else:
                raise Exception(" Value error for tie")
            
            nelements["VIA_TIE"] = via

            if r_t[-1] == None:
                del r_t[1]
            for i, r in enumerate(r_t):
                nelements[f"RECT_TIE{i}"] = r

        return nelements
