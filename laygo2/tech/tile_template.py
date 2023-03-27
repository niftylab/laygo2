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

from typing import Set, Tuple, List, Dict

from laygo2.object.template import UserDefinedTemplate
import laygo2.object.physical
import laygo2.object.database

import numpy as np

class TileMosfetTemplate(UserDefinedTemplate):
    """ 
    The class for basic mosfet cell. 
    The mosfet sub-elements will be placed by placement_pattern.

    generation flow:
        1. inserted parameters by __init__()
        2. self.generate_func, self.generete_pin, self.bbox are used for creating UDFtemplate
    """

    grids             = 0  
    """The main grid libarary for routing."""
    
    templates         = 0  
    """The main template library for calling nativeTemplates."""
    
    tparams: Dict[str, str]  = 0  
    """The mapping of the name of sub-elements and the name of native template.
       the name of sub_elements = {"core", "bndr", "bndl", "gbndr", "gbndl"}.
    """

    placement_pattern: List[str] = 0
    """The sequence of sub-elements placement."""
      
    transform_pattern: List[str] = 0            
    """The sequencye of sub-elements transform: {"R0", "MX", "MY", R180"}."""
    
    r2_tracks: Dict[str, int]         = 0  
    """The mapping of Mosfet terminal and r2 vertical track.
       Mosfet terminal = {"G", "D", "S"}
    """
    
    grid_name: str         = 0  
    """The name of main routing grid."""

    nf_core: int           = 1  
    """The core template number of finger."""

    libname:str           = 0 
    """The name of Native template library."""

    def __init__(self, templates, grids, grid_name, r2_tracks, tparams, placement_pattern, transform_pattern, name):
        self.templates = templates

        self.grids     = grids
        self.grid_name = grid_name # mosfet grid name
        self.r2_tracks = r2_tracks # gate, drain, source trakcs
        
        self.tparams           = tparams           # native template name dictionary
        self.placement_pattern = placement_pattern # placement pattern
        self.transform_pattern = transform_pattern # transform pattern
        
        self.libname = templates.libname
        self.nf_core   = 2
        
        super().__init__(name = name, bbox_func = self.bbox_func, pins_func = self.pins_func, generate_func = self.generate_func)

    def _unpack_list(self, list_to_unpack, postfix:str) -> dict:
        """ The recursive function for unpacking list.
            This is used for flatten list that derived from route functions
        
        Parameters
        ----------
        list_to_unpack: array_like
            The list to be unpacked.
        postfix: str
            The postfix of name. unpacked sub-elements should have the unique name for dictionary.
        """

        def _sub_unpack( nelement_sub, element ):
            """The main recursive function. Update nelements_sub"""

            i_postfix = len(nelement_sub) + 1
            if isinstance( element, (list, np.ndarray) ):
                for element2 in element:
                    _sub_unpack( nelement_sub, element2) 
            else: # main purpose
                nelement_sub[f"via_or_rect_{i_postfix}_{postfix}"] = element

        nelement_sub = {}
        _sub_unpack( nelement_sub, list_to_unpack)

        return nelement_sub

    def _update_params(self, params_in) -> dict:
        """ Update mosfet parameters
        """

        params = {}

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
        
        # about routing
        params["sdswap"]    = params_in.get("sdswap"   , False)  
        params["trackswap"] = params_in.get("trackswap", False)  
        params["tie"      ] = params_in.get("tie"      , False)  
        params["rail"     ] = params_in.get("rail", True )  
        
        return params

    def _generate_subinstance(self, params) -> dict:
        """The method of generating native_elements.
        """

        templates = self.templates
        tparams   = self.tparams
        params    = self._update_params(params)
        nf_core   = self.nf_core
        tfs       = self.transform_pattern
        
        iparams   = dict()  # mapping sub-elements nand and generated instance.
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
                icore_sub = templates[ tparams["core"] ].generate( name=f'IM{i}'   , shape = [1, 1], transform = tf_core )
                nelements_core[f"IM{i}"] = icore_sub
                pins_core[f"D{i}"]   = icore_sub.pins["D"]
                pins_core[f"S{i}"]   = icore_sub.pins["S"]
                pins_core[f"G{i}"]   = icore_sub.pins["G"]
            icore  = laygo2.object.physical.VirtualInstance( xy = [0,0], libname = "dummy", cellname="dummy",
                                                             pins = pins_core,
                                                             name = "IM0", native_elements = nelements_core, transform = "R0"
                                                            )
        else:
            icore  = templates[ tparams["core"] ].generate( name='ICORE0', shape = [nf, 1] , transform = tfs["core"] )
        
        ibndl  = templates[ tparams["bndl"] ].generate( name='IBNDL0', shape = [1, 1] , transform = tfs["bndl"] )
        ibndr  = templates[ tparams["bndr"] ].generate( name='IBNDR0', shape = [1, 1] , transform = tfs["bndr"] )
        
        idmyl  = templates[ tparams["dmyl"] ].generate( name='IDMYL0',  shape = [nfdmyl, 1], transform = tfs["dmyl"] )
        idmyr  = templates[ tparams["dmyr"] ].generate( name='IDMYR0',  shape = [nfdmyr, 1], transform = tfs["dmyr"] )

        igbndl = templates[ tparams["gbndl"] ].generate( name='IGDMYL0', shape = [1, 1], transform = tfs["gbndl"] )
        igbndr = templates[ tparams["gbndr"] ].generate( name='IGDMYR0', shape = [1, 1], transform = tfs["gbndr"] )

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

    def _mos_route(self, params) -> dict:
        """The method of routing mosfet"""

        params    = self._update_params(params)
        grids     = self.grids
        grid_name = self.grid_name
        r2_tracks = self.r2_tracks
        nf_core   = self.nf_core
        iparams   = self._mos_place(params)
        
        r12 = grids[grid_name]
        n_g = r2_tracks["G"]
        n_d = r2_tracks["D"]
        n_s = r2_tracks["S"]
        
        nelements = {}
        nelements.update(iparams)
        
        icore   = iparams["core"]
        nf      = params["nf"]

        G, S, D = "G", "S", "D"

        if params["trackswap"]:
            n_d, n_s = n_s, n_d
            S, D     = D, S

        # Core gate routing
        mn_list_g = [] 
        mn_list_d = [] 
        mn_list_s = []
        i_iter = int( nf / nf_core) 
        for i in range( i_iter ):
            if i_iter == 1:
                icore_sub =  icore
            else:
                if nf_core == 2:
                    icore_sub =  icore[i][0] # it is 2-dimensional array
                else:
                    icore_sub =  icore.native_elements[f"IM{i}"]
                
            for pin_name in icore_sub.pins.keys():
                if "G" in pin_name:
                    mn_list_g.append( r12.mn.center( icore_sub.pins[pin_name] ) )
                elif D in pin_name:
                    mn_list_d.append( r12.mn.center( icore_sub.pins[pin_name] ) )
                elif S in pin_name:
                    mn_list_s.append( r12.mn.center( icore_sub.pins[pin_name] ) )
        
        r_g = r12.route_via_track( mn = mn_list_g , track = [ None, n_g], via_tag = [False, True] )
        r_d = r12.route_via_track( mn = mn_list_d , track = [ None, n_d], via_tag = [False, True] )
        r_s = r12.route_via_track( mn = mn_list_s , track = [ None, n_s], via_tag = [None, True] )
        
        if r_g[-1] == None: # When there is no gate track for core routing
            mn_sub     = mn_list_g[0]
            mn_sub[1]  = n_g
            r_g        = [  r12.via( mn = mn_sub) ]
             
            r_g.append( r12.route( mn = [ mn_sub + [0,0], mn_sub + [0,0] ], via_tag =[None, None] ) ) # new track
            #r_g.append( r12.route( mn = [ mn_sub + [-1,0], mn_sub + [1,0] ], via_tag =[None, None] ) ) # new track
            
            y_org      = r_g[-1].xy[0][1]
            x_ext      = r_g[-1].hextension
            #r_g[-1].xy = [ [aaa + x_ext , y_org], [ bbb - x_ext, y_org]] # new track location
        
        if r_d[-1] == None: # When there is no drain track for core routing
            mn_sub    = mn_list_d[0]
            mn_sub[1] = n_d
            r_d[-1]   = r12.route( mn = [ mn_sub + [0,0], mn_sub + [0,0] ], via_tag =[None, None] )
            #r_d[-1]   = r12.route( mn = [ mn_sub + [-1,0], mn_sub + [1,0] ], via_tag =[None, None] )
        
        if r_s[-1] == None: # When there is no drain track for core routing
            mn_sub    = mn_list_s[0]
            mn_sub[1] = n_s
            r_s[-1]   = r12.route( mn = [ mn_sub + [0,0], mn_sub + [0,0] ], via_tag =[None, None] )
            #r_s[-1]   = r12.route( mn = [ mn_sub + [-1,0], mn_sub + [1,0] ], via_tag =[None, None] )

        nelement_sub = {}
        nelement_sub = self._unpack_list( r_d, "D")
        nelements.update(nelement_sub)
        
        nelement_sub = {}
        nelement_sub = self._unpack_list( r_s, "S")
        nelements.update(nelement_sub)

        nelement_sub = {}
        nelement_sub = self._unpack_list( r_g, "G")
        nelements.update(nelement_sub)

        nelements["RG0"] = r_g[-1]
        nelements["RD0"] = r_d[-1]
        nelements["RS0"] = r_s[-1]

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
            if params["tie"] == "S":
                r_t = r12.route_via_track( mn = mn_list_s, via_tag = [ False, True], track = [None, r12.mn.center( r_track)[1] ])
            
            elif params["tie"] == "D":
                r_t = r12.route_via_track( mn = mn_list_d, via_tag = [ False, True], track = [None, r12.mn.center( r_track)[1] ])

            elif params["tie"] == True:
                r_t = r12.route_via_track( mn = mn_list_s, via_tag = [ False, True], track = [None, r12.mn.center( r_track)[1] ])

            else:
                raise Exception
            
            if r_t[-1] == None:
                del r_t[1]
            nelement_sub = self._unpack_list( r_t, "TIE")
            nelements.update(nelement_sub)
        
        return nelements
  
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
        tparams   = self.tparams
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

class TileTapTemplate(TileMosfetTemplate):
    """The class of TileTap."""

    def _mos_route(self, params):
        params    = self._update_params(params)
        grids     = self.grids
        grid_name = self.grid_name
        r2_tracks = self.r2_tracks
        nf_core   = self.nf_core
        iparams   = self._mos_place(params)

        r12       = grids[grid_name]
        nelements = {}
        nelements.update(iparams)
        
        icore   = iparams["core"]
        nf      = params["nf"]

        mn_list_s = []
        i_iter = int( nf / nf_core) 
        for i in range( i_iter ):
            if i_iter == 1:
                icore_sub =  icore
            else:
                if nf_core == 2:
                    icore_sub =  icore[i][0] # it is 2-dimensional array
                else:
                    icore_sub =  icore.native_elements[f"IM{i}"]
                
            for pin_name in icore_sub.pins.keys():
                if "TAP0" in pin_name:
                    mn_list_s.append( r12.mn.center( icore_sub.pins[pin_name] ) )
                elif "TAP2" in pin_name:
                    mn_list_s.append( r12.mn.center( icore_sub.pins[pin_name] ) )
        
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
            
            if params["tie"] == True:
                r_t = r12.route_via_track( mn = mn_list_s, via_tag = [ False, True], track = [None, r12.mn.center( r_track)[1] ])

            else:
                raise Exception
            nelement_sub = self._unpack_list( r_t, "TIE")
            nelements.update(nelement_sub)

        return nelements
