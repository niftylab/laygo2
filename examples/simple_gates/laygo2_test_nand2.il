; laygo2 layout export skill script.

; laygo2 skill export script.
; - Initial code from BAG_framework (https://github.com/ucb-art/BAG_framework).
; - Users need to install CCSinvokeCdfCallbacks() from Cadence Online Support
;   (support.cadence.com) to enable pcell export.

; create or reset a library
procedure( _laygo2_reset_library(libname tech_libname "tt")
    let( (lib_obj cell)
        if( lib_obj = ddGetObj(libname nil nil nil nil "r") then
            ; if the library exists, delete all cells in the library
            foreach( cell lib_obj~>cells
                unless( ddDeleteObj(cell)
                    error("[LAYGO2] Cannot delete cell %s in library %s\n" cell~>name libname)
                );unless
            );foreach
            ddReleaseObj(lib_obj)
            't
        else
            ; create library if not exist
            lib_obj = ddCreateLib(libname)
            techBindTechFile(lib_obj tech_libname)  ; attach technology file
            ddReleaseObj(lib_obj)  ; close library
            't
        );if
    );let
);procedure

; create a new layout view and open it
procedure( _laygo2_open_layout(libname cellname viewname "tt")
    let( (cv)
        ; open layout cellview
        unless( cv = dbOpenCellViewByType( libname cellname viewname "maskLayout" "w" )
            error("[LAYGO2] Cannot create a new layout cell %s__%s (%s)." libname cellname viewname)
        ); unless
        printf("Created a layout %s__%s (%s)\n" libname cellname viewname)
    cv
    );let
);procedure

; create a new layout view and open it
procedure( _laygo2_save_and_close_layout(cv)
    dbSave(cv)
    dbClose(cv)
);procedure

; create an instance
procedure( _laygo2_generate_instance( cv name libname cellname viewname loc orient num_rows num_cols sp_rows sp_cols
                                      params params_order )
    let( (obj)
        if( inst_cv = dbOpenCellViewByType( libname cellname viewname nil "r" ) then  ; if the instance exists
            if( and( num_rows==1 num_cols==1 ) then  ; create a single instance
                if( params != nil then  ; for pcells
                    obj = dbCreateParamInst(cv inst_cv name loc orient 1 params)
                    when( obj
                        if( params_order == nil then  ; if parameter order is not specified,
                            params_order = mapcar( lambda( (x) car(x) ) params )
                        );if
                        abInvokeCdfCallbacks(obj ?order params_order)  ; run callback functions
                        ;CCSinvokeCdfCallbacks(obj ?order params_order)  ; run callback functions
                    );when
                else  ; for vanilla cells
                    obj = dbCreateInst(cv inst_cv name loc orient)
                );if
            else  ; create an array instance
                if( params != nil then  ; for pcells
                    obj = dbCreateParamSimpleMosaic(cv inst_cv name loc orient num_rows num_cols sp_rows sp_cols params)
                    when( obj
                        if( params_order == nil then  ; if parameter order is not specified,
                            params_order = mapcar( lambda( (x) car(x) ) params )
                        );if
                        abInvokeCdfCallbacks(obj ?order params_order)  ; run callback functions
                        ;CCSinvokeCdfCallbacks(obj ?order params_order)  ; run callback functions
                    );when
                else  ; for vanilla cells
                    obj = dbCreateSimpleMosaic(cv inst_cv name loc orient num_rows num_cols sp_rows sp_cols)
                );if
            );if
            unless( obj
                warn("[LAYGO2] Error creating an instance: %s of %s__%s (%s). Skipping." name libname cellname viewname)
            );unless
        else
            warn("[LAYGO2] Cannot find the instance %s__%s (%s). Skipping." libname cellname viewname)
        );if
        obj
    );let
);procedure

; create a rectangle
procedure( _laygo2_generate_rect( cv layer bbox )
    obj = dbCreateRect(cv layer bbox)
    obj
);procedure

; TODO: implement path creation function

; create a pin
procedure( _laygo2_generate_pin( cv, name, layer, bbox)
    let( (obj pin_w pin_h pin_xc pin_yc)
        pin_w = caadr(bbox) - caar(bbox)
        pin_h = cadr(cadr(bbox)) - cadr(car(bbox))
        pin_xc = (caar(bbox) + caadr(bbox)) / 2.0
        pin_yc = (cadr(car(bbox)) + cadr(cadr(bbox))) / 2.0
        if( pin_w >= pin_h then
            obj=dbCreateLabel( cv layer list(pin_xc pin_yc) name "centerCenter" "R0" "roman" pin_h )
        else
            obj=dbCreateLabel( cv layer list(pin_xc pin_yc) name "centerCenter" "R90" "roman" pin_w )
        ); if
    obj
    );let
);procedure

; test routine
let( (_laygo2_test cv)
    _laygo2_test = nil
    if( _laygo_test then

        libname="test2"
        tech_libname="testtech"
        cellname="test"
        viewname="layout"

        name="MYINST0"
        inst_libname="test"
        inst_cellname="mytest"
        inst_viewname="layout"
        loc = list(10 20)
        orient="R0"
        num_rows=2
        num_cols=3
        sp_rows=2
        sp_cols=1
        params=nil
        params_order=nil

        rect_layer = list("M2" "drawing")
        rect_bbox = list(list(-5 -10) list(5 10))

        pin_name = "mypin"
        pin_layer = list("M2" "pin")
        pin_bbox = list(list(5 -10) list(15 10))

        _laygo2_reset_library(libname tech_libname)
        cv = _laygo2_open_layout(libname cellname viewname)
        _laygo2_generate_instance( cv name inst_libname inst_cellname inst_viewname loc orient num_rows num_cols sp_rows
                                sp_cols params params_order )
        _laygo2_generate_rect( cv rect_layer rect_bbox )
        _laygo2_generate_pin( cv, pin_name, pin_layer, pin_bbox)
        _laygo2_save_and_close_layout(cv)
    );if
);let

; exporting laygo2_test__nand2
let( (cv) 
cv = _laygo2_open_layout("laygo2_test" "nand2" "layout")
_laygo2_generate_rect(cv, list( "prBoundary" "drawing" ), list( list( 0.0000  0.0000  ) list( 0.0300  0.1000  ) ) ) ; for the Rect object prBoundary0 
_laygo2_generate_rect(cv, list( "prBoundary" "drawing" ), list( list( 0.0300  0.0000  ) list( 0.0600  0.1000  ) ) ) ; for the Rect object prBoundary1 
_laygo2_generate_rect(cv, list( "prBoundary" "drawing" ), list( list( 0.0600  0.0000  ) list( 0.0900  0.1000  ) ) ) ; for the Rect object prBoundary2 
_laygo2_generate_rect(cv, list( "prBoundary" "drawing" ), list( list( 0.0900  0.0000  ) list( 0.1200  0.1000  ) ) ) ; for the Rect object prBoundary3 
_laygo2_generate_rect(cv, list( "nimplant" "drawing" ), list( list( -0.0150  0.0200  ) list( 0.0450  0.0800  ) ) ) ; for the Rect object impl0 
_laygo2_generate_rect(cv, list( "nimplant" "drawing" ), list( list( 0.0150  0.0200  ) list( 0.0750  0.0800  ) ) ) ; for the Rect object impl1 
_laygo2_generate_rect(cv, list( "nimplant" "drawing" ), list( list( 0.0450  0.0200  ) list( 0.1050  0.0800  ) ) ) ; for the Rect object impl2 
_laygo2_generate_rect(cv, list( "nimplant" "drawing" ), list( list( 0.0750  0.0200  ) list( 0.1350  0.0800  ) ) ) ; for the Rect object impl3 
_laygo2_generate_rect(cv, list( "diffusion" "drawing" ), list( list( -0.0100  0.0250  ) list( 0.0400  0.0750  ) ) ) ; for the Rect object diff0 
_laygo2_generate_rect(cv, list( "diffusion" "drawing" ), list( list( 0.0200  0.0250  ) list( 0.0700  0.0750  ) ) ) ; for the Rect object diff1 
_laygo2_generate_rect(cv, list( "diffusion" "drawing" ), list( list( 0.0500  0.0250  ) list( 0.1000  0.0750  ) ) ) ; for the Rect object diff2 
_laygo2_generate_rect(cv, list( "diffusion" "drawing" ), list( list( 0.0800  0.0250  ) list( 0.1300  0.0750  ) ) ) ; for the Rect object diff3 
_laygo2_generate_rect(cv, list( "poly" "drawing" ), list( list( 0.0120  0.0050  ) list( 0.0180  0.0950  ) ) ) ; for the Rect object poly0 
_laygo2_generate_rect(cv, list( "poly" "drawing" ), list( list( 0.0420  0.0050  ) list( 0.0480  0.0950  ) ) ) ; for the Rect object poly1 
_laygo2_generate_rect(cv, list( "poly" "drawing" ), list( list( 0.0720  0.0050  ) list( 0.0780  0.0950  ) ) ) ; for the Rect object poly2 
_laygo2_generate_rect(cv, list( "poly" "drawing" ), list( list( 0.1020  0.0050  ) list( 0.1080  0.0950  ) ) ) ; for the Rect object poly3 
_laygo2_generate_rect(cv, list( "metal1" "drawing" ), list( list( -0.0070  0.0230  ) list( 0.0070  0.0770  ) ) ) ; for the Rect object metal1_s0 
_laygo2_generate_rect(cv, list( "metal1" "drawing" ), list( list( 0.0230  0.0230  ) list( 0.0370  0.0770  ) ) ) ; for the Rect object metal1_s1 
_laygo2_generate_rect(cv, list( "metal1" "drawing" ), list( list( 0.0530  0.0230  ) list( 0.0670  0.0770  ) ) ) ; for the Rect object metal1_s2 
_laygo2_generate_rect(cv, list( "metal1" "drawing" ), list( list( 0.0830  0.0230  ) list( 0.0970  0.0770  ) ) ) ; for the Rect object metal1_s3 
_laygo2_generate_rect(cv, list( "metal1" "drawing" ), list( list( 0.0230  0.0230  ) list( 0.0370  0.0770  ) ) ) ; for the Rect object metal1_d0 
_laygo2_generate_rect(cv, list( "metal1" "drawing" ), list( list( 0.0530  0.0230  ) list( 0.0670  0.0770  ) ) ) ; for the Rect object metal1_d1 
_laygo2_generate_rect(cv, list( "metal1" "drawing" ), list( list( 0.0830  0.0230  ) list( 0.0970  0.0770  ) ) ) ; for the Rect object metal1_d2 
_laygo2_generate_rect(cv, list( "metal1" "drawing" ), list( list( 0.1130  0.0230  ) list( 0.1270  0.0770  ) ) ) ; for the Rect object metal1_d3 
_laygo2_generate_rect(cv, list( "metal1" "drawing" ), list( list( 0.0120  0.0850  ) list( 0.0480  0.0950  ) ) ) ; for the Rect object metal1_g0 
_laygo2_generate_rect(cv, list( "metal1" "drawing" ), list( list( 0.0420  0.0850  ) list( 0.0780  0.0950  ) ) ) ; for the Rect object metal1_g1 
_laygo2_generate_rect(cv, list( "metal1" "drawing" ), list( list( 0.0720  0.0850  ) list( 0.1080  0.0950  ) ) ) ; for the Rect object metal1_g2 
_laygo2_generate_rect(cv, list( "prBoundary" "drawing" ), list( list( 0.1200  0.0000  ) list( 0.1500  0.1000  ) ) ) ; for the Rect object prBoundary0 
_laygo2_generate_rect(cv, list( "prBoundary" "drawing" ), list( list( 0.1500  0.0000  ) list( 0.1800  0.1000  ) ) ) ; for the Rect object prBoundary1 
_laygo2_generate_rect(cv, list( "prBoundary" "drawing" ), list( list( 0.1800  0.0000  ) list( 0.2100  0.1000  ) ) ) ; for the Rect object prBoundary2 
_laygo2_generate_rect(cv, list( "nimplant" "drawing" ), list( list( 0.1050  0.0200  ) list( 0.1650  0.0800  ) ) ) ; for the Rect object impl0 
_laygo2_generate_rect(cv, list( "nimplant" "drawing" ), list( list( 0.1350  0.0200  ) list( 0.1950  0.0800  ) ) ) ; for the Rect object impl1 
_laygo2_generate_rect(cv, list( "nimplant" "drawing" ), list( list( 0.1650  0.0200  ) list( 0.2250  0.0800  ) ) ) ; for the Rect object impl2 
_laygo2_generate_rect(cv, list( "diffusion" "drawing" ), list( list( 0.1100  0.0250  ) list( 0.1600  0.0750  ) ) ) ; for the Rect object diff0 
_laygo2_generate_rect(cv, list( "diffusion" "drawing" ), list( list( 0.1400  0.0250  ) list( 0.1900  0.0750  ) ) ) ; for the Rect object diff1 
_laygo2_generate_rect(cv, list( "diffusion" "drawing" ), list( list( 0.1700  0.0250  ) list( 0.2200  0.0750  ) ) ) ; for the Rect object diff2 
_laygo2_generate_rect(cv, list( "poly" "drawing" ), list( list( 0.1320  0.0050  ) list( 0.1380  0.0950  ) ) ) ; for the Rect object poly0 
_laygo2_generate_rect(cv, list( "poly" "drawing" ), list( list( 0.1620  0.0050  ) list( 0.1680  0.0950  ) ) ) ; for the Rect object poly1 
_laygo2_generate_rect(cv, list( "poly" "drawing" ), list( list( 0.1920  0.0050  ) list( 0.1980  0.0950  ) ) ) ; for the Rect object poly2 
_laygo2_generate_rect(cv, list( "metal1" "drawing" ), list( list( 0.1130  0.0230  ) list( 0.1270  0.0770  ) ) ) ; for the Rect object metal1_s0 
_laygo2_generate_rect(cv, list( "metal1" "drawing" ), list( list( 0.1430  0.0230  ) list( 0.1570  0.0770  ) ) ) ; for the Rect object metal1_s1 
_laygo2_generate_rect(cv, list( "metal1" "drawing" ), list( list( 0.1730  0.0230  ) list( 0.1870  0.0770  ) ) ) ; for the Rect object metal1_s2 
_laygo2_generate_rect(cv, list( "metal1" "drawing" ), list( list( 0.1430  0.0230  ) list( 0.1570  0.0770  ) ) ) ; for the Rect object metal1_d0 
_laygo2_generate_rect(cv, list( "metal1" "drawing" ), list( list( 0.1730  0.0230  ) list( 0.1870  0.0770  ) ) ) ; for the Rect object metal1_d1 
_laygo2_generate_rect(cv, list( "metal1" "drawing" ), list( list( 0.2030  0.0230  ) list( 0.2170  0.0770  ) ) ) ; for the Rect object metal1_d2 
_laygo2_generate_rect(cv, list( "metal1" "drawing" ), list( list( 0.1320  0.0850  ) list( 0.1680  0.0950  ) ) ) ; for the Rect object metal1_g0 
_laygo2_generate_rect(cv, list( "metal1" "drawing" ), list( list( 0.1620  0.0850  ) list( 0.1980  0.0950  ) ) ) ; for the Rect object metal1_g1 
_laygo2_generate_rect(cv, list( "prBoundary" "drawing" ), list( list( 0.0000  0.2000  ) list( 0.0300  0.1000  ) ) ) ; for the Rect object prBoundary0 
_laygo2_generate_rect(cv, list( "prBoundary" "drawing" ), list( list( 0.0300  0.2000  ) list( 0.0600  0.1000  ) ) ) ; for the Rect object prBoundary1 
_laygo2_generate_rect(cv, list( "prBoundary" "drawing" ), list( list( 0.0600  0.2000  ) list( 0.0900  0.1000  ) ) ) ; for the Rect object prBoundary2 
_laygo2_generate_rect(cv, list( "prBoundary" "drawing" ), list( list( 0.0900  0.2000  ) list( 0.1200  0.1000  ) ) ) ; for the Rect object prBoundary3 
_laygo2_generate_rect(cv, list( "nwell" "drawing" ), list( list( -0.0200  0.2000  ) list( 0.0500  0.1000  ) ) ) ; for the Rect object well0 
_laygo2_generate_rect(cv, list( "nwell" "drawing" ), list( list( 0.0100  0.2000  ) list( 0.0800  0.1000  ) ) ) ; for the Rect object well1 
_laygo2_generate_rect(cv, list( "nwell" "drawing" ), list( list( 0.0400  0.2000  ) list( 0.1100  0.1000  ) ) ) ; for the Rect object well2 
_laygo2_generate_rect(cv, list( "nwell" "drawing" ), list( list( 0.0700  0.2000  ) list( 0.1400  0.1000  ) ) ) ; for the Rect object well3 
_laygo2_generate_rect(cv, list( "pimplant" "drawing" ), list( list( -0.0150  0.1800  ) list( 0.0450  0.1200  ) ) ) ; for the Rect object impl0 
_laygo2_generate_rect(cv, list( "pimplant" "drawing" ), list( list( 0.0150  0.1800  ) list( 0.0750  0.1200  ) ) ) ; for the Rect object impl1 
_laygo2_generate_rect(cv, list( "pimplant" "drawing" ), list( list( 0.0450  0.1800  ) list( 0.1050  0.1200  ) ) ) ; for the Rect object impl2 
_laygo2_generate_rect(cv, list( "pimplant" "drawing" ), list( list( 0.0750  0.1800  ) list( 0.1350  0.1200  ) ) ) ; for the Rect object impl3 
_laygo2_generate_rect(cv, list( "diffusion" "drawing" ), list( list( -0.0100  0.1750  ) list( 0.0400  0.1250  ) ) ) ; for the Rect object diff0 
_laygo2_generate_rect(cv, list( "diffusion" "drawing" ), list( list( 0.0200  0.1750  ) list( 0.0700  0.1250  ) ) ) ; for the Rect object diff1 
_laygo2_generate_rect(cv, list( "diffusion" "drawing" ), list( list( 0.0500  0.1750  ) list( 0.1000  0.1250  ) ) ) ; for the Rect object diff2 
_laygo2_generate_rect(cv, list( "diffusion" "drawing" ), list( list( 0.0800  0.1750  ) list( 0.1300  0.1250  ) ) ) ; for the Rect object diff3 
_laygo2_generate_rect(cv, list( "poly" "drawing" ), list( list( 0.0120  0.1950  ) list( 0.0180  0.1050  ) ) ) ; for the Rect object poly0 
_laygo2_generate_rect(cv, list( "poly" "drawing" ), list( list( 0.0420  0.1950  ) list( 0.0480  0.1050  ) ) ) ; for the Rect object poly1 
_laygo2_generate_rect(cv, list( "poly" "drawing" ), list( list( 0.0720  0.1950  ) list( 0.0780  0.1050  ) ) ) ; for the Rect object poly2 
_laygo2_generate_rect(cv, list( "poly" "drawing" ), list( list( 0.1020  0.1950  ) list( 0.1080  0.1050  ) ) ) ; for the Rect object poly3 
_laygo2_generate_rect(cv, list( "metal2" "drawing" ), list( list( -0.0070  0.1770  ) list( 0.0070  0.1230  ) ) ) ; for the Rect object metal1_s0 
_laygo2_generate_rect(cv, list( "metal2" "drawing" ), list( list( 0.0230  0.1770  ) list( 0.0370  0.1230  ) ) ) ; for the Rect object metal1_s1 
_laygo2_generate_rect(cv, list( "metal2" "drawing" ), list( list( 0.0530  0.1770  ) list( 0.0670  0.1230  ) ) ) ; for the Rect object metal1_s2 
_laygo2_generate_rect(cv, list( "metal2" "drawing" ), list( list( 0.0830  0.1770  ) list( 0.0970  0.1230  ) ) ) ; for the Rect object metal1_s3 
_laygo2_generate_rect(cv, list( "metal1" "drawing" ), list( list( 0.0230  0.1770  ) list( 0.0370  0.1230  ) ) ) ; for the Rect object metal1_d0 
_laygo2_generate_rect(cv, list( "metal1" "drawing" ), list( list( 0.0530  0.1770  ) list( 0.0670  0.1230  ) ) ) ; for the Rect object metal1_d1 
_laygo2_generate_rect(cv, list( "metal1" "drawing" ), list( list( 0.0830  0.1770  ) list( 0.0970  0.1230  ) ) ) ; for the Rect object metal1_d2 
_laygo2_generate_rect(cv, list( "metal1" "drawing" ), list( list( 0.1130  0.1770  ) list( 0.1270  0.1230  ) ) ) ; for the Rect object metal1_d3 
_laygo2_generate_rect(cv, list( "metal1" "drawing" ), list( list( 0.0120  0.1150  ) list( 0.0480  0.1050  ) ) ) ; for the Rect object metal1_g0 
_laygo2_generate_rect(cv, list( "metal1" "drawing" ), list( list( 0.0420  0.1150  ) list( 0.0780  0.1050  ) ) ) ; for the Rect object metal1_g1 
_laygo2_generate_rect(cv, list( "metal1" "drawing" ), list( list( 0.0720  0.1150  ) list( 0.1080  0.1050  ) ) ) ; for the Rect object metal1_g2 
_laygo2_generate_rect(cv, list( "prBoundary" "drawing" ), list( list( 0.1200  0.2000  ) list( 0.1500  0.1000  ) ) ) ; for the Rect object prBoundary0 
_laygo2_generate_rect(cv, list( "prBoundary" "drawing" ), list( list( 0.1500  0.2000  ) list( 0.1800  0.1000  ) ) ) ; for the Rect object prBoundary1 
_laygo2_generate_rect(cv, list( "prBoundary" "drawing" ), list( list( 0.1800  0.2000  ) list( 0.2100  0.1000  ) ) ) ; for the Rect object prBoundary2 
_laygo2_generate_rect(cv, list( "nwell" "drawing" ), list( list( 0.1000  0.2000  ) list( 0.1700  0.1000  ) ) ) ; for the Rect object well0 
_laygo2_generate_rect(cv, list( "nwell" "drawing" ), list( list( 0.1300  0.2000  ) list( 0.2000  0.1000  ) ) ) ; for the Rect object well1 
_laygo2_generate_rect(cv, list( "nwell" "drawing" ), list( list( 0.1600  0.2000  ) list( 0.2300  0.1000  ) ) ) ; for the Rect object well2 
_laygo2_generate_rect(cv, list( "pimplant" "drawing" ), list( list( 0.1050  0.1800  ) list( 0.1650  0.1200  ) ) ) ; for the Rect object impl0 
_laygo2_generate_rect(cv, list( "pimplant" "drawing" ), list( list( 0.1350  0.1800  ) list( 0.1950  0.1200  ) ) ) ; for the Rect object impl1 
_laygo2_generate_rect(cv, list( "pimplant" "drawing" ), list( list( 0.1650  0.1800  ) list( 0.2250  0.1200  ) ) ) ; for the Rect object impl2 
_laygo2_generate_rect(cv, list( "diffusion" "drawing" ), list( list( 0.1100  0.1750  ) list( 0.1600  0.1250  ) ) ) ; for the Rect object diff0 
_laygo2_generate_rect(cv, list( "diffusion" "drawing" ), list( list( 0.1400  0.1750  ) list( 0.1900  0.1250  ) ) ) ; for the Rect object diff1 
_laygo2_generate_rect(cv, list( "diffusion" "drawing" ), list( list( 0.1700  0.1750  ) list( 0.2200  0.1250  ) ) ) ; for the Rect object diff2 
_laygo2_generate_rect(cv, list( "poly" "drawing" ), list( list( 0.1320  0.1950  ) list( 0.1380  0.1050  ) ) ) ; for the Rect object poly0 
_laygo2_generate_rect(cv, list( "poly" "drawing" ), list( list( 0.1620  0.1950  ) list( 0.1680  0.1050  ) ) ) ; for the Rect object poly1 
_laygo2_generate_rect(cv, list( "poly" "drawing" ), list( list( 0.1920  0.1950  ) list( 0.1980  0.1050  ) ) ) ; for the Rect object poly2 
_laygo2_generate_rect(cv, list( "metal2" "drawing" ), list( list( 0.1130  0.1770  ) list( 0.1270  0.1230  ) ) ) ; for the Rect object metal1_s0 
_laygo2_generate_rect(cv, list( "metal2" "drawing" ), list( list( 0.1430  0.1770  ) list( 0.1570  0.1230  ) ) ) ; for the Rect object metal1_s1 
_laygo2_generate_rect(cv, list( "metal2" "drawing" ), list( list( 0.1730  0.1770  ) list( 0.1870  0.1230  ) ) ) ; for the Rect object metal1_s2 
_laygo2_generate_rect(cv, list( "metal1" "drawing" ), list( list( 0.1430  0.1770  ) list( 0.1570  0.1230  ) ) ) ; for the Rect object metal1_d0 
_laygo2_generate_rect(cv, list( "metal1" "drawing" ), list( list( 0.1730  0.1770  ) list( 0.1870  0.1230  ) ) ) ; for the Rect object metal1_d1 
_laygo2_generate_rect(cv, list( "metal1" "drawing" ), list( list( 0.2030  0.1770  ) list( 0.2170  0.1230  ) ) ) ; for the Rect object metal1_d2 
_laygo2_generate_rect(cv, list( "metal1" "drawing" ), list( list( 0.1320  0.1150  ) list( 0.1680  0.1050  ) ) ) ; for the Rect object metal1_g0 
_laygo2_generate_rect(cv, list( "metal1" "drawing" ), list( list( 0.1620  0.1150  ) list( 0.1980  0.1050  ) ) ) ; for the Rect object metal1_g1 
_laygo2_generate_rect(cv, list( "metal2" "drawing" ), list( list( 0.1350  0.0850  ) list( 0.1950  0.0950  ) ) ) ; for the Rect object NoName_0 
_laygo2_generate_rect(cv, list( "via1" "drawing" ), list( list( 0.1450  0.0850  ) list( 0.1550  0.0950  ) ) ) ; for the Rect object contact 
_laygo2_generate_rect(cv, list( "via1" "drawing" ), list( list( 0.1750  0.0850  ) list( 0.1850  0.0950  ) ) ) ; for the Rect object contact 
_laygo2_generate_rect(cv, list( "metal2" "drawing" ), list( list( 0.1350  0.1050  ) list( 0.1950  0.1150  ) ) ) ; for the Rect object NoName_3 
_laygo2_generate_rect(cv, list( "via1" "drawing" ), list( list( 0.1450  0.1050  ) list( 0.1550  0.1150  ) ) ) ; for the Rect object contact 
_laygo2_generate_rect(cv, list( "via1" "drawing" ), list( list( 0.1750  0.1050  ) list( 0.1850  0.1150  ) ) ) ; for the Rect object contact 
_laygo2_generate_rect(cv, list( "via2" "drawing" ), list( list( 0.1450  0.0850  ) list( 0.1550  0.0950  ) ) ) ; for the Rect object contact 
_laygo2_generate_rect(cv, list( "metal3" "drawing" ), list( list( 0.1450  0.0750  ) list( 0.1550  0.1250  ) ) ) ; for the Rect object NoName_7 
_laygo2_generate_rect(cv, list( "via2" "drawing" ), list( list( 0.1450  0.1050  ) list( 0.1550  0.1150  ) ) ) ; for the Rect object contact 
_laygo2_generate_rect(cv, list( "metal2" "drawing" ), list( list( 0.0150  0.0850  ) list( 0.1050  0.0950  ) ) ) ; for the Rect object NoName_9 
_laygo2_generate_rect(cv, list( "via1" "drawing" ), list( list( 0.0250  0.0850  ) list( 0.0350  0.0950  ) ) ) ; for the Rect object contact 
_laygo2_generate_rect(cv, list( "via1" "drawing" ), list( list( 0.0550  0.0850  ) list( 0.0650  0.0950  ) ) ) ; for the Rect object contact 
_laygo2_generate_rect(cv, list( "via1" "drawing" ), list( list( 0.0850  0.0850  ) list( 0.0950  0.0950  ) ) ) ; for the Rect object contact 
_laygo2_generate_rect(cv, list( "metal2" "drawing" ), list( list( 0.0150  0.1050  ) list( 0.1050  0.1150  ) ) ) ; for the Rect object NoName_13 
_laygo2_generate_rect(cv, list( "via1" "drawing" ), list( list( 0.0250  0.1050  ) list( 0.0350  0.1150  ) ) ) ; for the Rect object contact 
_laygo2_generate_rect(cv, list( "via1" "drawing" ), list( list( 0.0550  0.1050  ) list( 0.0650  0.1150  ) ) ) ; for the Rect object contact 
_laygo2_generate_rect(cv, list( "via1" "drawing" ), list( list( 0.0850  0.1050  ) list( 0.0950  0.1150  ) ) ) ; for the Rect object contact 
_laygo2_generate_rect(cv, list( "via2" "drawing" ), list( list( 0.0250  0.0850  ) list( 0.0350  0.0950  ) ) ) ; for the Rect object contact 
_laygo2_generate_rect(cv, list( "metal3" "drawing" ), list( list( 0.0250  0.0750  ) list( 0.0350  0.1250  ) ) ) ; for the Rect object NoName_18 
_laygo2_generate_rect(cv, list( "via2" "drawing" ), list( list( 0.0250  0.1050  ) list( 0.0350  0.1150  ) ) ) ; for the Rect object contact 
_laygo2_generate_rect(cv, list( "metal2" "drawing" ), list( list( 0.0150  0.0350  ) list( 0.2250  0.0450  ) ) ) ; for the Rect object NoName_20 
_laygo2_generate_rect(cv, list( "via1" "drawing" ), list( list( 0.0250  0.0350  ) list( 0.0350  0.0450  ) ) ) ; for the Rect object contact 
_laygo2_generate_rect(cv, list( "via1" "drawing" ), list( list( 0.0850  0.0350  ) list( 0.0950  0.0450  ) ) ) ; for the Rect object contact 
_laygo2_generate_rect(cv, list( "via1" "drawing" ), list( list( 0.1450  0.0350  ) list( 0.1550  0.0450  ) ) ) ; for the Rect object contact 
_laygo2_generate_rect(cv, list( "via1" "drawing" ), list( list( 0.2050  0.0350  ) list( 0.2150  0.0450  ) ) ) ; for the Rect object contact 
_laygo2_generate_rect(cv, list( "metal2" "drawing" ), list( list( 0.1050  0.0550  ) list( 0.1950  0.0650  ) ) ) ; for the Rect object NoName_25 
_laygo2_generate_rect(cv, list( "via1" "drawing" ), list( list( 0.1150  0.0550  ) list( 0.1250  0.0650  ) ) ) ; for the Rect object contact 
_laygo2_generate_rect(cv, list( "via1" "drawing" ), list( list( 0.1750  0.0550  ) list( 0.1850  0.0650  ) ) ) ; for the Rect object contact 
_laygo2_generate_rect(cv, list( "metal2" "drawing" ), list( list( 0.0150  0.1350  ) list( 0.2250  0.1450  ) ) ) ; for the Rect object NoName_28 
_laygo2_generate_rect(cv, list( "via1" "drawing" ), list( list( 0.0250  0.1350  ) list( 0.0350  0.1450  ) ) ) ; for the Rect object contact 
_laygo2_generate_rect(cv, list( "via1" "drawing" ), list( list( 0.0850  0.1350  ) list( 0.0950  0.1450  ) ) ) ; for the Rect object contact 
_laygo2_generate_rect(cv, list( "via1" "drawing" ), list( list( 0.1450  0.1350  ) list( 0.1550  0.1450  ) ) ) ; for the Rect object contact 
_laygo2_generate_rect(cv, list( "via1" "drawing" ), list( list( 0.2050  0.1350  ) list( 0.2150  0.1450  ) ) ) ; for the Rect object contact 
_laygo2_generate_rect(cv, list( "via2" "drawing" ), list( list( 0.1750  0.0550  ) list( 0.1850  0.0650  ) ) ) ; for the Rect object contact 
_laygo2_generate_rect(cv, list( "metal3" "drawing" ), list( list( 0.1750  0.0450  ) list( 0.1850  0.1550  ) ) ) ; for the Rect object NoName_34 
_laygo2_generate_rect(cv, list( "via2" "drawing" ), list( list( 0.1750  0.1350  ) list( 0.1850  0.1450  ) ) ) ; for the Rect object contact 
_laygo2_generate_rect(cv, list( "metal2" "drawing" ), list( list( -0.0150  0.0050  ) list( 0.1650  0.0150  ) ) ) ; for the Rect object NoName_36 
_laygo2_generate_rect(cv, list( "via1" "drawing" ), list( list( -0.0050  0.0050  ) list( 0.0050  0.0150  ) ) ) ; for the Rect object contact 
_laygo2_generate_rect(cv, list( "metal1" "drawing" ), list( list( -0.0070  0.0030  ) list( 0.0070  0.0230  ) ) ) ; for the Rect object metal1 
_laygo2_generate_rect(cv, list( "via1" "drawing" ), list( list( 0.0550  0.0050  ) list( 0.0650  0.0150  ) ) ) ; for the Rect object contact 
_laygo2_generate_rect(cv, list( "metal1" "drawing" ), list( list( 0.0530  0.0030  ) list( 0.0670  0.0230  ) ) ) ; for the Rect object metal1 
_laygo2_generate_rect(cv, list( "via1" "drawing" ), list( list( 0.1150  0.0050  ) list( 0.1250  0.0150  ) ) ) ; for the Rect object contact 
_laygo2_generate_rect(cv, list( "metal1" "drawing" ), list( list( 0.1130  0.0030  ) list( 0.1270  0.0230  ) ) ) ; for the Rect object metal1 
_laygo2_generate_rect(cv, list( "metal2" "drawing" ), list( list( -0.0150  0.1850  ) list( 0.1950  0.1950  ) ) ) ; for the Rect object NoName_40 
_laygo2_generate_rect(cv, list( "via1" "drawing" ), list( list( -0.0050  0.1850  ) list( 0.0050  0.1950  ) ) ) ; for the Rect object contact 
_laygo2_generate_rect(cv, list( "metal1" "drawing" ), list( list( -0.0070  0.1770  ) list( 0.0070  0.1970  ) ) ) ; for the Rect object metal1 
_laygo2_generate_rect(cv, list( "via1" "drawing" ), list( list( 0.0550  0.1850  ) list( 0.0650  0.1950  ) ) ) ; for the Rect object contact 
_laygo2_generate_rect(cv, list( "metal1" "drawing" ), list( list( 0.0530  0.1770  ) list( 0.0670  0.1970  ) ) ) ; for the Rect object metal1 
_laygo2_generate_rect(cv, list( "via1" "drawing" ), list( list( 0.1150  0.1850  ) list( 0.1250  0.1950  ) ) ) ; for the Rect object contact 
_laygo2_generate_rect(cv, list( "metal1" "drawing" ), list( list( 0.1130  0.1770  ) list( 0.1270  0.1970  ) ) ) ; for the Rect object metal1 
_laygo2_generate_rect(cv, list( "via1" "drawing" ), list( list( 0.1150  0.1850  ) list( 0.1250  0.1950  ) ) ) ; for the Rect object contact 
_laygo2_generate_rect(cv, list( "metal1" "drawing" ), list( list( 0.1130  0.1770  ) list( 0.1270  0.1970  ) ) ) ; for the Rect object metal1 
_laygo2_generate_rect(cv, list( "via1" "drawing" ), list( list( 0.1750  0.1850  ) list( 0.1850  0.1950  ) ) ) ; for the Rect object contact 
_laygo2_generate_rect(cv, list( "metal1" "drawing" ), list( list( 0.1730  0.1770  ) list( 0.1870  0.1970  ) ) ) ; for the Rect object metal1 
_laygo2_generate_pin(cv, "A", list( "metal3" "pin" ), list( list( 0.1450  0.0900  ) list( 0.1550  0.1100  ) ) ) ; for the Pin object A 
_laygo2_generate_pin(cv, "B", list( "metal3" "pin" ), list( list( 0.0250  0.0900  ) list( 0.0350  0.1100  ) ) ) ; for the Pin object B 
_laygo2_generate_pin(cv, "O", list( "metal3" "pin" ), list( list( 0.1750  0.0600  ) list( 0.1850  0.1400  ) ) ) ; for the Pin object O 
_laygo2_generate_pin(cv, "VSS", list( "metal2" "pin" ), list( list( 0.0000  0.0050  ) list( 0.1500  0.0150  ) ) ) ; for the Pin object VSS 
_laygo2_generate_pin(cv, "VDD", list( "metal2" "pin" ), list( list( 0.0000  0.1850  ) list( 0.1800  0.1950  ) ) ) ; for the Pin object VDD 
_laygo2_save_and_close_layout(cv)
);let
