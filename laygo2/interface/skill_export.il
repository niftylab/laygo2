; laygo2 skill export script.
; - Initial code from BAG_framework (https://github.com/ucb-art/BAG_framework).
; - Users need to install CCSinvokeCdfCallbacks() from Cadence Online Support
;   (support.cadence.com) to enable pcell export.
;   (link: https://support.cadence.com/apex/ArticleAttachmentPortal?id=a1Od0000000namyEAA&pageName=ArticleContent)

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
                        ;abInvokeCdfCallbacks(obj ?order params_order)  ; run callback functions
                        CCSinvokeCdfCallbacks(obj ?order params_order)  ; run callback functions
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
                        ;abInvokeCdfCallbacks(obj ?order params_order)  ; run callback functions
                        CCSinvokeCdfCallbacks(obj ?order params_order)  ; run callback functions
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
procedure( _laygo2_generate_rect( cv layer bbox color )
    ;obj = dbCreateRect(cv layer bbox)
    obj = dbCreateRect(cv layer bbox)
;    obj = dbSetShapeColor(dbCreateRect(cv layer bbox) color) ; coloring func. added
    if( color != "not MPT" then
        dbSetShapeColor( obj color )
        dbSetShapeColorLocked(obj t)
    else )
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
    if( _laygo2_test then

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
