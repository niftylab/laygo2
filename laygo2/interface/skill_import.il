; laygo2 skill import script.
; - Initial code from BAG_framework (https://github.com/ucb-art/BAG_framework).
; - Users need to install CCSinvokeCdfCallbacks() from Cadence Online Support
;   (support.cadence.com) to enable pcell export.
;   (link: https://support.cadence.com/apex/ArticleAttachmentPortal?id=a1Od0000000namyEAA&pageName=ArticleContent)

; Parse the structure of the given cellview.
; Works on layout.
procedure( _laygo2_parse_cad_layout(lib_name cell_name file_name mpt "tttt")
    let( (cv cell_type p indent rect_cnt label_cnt inst_cnt)

        indent = ""
        cell_type = "layout"
        unless( cv = dbOpenCellViewByType( lib_name cell_name cell_type nil "r" )
            error( "Cannot find layout view of cell %s__%s" lib_name cell_name )
        )
        p = outfile( file_name "w" )

        ; print cellview information
        printf( "*INFO* Writing cell %s__%s (%s) netlist to %s\n" lib_name cell_name cell_type file_name )
        fprintf( p "%slib_name: %s\n" indent lib_name )
        fprintf( p "%scell_name: %s\n" indent cell_name )

        ; print rects
        if( not(cv~>shapes) then
            fprintf( p "%srects: {}\n" indent )
        else
            rect_cnt = 0
            fprintf( p "%srects:\n" indent )
            foreach( shape cv~>shapes
                if( (shape~>objType == "rect") then
                    rect_cnt++ 
                    ; print entry for rect
                    indent = "  "
                    fprintf( p "%s%d:\n" indent rect_cnt )
                    ; print rect master information.
                    indent = "    "
                    fprintf( p "%slayer: %s %s\n" indent nthelem(1 shape~>lpp) nthelem(2 shape~>lpp))
                    fprintf( p "%sbBox: [[%f, %f], [%f, %f]]\n" indent 
                                 nthelem(1 nthelem(1 shape~>bBox)) nthelem(2 nthelem(1 shape~>bBox)) 
                                 nthelem(1 nthelem(2 shape~>bBox)) nthelem(2 nthelem(2 shape~>bBox))
                           );fprintf
                    if( mpt == "t" then
                        fprintf( p "%scolor: %s\n" indent dbGetShapeColor(shape)) ; get the color of the shape
                    else )
                )
            );if
            if((rect_cnt == 0) then
                fprintf( p "  {}\n" )
           );if
        )

        ; print labels
        indent = ""
        if( not(cv~>shapes) then
            fprintf( p "%slabels: {}\n" indent )
        else
            label_cnt = 0
            fprintf( p "%slabels:\n" indent )
            foreach( shape cv~>shapes
                if( (shape~>objType == "label") then
                    label_cnt++ 
                    ; print entry for label
                    indent = "  "
                    fprintf( p "%s%d:\n" indent label_cnt )
                    ; print label master information.
                    indent = "    "
                    fprintf( p "%slabel: %s\n" indent shape~>theLabel )
                    fprintf( p "%slayer: %s %s\n" indent nthelem(1 shape~>lpp) nthelem(2 shape~>lpp))
                    fprintf( p "%sxy: [%f, %f]\n" indent nthelem(1 shape~>xy) nthelem(2 shape~>xy))
                )
                if( (shape~>objType == "textDisplay") then ;some labels are instantiated as text displays
                    label_cnt++
                    ; print entry for label
                    indent = "  "
                    fprintf( p "%s%d:\n" indent label_cnt )
                    ; print label master information.
                    indent = "    "
                    fprintf( p "%slabel: %s\n" indent shape~>owner~>name )
                    fprintf( p "%slayer: %s %s\n" indent nthelem(1 shape~>lpp) nthelem(2 shape~>lpp))
                    fprintf( p "%sxy: [%f, %f]\n" indent nthelem(1 shape~>xy) nthelem(2 shape~>xy))
                )
            );if
            if((label_cnt == 0) then
                fprintf( p "  {}\n" )
           );if
        )
        
        ; print instances
        indent = ""
        if( not(cv~>instances) then
            fprintf( p "%sinstances: {}\n" indent )
        else
            inst_cnt = 0
            fprintf( p "%sinstances:\n" indent )
            foreach( inst cv~>instances
                inst_cnt++
                ; print entry for instance
                indent = "  "
                fprintf( p "%s%s:\n" indent inst~>name )
                ; print instance master information.
                indent = "    "
                fprintf( p "%slib_name: %s\n" indent inst~>libName )
                fprintf( p "%scell_name: %s\n" indent inst~>cellName )
                fprintf( p "%sxy: [%f, %f]\n" indent nthelem(1 inst~>xy) nthelem(2 inst~>xy))
                if( (inst~>objType == "mosaic") then
                    fprintf( p "%scols: %d\n" indent inst~>columns)
                    fprintf( p "%srows: %d\n" indent inst~>rows)
                    fprintf( p "%ssp_cols: %f\n" indent inst~>uX)
                    fprintf( p "%ssp_rows: %f\n" indent inst~>uY)
                    fprintf( p "%srotation: %s\n" indent car(inst~>tileArray))
                    else
                    fprintf( p "%srotation: %s\n" indent inst~>orient)
                );if
            )
            when(inst_cnt == 0
                fprintf( p "  {}\n" )
            )
        )
        
        ; close resources
        close(p)
        dbClose(cv)
    )
)

; get a list of cells containing in the specficied library
procedure( _laygo2_get_cell_list(lib_name file_name "tt")
    let( (lib cellname p)
        lib=ddGetObj(lib_name)
        p = outfile( file_name "w" )
        fprintf( p "%s: [" lib_name)
        foreach( cellname lib~>cells~>name
            fprintf( p "%s, " cellname)
        );foreach
        fprintf( p "] \n" )
        ; close resources
        close(p)
    );let
)

; test routine
let( (_laygo2_test cv)
    ;_laygo2_test = t
    _laygo2_test = nil
    if( _laygo2_test then

        libname="logic_generated"
        cellname="inv_2x"
        viewname="layout"

        filename1 = "skill_import1.yaml"
        filename2 = "skill_import2.yaml"

        mpt = "nil"
        ;mpt = "t"

	_laygo2_get_cell_list(libname filename1)
        _laygo2_parse_cad_layout(libname cellname filename2 mpt)
    );if
);let
