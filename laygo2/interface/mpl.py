#!/usr/bin/python
########################################################################################################################
#
# Copyright (C) 2023, Nifty Chips Laboratory at Hanyang University - All Rights Reserved
# 
# Unauthorized copying of this software package, via any medium is strictly prohibited
# Proprietary and confidential
# Written by Jaeduk Han, 07-23-2023
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

"""
This module implements the interface with matplotlib.

"""
from typing import (
    TYPE_CHECKING,
    Union,
    List
)

import logging
from math import log10
from decimal import *

import numpy as np
import laygo2.util.transform as tf

import matplotlib
import matplotlib.pyplot as plt

import laygo2.object

#from laygo2._typing import Path

# Type checking
from typing import TYPE_CHECKING, overload, Generic, Dict
from typing import List, Tuple, Iterable, Type, Union, Any, Optional
if TYPE_CHECKING:
    import laygo2

__author__ = ""
__maintainer__ = ""
__status__ = "Prototype"

# support for hover information.
# https://stackoverflow.com/a/47166787/7128154
# https://matplotlib.org/3.3.3/api/collections_api.html#matplotlib.collections.PathCollection
# https://matplotlib.org/3.3.3/api/path_api.html#matplotlib.path.Path
# https://stackoverflow.com/questions/15876011/add-information-to-matplotlib-navigation-toolbar-status-bar
# https://stackoverflow.com/questions/36730261/matplotlib-path-contains-point
# https://stackoverflow.com/a/36335048/7128154
class StatusbarHoverManager:
    """
    Manage hover information for mpl.axes.Axes object based on appearing
    artists.

    Attributes
    ----------
    ax : mpl.axes.Axes
        subplot to show status information
    artists : list of mpl.artist.Artist
        elements on the subplot, which react to mouse over
    labels : list (list of strings) or strings
        each element on the top level corresponds to an artist.
        if the artist has items
        (i.e. second return value of contains() has key 'ind'),
        the element has to be of type list.
        otherwise the element if of type string
    cid : to reconnect motion_notify_event
    """
    def __init__(self, fig, ax):
        assert isinstance(ax, matplotlib.axes.Axes)

        def hover(event):
            if event.inaxes != ax:
                return
            info = 'x={:.0f}, y={:.0f}'.format(event.xdata, event.ydata)
            self.text.set_text(info)
            # replace with self.text.set_text(info) if you want to show the info in the figure information bar.
            #ax.format_coord = lambda x, y: info
        cid = ax.figure.canvas.mpl_connect("motion_notify_event", hover)
        self.ax = ax
        self.cid = cid
        self.artists = []
        self.labels = []
        text = fig.text(0.0, 0.02, "", va="bottom", ha="left", fontsize=6)
        self.fig = fig
        self.text = text

    def add_artist_labels(self, artist, label):
        if isinstance(artist, list):
            assert len(artist) == 1
            artist = artist[0]

        self.artists += [artist]
        self.labels += [label]

        def hover(event):
            if event.inaxes != self.ax:
                return
            info = 'x={:.0f}, y={:.0f}'.format(event.xdata, event.ydata)
            for aa, artist in enumerate(self.artists):
                cont, dct = artist.contains(event)
                if not cont:
                    continue
                inds = dct.get('ind')
                if inds is not None:  # artist contains items
                    for ii in inds:
                        lbl = self.labels[aa][ii]
                        info += '\n   {:}'.format(lbl)
                else:
                    lbl = self.labels[aa]
                    info += '\n   {:}'.format(lbl)
            self.text.set_text(info)
            self.fig.canvas.draw_idle()
            # replace with self.text.set_text(info) if you want to show the info in the figure information bar.
            #self.ax.format_coord = lambda x, y: info

        self.ax.figure.canvas.mpl_disconnect(self.cid)
        self.cid = self.ax.figure.canvas.mpl_connect("motion_notify_event", hover)


def _translate_obj(
        objname: str, 
        obj: "laygo2.object.physical.PhysicalObject", 
        colormap: dict, 
        scale=1, 
        master: "laygo2.object.physical.PhysicalObject" = None, 
        offset=np.array([0, 0])
        ):
    """
    Convert a layout object to corresponding matplotlib patch object.

    Parameters
    ----------
    objname: str
        The name of the object. 
    obj: laygo2.object.PhysicalObject
        The object to be converted.
    colormap: dict
        A dictionary that contains layer-color mapping information.
    scale: float
        (optional) The scaling factor between laygo2's integer coordinates and plot coordinates.
    master: laygo2.object.PhysicalObject
        (optional) The master object of the translated object.
    offset: np.array
        (optional) The offset of the translated object's position.
    
    Returns
    -------
    pypobjs: list
        A list of matplotlib patch objects. Each element is a list of the form [patch, layer, annotation, (on-figure annotation)].
    """
    if master is None:
        mxy = np.array([0, 0])
        mtf = "R0"
    else:  # if the translated object has a master (e.g. VirtualInstance)
        mxy = master.xy
        mtf = master.transform
    if obj.__class__ == laygo2.object.physical.Rect:
        # Compute the rectangle size
        _xy = np.sort(obj.xy, axis=0)  # make sure obj.xy is sorted
        _xy = mxy + np.dot(
            _xy + np.array([[-obj.hextension, -obj.vextension], [obj.hextension, obj.vextension]]),
            tf.Mt(mtf).T,
        )
        size = [_xy[1, 0] - _xy[0, 0], _xy[1, 1] - _xy[0, 1]]
        # Create a rectangle patch
        if obj.layer[0] in colormap:
            rect = matplotlib.patches.Rectangle(
                (_xy[0, 0], _xy[0, 1]),
                size[0],
                size[1],
                facecolor=colormap[obj.layer[0]][1],
                edgecolor=colormap[obj.layer[0]][0],
                alpha=colormap[obj.layer[0]][2],
                lw=2,
            )
            return [[rect, obj.layer[0], "[rect] " + objname + "/" + obj.layer[0] + "/" +str(obj.netname)]]
        return []
    elif obj.__class__ == laygo2.object.physical.Path:
        # TODO: implement path export function.
        pass
    elif obj.__class__ == laygo2.object.physical.Pin:
        if obj.elements is None:
            _objelem = [obj]
        else:
            _objelem = obj.elements
        for idx, _obj in np.ndenumerate(_objelem):
            # Compute the pin rectangle size
            _xy = mxy + np.dot(_obj.xy, tf.Mt(mtf).T)
            size = [_xy[1, 0] - _xy[0, 0], _xy[1, 1] - _xy[0, 1]]
            if obj.layer[0] in colormap:
                # Create a pin rectangle patch
                rect = matplotlib.patches.Rectangle(
                    (_xy[0, 0], _xy[0, 1]),
                    size[0],
                    size[1],
                    facecolor=colormap[obj.layer[0]][1],
                    edgecolor=colormap[obj.layer[0]][0],
                    alpha=colormap[obj.layer[0]][2],
                    lw=2,
                )
                pypobj = [rect, obj.layer[0], "[pin] " + objname + "/" + obj.layer[0] + "/" + str(obj.netname)]
                if master is None:  # add on-figure annotatation if the pin is at the top level
                    pypobj.append(obj.layer[0]+"/"+str(obj.netname))
                return [pypobj]
            return []
    elif obj.__class__ == laygo2.object.physical.Text:
        return [["text", obj.layer[0], obj.text, obj.xy]]
        # TODO: implement text export function.
        pass
    elif obj.__class__ == laygo2.object.physical.Instance:
        # Check if the instance belongs to a virtual instance or array.
        if master is not None:
            _master = master
        elif obj.master is not None:
            _master = obj.master
        else:
            _master = None
        if obj.shape is None:  # single instance
            _xy = mxy + np.dot(obj.xy, tf.Mt(mtf).T)
            _xy0 = _xy
            _xy1 = np.dot(obj.xy1-obj.xy0, tf.Mt(mtf).T) #* np.array([num_rows, num_cols])
            rect = matplotlib.patches.Rectangle(
                (_xy0[0], _xy0[1]),
                _xy1[0],
                _xy1[1],
                facecolor=colormap["__instance__"][1],
                edgecolor=colormap["__instance__"][0],
                alpha=colormap["__instance__"][2],
                lw=2,
            )
            pypobj = [rect, "__instance__", "[inst] " + objname + "/" + obj.cellname]
            if master is None:  # add on-figure annotatation if the instance is at the top level
                if not obj.cellname.startswith("via"):
                    pypobj.append(objname+"/"+obj.cellname)
            pypobjs = [pypobj]
        else: # array instance
            pypobjs = []
            for i, e in np.ndenumerate(obj.elements):
                pypobjs += _translate_obj(e.name+"_"+str(i), e, colormap, scale=scale, master=_master)

        # Instance pins
        for pn, p in obj.pins.items():
            _pypobj = _translate_obj(pn, p, colormap, scale=scale, master=_master, offset=offset)
            _pypobj[0][1] = "__instance_pin__"
            _pypobj[0][0].set(edgecolor=colormap["__instance_pin__"][0])
            _pypobj[0][0].set(facecolor=colormap["__instance_pin__"][1])
            _pypobj[0][2] = "[inst] " + objname + " " + _pypobj[0][2]
            pypobjs += _pypobj
        return pypobjs
    elif obj.__class__ == laygo2.object.physical.VirtualInstance:
        # construct the list that constains patches of elements.
        pypobjs = []
        if obj.shape is None:
            _xy = mxy + np.dot(obj.xy, tf.Mt(mtf).T)
            _xy0 = _xy
            _xy1 = np.dot(obj.xy1-obj.xy0, tf.Mt(mtf).T) 
            rect = matplotlib.patches.Rectangle(
                (_xy0[0], _xy0[1]),
                _xy1[0],
                _xy1[1],
                facecolor=colormap["__instance__"][1],
                edgecolor=colormap["__instance__"][0],
                alpha=colormap["__instance__"][2],
                lw=2,
            )
            pypobjs = [[rect, "__instance__", "[vinst] " + objname + "/" + obj.cellname]]
            pypobjs[0].append(objname+"/"+obj.cellname)
            for elem_name, elem in obj.native_elements.items():
                if not elem.__class__ == laygo2.object.physical.Pin:
                    if obj.name == None:
                        objname = "NoName"
                    else:
                        objname = obj.name
                    _pypobj = _translate_obj(objname, elem, colormap, scale=scale, master=obj)
                    for _p in _pypobj:
                        _p[2] = "[vinst] " + objname + " " + _p[2]
                    pypobjs += _pypobj
        else:  # arrayed VirtualInstance
            for i, j in np.ndindex(tuple(obj.shape.tolist())):  # iterate over obj.shape
                for elem_name, elem in obj.native_elements.items():
                    if not elem.__class__ == laygo2.object.physical.Pin:
                        _pypobj = _translate_obj(
                            obj.name + "_" + elem_name + str(i) + "_" + str(j),
                            elem,
                            colormap,
                            scale=scale,
                            master=obj[i, j],
                        )
                        _pypobj[0][2] = "[vinst] " + objname + " " + _pypobj[0][2]
                        pypobjs += _pypobj

        # Instance pins
        for pn, p in obj.pins.items():
            # master coordinate is already included in the pin object of the virtual instance.
            _pypobj = _translate_obj(pn, p, colormap, scale=scale, master=None, offset=offset)  
            _pypobj[0][1] = "__instance_pin__"
            _pypobj[0][0].set(edgecolor=colormap["__instance_pin__"][0])
            _pypobj[0][0].set(facecolor=colormap["__instance_pin__"][1])
            _pypobj[0][2] = "[vinst] " + objname + " " + _pypobj[0][2]
            pypobjs += _pypobj
        return pypobjs

    else:
        return []
        # return [obj.translate_to_matplotlib()]
    return []

def export(
    db: Union["laygo2.object.database.Library", "laygo2.object.database.Design"],
    cellname: str = None,
    tech: "laygo2.object.technology.BaseTechnology" = None,
    scale = 1,
    colormap: dict = None,
    order = None,
    xlim: list = None,
    ylim: list = None,
    show: bool = False,
    filename: str = None,
    annotate_grid: List["laygo2.object.physical.Grid"] = None,
):
    """
    Export a laygo2.object.database.Library or Design object to a matplotlib plot.

    Parameters
    ----------
    db: laygo2.database.Library or laygo2.database design
        The library database or design to exported.
    cellname: str or List[str]
        (optional) The name(s) of cell(s) to be exported.
    tech: laygo2.technology.BaseTechnology
        (optional) The technology object to be used for the export.
    scale: float
        (optional) The scaling factor between laygo2's integer coordinates and plot coordinates.
    colormap: dict
        A dictionary that contains layer-color mapping information.
    order: list
        A list that contains the order of layers to be displayed (former is plotted first).
    xlim: list
        (optional) A list that specifies the range of plot in x-axis.
    ylim: list
        (optional) A list that specifies the range of plot in y-axis.
    filename: str
        (optional) If specified, export a output file for the plot.
    annotate_grid: list
        (optional) A list of grid objects to be annotated.

    Returns
    -------
    matplotlib.pyplot.figure or list: The generated figure object(s).

    """
    # colormap
    if colormap is None:
        if tech is None:
            colormap = dict()
        else:
            tech_params = tech.tech_params
            colormap = tech_params['export']['mpl']['colormap']

    # a list to align layered objects in order
    if order is None:
        if tech is None:
            order = []
        else:
            tech_params = tech.tech_params
            order = tech_params['export']['mpl']['order']

    # cell name handling.
    if isinstance(db, laygo2.object.database.Design):
        cellname = [db.cellname]
        db = {db.cellname:db}
    if isinstance(db, laygo2.object.database.Library):
        cellname = db.keys() if cellname is None else cellname  # export all cells if cellname is not given.
        cellname = [cellname] if isinstance(cellname, str) else cellname  # convert to a list for iteration.

    fig = []
    _fig = plt.figure()
    for cn in cellname:
        if not show:
            _fig.set_figwidth(max(6, int(6*db[cn].bbox[1, 0]/1200)))
            _fig.set_figheight(max(6, int(6*db[cn].bbox[1, 1]/1200)))
        pypobjs = []
        ax = _fig.add_subplot(111)
        shm = StatusbarHoverManager(_fig, ax)
        for objname, obj in db[cn].items():
            pypobjs += _translate_obj(objname, obj, colormap, scale=scale)
        for _alignobj in order:
            for _pypobj in pypobjs:
                if _pypobj[1] == _alignobj:  # layer is matched.
                    if isinstance(_pypobj[0], str):
                        if _pypobj[0] == 'text':  # Text
                            color = "black"
                            ax.annotate(
                                _pypobj[2], (_pypobj[3][0], _pypobj[3][1]), color=color, weight="bold", fontsize=6, ha="center", va="center"
                            )
                    elif _pypobj[0].__class__ == matplotlib.patches.Rectangle:  # Rect
                        ax.add_patch(_pypobj[0])
                        if len(_pypobj) >= 3:  # annotation.
                            shm.add_artist_labels(_pypobj[0], _pypobj[2])
                            #ax.add_artist(_pypobj[0])
                            if len(_pypobj) == 4:  # text annotation.
                                rx, ry = _pypobj[0].get_xy()
                                cx = rx + _pypobj[0].get_width() / 2.0
                                cy = ry + _pypobj[0].get_height() / 2.0
                                if _pypobj[1] == "__instance__":
                                    color = _pypobj[0].get_edgecolor()
                                elif _pypobj[1] == "__instance_pin__":
                                    color = _pypobj[0].get_edgecolor()
                                else:
                                    color = "black"
                                ax.annotate(
                                    _pypobj[3], (cx, cy), color=color, weight="bold", fontsize=6, ha="center", va="center"
                                )
        if annotate_grid is not None:
            for grid in annotate_grid:
                laygo2.interface.mpl.export_grid(
                    obj=grid,
                    colormap=colormap,
                    order=order,
                    xlim=db[cn].bbox[:, 0],
                    ylim=db[cn].bbox[:, 1],
                    filename=None,
                    fig=_fig,
                    ax=ax,
                    for_annotation=True,
                )
        # scale
        ax.set_aspect('equal', adjustable='box')
        #_fig.tight_layout()
        plt.autoscale()
        if not (xlim == None):
            plt.xlim(xlim)
        if not (ylim == None):
            plt.ylim(ylim)
        fig.append(_fig)
    
    if len(fig) == 1:
        fig = fig[-1]
    #fig = fig[0]

    if filename is not None:
        plt.savefig(filename)

    if show:
        plt.show()
    else:
        plt.clf()
        plt.close()

    return fig

def export_instance(
    obj: "laygo2.object.physical.Instance",
    scale = 1,
    colormap: dict = None,
    order: list = None,
    xlim: list = None,
    ylim: list = None,
    filename: str = None,
):
    """
    Export a laygo2.object.physical.Instance object to a matplotlib plot.

    Parameters
    ----------
    obj: laygo2.object.physical.Instance
        The instance object to exported.
    scale: float
        (optional) The scaling factor between laygo2's integer coordinates and plot coordinates.
    colormap: dict
        A dictionary that contains layer-color mapping information.
    order: list
        A list that contains the order of layers to be displayed (former is plotted first).
    xlim: list
        (optional) A list that specifies the range of plot in x-axis.
    ylim: list
        (optional) A list that specifies the range of plot in y-axis.
    filename: str
        (optional) If specified, export a output file for the plot.

    Returns
    -------
    matplotlib.pyplot.figure or list: The generated figure object(s).

    """

    # colormap
    if colormap is None:
        colormap = dict()

    # a list to align layered objects in order
    if order is None:
        order = []

    '''
    # xlim and ylim
    if xlim is None:
        xlim = [obj.bbox[0][0] - obj.width, obj.bbox[1][0] + obj.width]
    if ylim is None:
        ylim = [obj.bbox[0][1] - obj.height, obj.bbox[1][1] + obj.height]
    '''

    fig = plt.figure()
    pypobjs = []
    ax = fig.add_subplot(111)
    pypobjs += _translate_obj(obj.name, obj, colormap, scale=scale)
    for _alignobj in order:
        for _pypobj in pypobjs:
            if _pypobj[1] == _alignobj:  # layer is matched.
                if isinstance(_pypobj[0], str):
                    if _pypobj[0] == 'text':  # Text
                        # [["text", obj.layer[0], obj.text, obj.xy]]
                        color = "black"
                        ax.annotate(
                            _pypobj[2], (_pypobj[3][0], _pypobj[3][1]), color=color, weight="bold", fontsize=6, ha="center", va="center"
                        )
                elif _pypobj[0].__class__ == matplotlib.patches.Rectangle:  # Rect
                    ax.add_patch(_pypobj[0])
                    if len(_pypobj) == 3:  # annotation.
                        ax.add_artist(_pypobj[0])
                        rx, ry = _pypobj[0].get_xy()
                        cx = rx + _pypobj[0].get_width() / 2.0
                        cy = ry + _pypobj[0].get_height() / 2.0
                        if _pypobj[1] == "__instance_pin__":
                            color = _pypobj[0].get_edgecolor()
                        else:
                            color = "black"
                        ax.annotate(
                            _pypobj[2], (cx, cy), color=color, weight="bold", fontsize=6, ha="center", va="center"
                        )

    # scale
    plt.autoscale()
    if not (xlim is None):
        plt.xlim(xlim)
    if not (ylim is None):
        plt.ylim(ylim)

    if filename is not None:
        plt.savefig(filename)

    plt.show()

    return fig


def export_grid(
    obj: "laygo2.object.physical.Grid",
    colormap: dict = None,
    order: list = None,
    xlim: list = None,
    ylim: list = None,
    filename: str = None,
    fig: matplotlib.pyplot.figure = None,
    ax: matplotlib.pyplot.axis = None,
    for_annotation: bool = False,
):
    """
    Export a laygo2.object.grid.Grid object to a matplotlib plot.

    Parameters
    ----------
    obj: laygo2.object.grid.Grid
        The grid object to exported.
    scale: float
        (optional) The scaling factor between laygo2's integer coordinates and plot coordinates.
    colormap: dict
        A dictionary that contains layer-color mapping information.
    order: list
        A list that contains the order of layers to be displayed (former is plotted first).
    xlim: list
        (optional) A list that specifies the range of plot in x-axis.
    ylim: list
        (optional) A list that specifies the range of plot in y-axis.
    filename: str
        (optional) If specified, export a output file for the plot.
    fig: matplotlib.pyplot.figure
        (optional) The figure object to be plotted on.
    ax: matplotlib.pyplot.axis
        (optional) The axis object to be plotted on.
    for_annotation: bool
        (optional) If True, only draw the dashed grid lines.

    Returns
    -------
    matplotlib.pyplot.figure or list: The generated figure object(s).

    """
    # colormap
    if colormap is None:
        colormap = dict()

    if for_annotation:
        lw = 0.5
        linestyle = "--"
        wscl = 0
    else:
        lw = 2.0
        linestyle = "-"
        wscl = 1

    # a list to align layered objects in order
    if order is None:
        order = []

    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    pypobjs = []
    # scope
    _xy = (obj.vgrid.range[0], obj.hgrid.range[0])
    _width = obj.vgrid.range[1] - obj.vgrid.range[0]
    _height = obj.hgrid.range[1] - obj.hgrid.range[0]
    rect = matplotlib.patches.Rectangle(_xy, _width, _height, facecolor="none", edgecolor="black", linestyle=linestyle, lw=lw)
    ax.add_patch(rect)
    if not for_annotation:
        rx, ry = rect.get_xy()
        cx = rx + rect.get_width() / 2.0
        cy = ry + rect.get_height() / 2.0
        ax.annotate(
            obj.name, (cx, cy), color="black", weight="bold", fontsize=6, ha="center", va="center"
        )
    if obj.__class__ == laygo2.object.grid.RoutingGrid:  # Routing grid
        if for_annotation:
            rng = range(obj.vgrid<=xlim[0],obj.vgrid>=xlim[1])
            height0 = ylim[1]
        else:
            rng = range(len(obj.vgrid.elements))
            height0 = obj.hgrid.range[1]
        for i in rng:  # vertical routing grid
            ve = obj.vgrid[i][0]
            _xy = (ve - obj.vwidth[i]/2*wscl, obj.hgrid.range[0]-obj.vextension[i])
            _width = obj.vwidth[i]*wscl
            _height = height0 - obj.hgrid.range[0] + 2 * obj.vextension[i]
            facecolor=colormap[obj.vlayer[i][0]][1]
            edgecolor=colormap[obj.vlayer[i][0]][0]
            alpha=colormap[obj.vlayer[i][0]][2]
            rect = matplotlib.patches.Rectangle(_xy, _width, _height, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, linestyle=linestyle, lw=lw)
            ax.add_patch(rect)
            _xy = (ve, obj.hgrid.range[0]-obj.vextension[i])
            ax.annotate(_xy[0], _xy, color="black", weight="bold", fontsize=6, ha="center", va="center")
        if for_annotation:
            rng = range(obj.hgrid<=ylim[0],obj.hgrid>=ylim[1])
            width0 = xlim[1]
        else:
            rng = range(len(obj.hgrid.elements)) 
            width0 = obj.vgrid.range[1] 
        for i in rng: # horizontal routing grid
            he = obj.hgrid[i][0]
            _xy = (obj.vgrid.range[0] - obj.hextension[i], he - obj.hwidth[i]/2*wscl)
            _width = width0 - obj.vgrid.range[0] + 2*obj.hextension[i]
            _height = obj.hwidth[i]*wscl
            facecolor=colormap[obj.hlayer[i][0]][1]
            edgecolor=colormap[obj.hlayer[i][0]][0]
            alpha=colormap[obj.hlayer[i][0]][2]
            rect = matplotlib.patches.Rectangle(_xy, _width, _height, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, linestyle=linestyle, lw=lw)
            ax.add_patch(rect)
            _xy = (obj.vgrid.range[0] - obj.hextension[i], he)
            ax.annotate(_xy[1], _xy, color="black", weight="bold", fontsize=6, ha="center", va="center")
        # viamap
        if for_annotation is False:
            for i in range(len(obj.vgrid.elements)):  # vertical routing grid
                for j in range(len(obj.hgrid.elements)):  # horizontal routing grid
                    v = obj.viamap[i, j]
                    x = obj.vgrid.elements[i]
                    y = obj.hgrid.elements[j]
                    circ = matplotlib.patches.Circle((x, y), radius=2, facecolor="black", edgecolor="black") #, **kwargs)
                    ax.add_patch(circ)
                    ax.annotate(v.name, (x+2, y), color="black", fontsize=4, ha="left", va="bottom")

    for _alignobj in order:
        for _pypobj in pypobjs:
            if _pypobj[1] == _alignobj:  # layer is matched.
                if isinstance(_pypobj[0], str):
                    if _pypobj[0] == 'text':  # Text
                        # [["text", obj.layer[0], obj.text, obj.xy]]
                        color = "black"
                        ax.annotate(
                            _pypobj[2], (_pypobj[3][0], _pypobj[3][1]), color=color, weight="bold", fontsize=6, ha="center", va="center"
                        )
                elif _pypobj[0].__class__ == matplotlib.patches.Rectangle:  # Rect
                    ax.add_patch(_pypobj[0])
                    if len(_pypobj) == 3:  # annotation.
                        ax.add_artist(_pypobj[0])
                        rx, ry = _pypobj[0].get_xy()
                        cx = rx + _pypobj[0].get_width() / 2.0
                        cy = ry + _pypobj[0].get_height() / 2.0
                        if _pypobj[1] == "__instance_pin__":
                            color = _pypobj[0].get_edgecolor()
                        else:
                            color = "black"
                        ax.annotate(
                            _pypobj[2], (cx, cy), color=color, weight="bold", fontsize=6, ha="center", va="center"
                        )

    # scale
    plt.autoscale()
    if not for_annotation:
        if not (xlim == None):
            plt.xlim(xlim)
        if not (ylim == None):
            plt.ylim(ylim)

    if filename is not None:
        plt.savefig(filename)

    if not for_annotation:
        plt.show()

    return fig
