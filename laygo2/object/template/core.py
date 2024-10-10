#!/usr/bin/python
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

__author__ = "Jaeduk Han"
__maintainer__ = "Jaeduk Han"
__status__ = "Prototype"

from abc import *
import numpy as np
import laygo2.object


class Template(metaclass=ABCMeta):
    """
    The base class that defines the functions and attributes of the template.

    """

    name = None
    """str: Template name."""

    def __init__(self, name=None):
        """
        The constructor function.

        Parameters
        ----------
        name : str or None, optional.
            The name of this template.
        """
        self.name = name

    def __str__(self):
        """Return a string representation of this template's information. """
        return self.summarize()

    def summarize(self):
        """Return the summary of the template information."""
        return (
            self.__repr__() + " "
            "name: "
            + self.name
            + ", "
            + "class: "
            + self.__class__.__name__
            + ", "
            + ""
        )

    def height(self, params=None):
        """int: Return the height of the template."""
        return abs(self.bbox(params=params)[0, 1] - self.bbox(params=params)[1, 1])

    def width(self, params=None):
        """int: Return the width of the template."""
        return abs(self.bbox(params=params)[0, 0] - self.bbox(params=params)[1, 0])

    def size(self, params=None):
        """int: Return the size of the template."""
        return np.array([self.width(params=params), self.height(params=params)])

    @abstractmethod
    def bbox(self, params=None):
        """numpy.ndarray: (Abstract method) the physical bounding box of the template."""
        pass

    @abstractmethod
    def pins(self, params=None):
        """dict: (Abstract method) Dictionary storing the pins of the template."""
        pass

    @abstractmethod
    def generate(self, name=None, shape=None, pitch=None, transform="R0", netmap=None, params=None):
        """instance: (Abstract method) Generate an instance from the template."""
        pass
        


class NativeInstanceTemplate(Template):
    """
    NativeInstanceTemplate class implements the template that generate Instance.

    """

    libname = None
    """str: Library name.

    Example
    -------
    >>> from laygo2.object.physical import Pin
    >>> from laygo2.object.template import NativeInstanceTemplate
    >>> p = dict() 
    >>> p['i'] = Pin(xy=[[0, 0], [10, 10]], layer=['M1', 'drawing'], 
                     netname='i')
    >>> p['o'] = Pin(xy=[[90, 90], [100, 100]], layer=['M1', 'drawing'], 
                     netname='o')
    >>> nt = NativeInstanceTemplate(libname='mylib', cellname='mytemp', 
                                    bbox=[[0, 0], [100, 100]], pins=p)
    >>> nt.libname 
    'mylib'
    
    .. image:: ../assets/img/object_template_NativeInstanceTemplate_libname.png
          :height: 250

    """

    cellname = None
    """str: Cell name.

    Example
    -------
    >>> from laygo2.object.physical import Pin
    >>> from laygo2.object.template import NativeInstanceTemplate
    >>> p = dict() 
    >>> p['i'] = Pin(xy=[[0, 0], [10, 10]], layer=['M1', 'drawing'], 
                     netname='i')
    >>> p['o'] = Pin(xy=[[90, 90], [100, 100]], layer=['M1', 'drawing'], 
                     netname='o')
    >>> nt = NativeInstanceTemplate(libname='mylib', cellname='mytemp', 
                                    bbox=[[0, 0], [100, 100]], pins=p)
    >>> nt.cellname 
    'mytemp'
    
    .. image:: ../assets/img/object_template_NativeInstanceTemplate_cellname.png
          :height: 250

    """

    _bbox = np.array([[0, 0], [0, 0]])

    _pins = None

    def __init__(self, libname, cellname, bbox=np.array([[0, 0], [0, 0]]), pins=None):
        """
        Constructor function.

        Parameters
        ----------
        libname : str
            Library name.
        cellname : str
            Cell name.
        bbox : numpy.ndarray
            Bounding box of the object.
        pins : dict
            Dictionary storing the template's pin objects.

        Returns
        -------
        NativeInstanceTemplate

        Example
        -------
        >>> from laygo2.object.physical import Pin
        >>> from laygo2.object.template import NativeInstanceTemplate
        >>> p = dict() 
        >>> p['i'] = Pin(xy=[[0, 0], [10, 10]], layer=['M1', 'drawing'], 
                         netname='i')
        >>> p['o'] = Pin(xy=[[90, 90], [100, 100]], layer=['M1', 'drawing'], 
                         netname='o')
        >>> nt = NativeInstanceTemplate(libname='mylib', cellname='mytemp', 
                                        bbox=[[0, 0], [100, 100]], pins=p)
        <laygo2.object.template.NativeInstanceTemplate object>
        >>> print(nt)
        <laygo2.object.template.NativeInstanceTemplate object at 0x0000013C01..> 
         name: mytemp, class: NativeInstanceTemplate, bbox: [[0, 0], [100, 100]], 
         pins: {'i': <laygo2.object.physical.Pin object at 0x0000013C01CEFDC0>, 
                'o': <laygo2.object.physical.Pin object at 0x0000013C01C30BE0>},

        .. image:: ../assets/img/object_template_NativeInstanceTemplate_init.png
          :height: 250

        """
        self.libname = libname
        self.cellname = cellname
        self._bbox = None if bbox is None else np.asarray(bbox)
        self._pins = pins
        Template.__init__(self, name=cellname)

    def summarize(self):
        """Return the summary of the object information."""
        return (
            self.__repr__() + " "
            "name: "
            + self.name
            + ", "
            + "class: "
            + self.__class__.__name__
            + ", "
            + "bbox: "
            + str(self.bbox().tolist())
            + ", "
            + "pins: "
            + str(self.pins())
            + ", "
            + ""
        )

    # Core template functions
    def bbox(self, params=None):
        """
        Bounding box of the object.

        Parameters
        ----------
        None

        Returns
        -------
        numpy.ndarray

        Example
        -------
        >>> from laygo2.object.physical import Pin
        >>> from laygo2.object.template import NativeInstanceTemplate
        >>> p = dict() 
        >>> p['i'] = Pin(xy=[[0, 0], [10, 10]], layer=['M1', 'drawing'], 
                         netname='i')
        >>> p['o'] = Pin(xy=[[90, 90], [100, 100]], layer=['M1', 'drawing'], 
                         netname='o')
        >>> nt = NativeInstanceTemplate(libname='mylib', cellname='mytemp', 
                                        bbox=[[0, 0], [100, 100]], pins=p)
        >>> nt.bbox()
        array([[  0,   0],
               [100, 100]])

        .. image:: ../assets/img/object_template_NativeInstanceTemplate_bbox.png
          :height: 250

        """
        return self._bbox

    def pins(self, params=None):
        """
        Dictionary storing pins of the object.

        Parameters
        ----------
        None

        Returns
        -------
        dict : Dictionary storing pin objects.

        Example
        -------
        >>> from laygo2.object.physical import Pin
        >>> from laygo2.object.template import NativeInstanceTemplate
        >>> p = dict() 
        >>> p['i'] = Pin(xy=[[0, 0], [10, 10]], layer=['M1', 'drawing'], 
                         netname='i')
        >>> p['o'] = Pin(xy=[[90, 90], [100, 100]], layer=['M1', 'drawing'], 
                         netname='o')
        >>> nt = NativeInstanceTemplate(libname='mylib', cellname='mytemp', 
                                        bbox=[[0, 0], [100, 100]], pins=p)
        >>> nt.pins()
        >>> {'i': <laygo2.object.physical.Pin object at 0x000002578C762500>, 
             'o': <laygo2.object.physical.Pin object at 0x00000257FBA87C40>}                   
        >>> print(nt.pins()['i']) 
            <laygo2.object.physical.Pin object at 0x000002578C762500> 
                name: None,
                class: Pin,
                xy: [[0, 0], [10, 10]],
                params: None, 
                layer: ['M1' 'drawing'], 
                netname: i, 
                shape: None, 
                master: None

        .. image:: ../assets/img/object_template_ParameterizedInstanceTemplate_pins.png
          :height: 250

        """
        return self._pins

    def generate(self, name=None, shape=None, pitch=None, transform="R0", netmap=None, params=None):
        """
        Generate Instance object.

        Parameters
        ----------
        name : str
            Name of the instance to be generated.
        shape : numpy.ndarray, optional.
            Shape of the object to be generated.
        pitch : numpy.ndarray, optional.
            Element pitch of the array object to be generated.
        transform : str
            Transformation attribute of the object to be generated.
        netmap : dict, optional.
            Dictionary storing net-pin conversion mappings.
        params : dict, optional.
            Dictionary storing the parameters associated with the object.

        Returns
        -------
        (laygo2.object.physical.Instance) generated Instance object

        See Also
        --------
        Class Instance

        Example
        -------
        >>> from laygo2.object.physical import Pin
        >>> from laygo2.object.template import NativeInstanceTemplate
        >>> p = dict() 
        >>> p['i'] = Pin(xy=[[0, 0], [10, 10]], layer=['M1', 'drawing'], 
                         netname='i')
        >>> p['o'] = Pin(xy=[[90, 90], [100, 100]], layer=['M1', 'drawing'], 
                         netname='o')
        >>> nt = NativeInstanceTemplate(libname='mylib', cellname='mytemp', 
                                        bbox=[[0, 0], [100, 100]], pins=p)
        >>> i0 = nt.generate(name="I0")
        >>> i1 = nt.generate(name="I1")
        >>> print(i0)
        <laygo2.object.physical.Instance object at 0x000002C3F717F850> 
        name: I0,
        class: Instance,
        xy: [0, 0],
        params: None,
        size: [100, 100],
        shape: None,
        pitch: [100, 100],
        transform: R0,
        pins: {'i': <laygo2.object.physical.Pin object at 0x000002C3F717F820>, 
               'o': <laygo2.object.physical.Pin object at 0x000002C3F717FD90>},
        >>> print(i1)
        <laygo2.object.physical.Instance object at 0x000002C3FF91C100> 
        name: I1,
        class: Instance,
        xy: [0, 0],
        params: None,
        size: [100, 100],
        shape: None,
        pitch: [100, 100],
        transform: R0,
        pins: {'i': <laygo2.object.physical.Pin object at 0x000002C3FF91C0A0>, 
               'o': <laygo2.object.physical.Pin object at 0x000002C3FF91C070>},
        
        .. image:: ../assets/img/object_template_ParameterizedInstanceTemplate_generate.png
          :height: 250

        """
        inst = laygo2.object.physical.Instance(
            libname=self.libname,
            cellname=self.cellname,
            xy=np.array([0, 0]),
            shape=shape,
            pitch=pitch,
            unit_size=self.size(params),
            pins=self.pins(params),
            transform=transform,
            name=name,
            params=params,
        )
        # update netnames if netmap is provided.
        if netmap is not None:
            inst.update_netname(netmap=netmap)  
        return inst 

    # I/O functions
    def export_to_dict(self):
        """
        Return a dictionary containing the template information.

        Parameters
        ----------
        None

        Returns
        -------
        dict

        See Also
        --------
        Class Instance

        Example
        -------
        >>> from laygo2.object.physical import Pin
        >>> from laygo2.object.template import NativeInstanceTemplate
        >>> p = dict() 
        >>> p['i'] = Pin(xy=[[0, 0], [10, 10]], layer=['M1', 'drawing'], 
                         netname='i')
        >>> p['o'] = Pin(xy=[[90, 90], [100, 100]], layer=['M1', 'drawing'], 
                         netname='o')
        >>> nt = NativeInstanceTemplate(libname='mylib', cellname='mytemp', 
                                        bbox=[[0, 0], [100, 100]], pins=p)
        >>> nt.export_to_dict()
            {'libname': 'mylib', 
             'cellname': 'mytemp', 
             'bbox': [[0, 0], [100, 100]], 
             'pins': {'i': {'xy': [[0, 0], [10, 10]], 
                            'layer': ['M1', 'drawing'], 
                            'name': None, 
                            'netname': 'i'}, 
                      'o': {'xy': [[90, 90], [100, 100]], 
                            'layer': ['M1', 'drawing'], 
                            'name': None, 
                            'netname': 'o'}
                     }
            }

        .. image:: ../assets/img/object_template_NativeInstanceTemplate_export_to_dict.png
          :height: 250

        """
        db = dict()
        db["libname"] = self.libname
        db["cellname"] = self.cellname
        db["bbox"] = self.bbox().tolist()
        db["pins"] = dict()
        for pn, p in self.pins().items():
            db["pins"][pn] = p.export_to_dict()
        return db


class ParameterizedInstanceTemplate(Template):
    """
    The ParameterizedInstanceTemplate class implements a template for 
    generating instances with varying size (bounding box) and pin 
    configurations based on input parameters, such as instances mapped 
    to Cadence Virtuoso's pcells or Pycells.

    """

    libname = None
    """str: The libname of the instance to be generated."""

    cellname = None
    """str: The cellname of the instance to be generated."""

    _bbox = None

    _pins = None

    def __init__(self, libname, cellname, bbox_func=None, pins_func=None):
        """
        Constructor of ParameterizedInstanceTemplate class.

        Parameters
        ----------
        libname : str
            The library name.
        cellname : str
            The cell name of the template.
        bbox_func : callable
            The function that computes the bounding box of the template from 
            its input parameters.
        pins_func : callable
            The function that returns a dictionary that contains its pin 
            objects for its input parameters.

        Returns
        -------
        laygo2.object.template.ParameterizedInstanceTemplate

        Example
        -------
        >>> import numpy as np
        >>> from laygo2.object.template import ParameterizedInstanceTemplate
        >>> from laygo2.object.physical import Pin
        >>> # bbox computation function.
        >>> def pcell_bbox_func(params):
                return np.array([[0, 0], [100 * params["mult"], 100]])
        >>> # pin generation function.
        >>> def pcell_pins_func(params):
        >>>     template_pins = dict()
        >>>     for i in range(params["mult"]):
        >>>         template_pins["in" + str(i)] = Pin(
        >>>             xy=[[i * 100 + 0, 0], [i * 100 + 10, 10]],
        >>>             layer=["M1", "drawing"],
        >>>             netname="in" + str(i),
        >>>         )
        >>>         template_pins["out" + str(i)] = Pin(
        >>>             xy=[[i * 100 + 90, 90], [i * 100 + 90, 100]],
        >>>             layer=["M1", "drawing"],
        >>>             netname="out" + str(i),
        >>>         )
        >>>     return template_pins
        >>> # Create a template
        >>> pcell_temp = ParameterizedInstanceTemplate(
        >>>     libname="mylib",
        >>>     cellname="mypcelltemplate",
        >>>     bbox_func=pcell_bbox_func,
        >>>     pins_func=pcell_pins_func,
        >>> )
        >>> # Generate an instance for input parameters.
        >>> pcell_inst_params = {"mult": 4}
        >>> pcell_inst = pcell_temp.generate(
        >>>    name="mypcellinst",
        >>>    transform="R0",
        >>>    params=pcell_inst_params,
        >>> )
        >>> # display
        >>> print(pcell_temp)
            <laygo2.object.template.ParameterizedInstanceTemplate object at 0x000001B1BA91D9C0>
            name: mypcelltemplate, 
            class: ParameterizedInstanceTemplate,
        >>> print(pcell_inst)
            xy: [0, 0],
            params: {'mult': 4},
            size: [400, 100],
            shape: None,
            pitch: [400, 100],
            transform: R0,
            pins: {'in0': <laygo2.object.physical.Pin object at 0x000001B1BA91FCA0>, 
                   'out0': <laygo2.object.physical.Pin object at 0x000001B1BA91FC70>, 
                   'in1': <laygo2.object.physical.Pin object at 0x000001B1BA91FC10>, 
                   'out1': <laygo2.object.physical.Pin object at 0x000001B1BA91E8C0>, 
                   'in2': <laygo2.object.physical.Pin object at 0x000001B1BA91E890>, 
                   'out2': <laygo2.object.physical.Pin object at 0x000001B1BA91FBE0>, 
                   'in3': <laygo2.object.physical.Pin object at 0x000001B1BA91E7D0>, 
                   'out3': <laygo2.object.physical.Pin object at 0x000001B1BA91E710>},

        .. image:: ../assets/img/object_template_ParameterizedInstanceTemplate_init.png
          :height: 250

        """
        self.libname = libname
        self.cellname = cellname
        self._bbox = bbox_func
        self._pins = pins_func
        Template.__init__(self, name=cellname)

    # Core template functions
    def bbox(self, params=None):
        """
        Bounding box of the template object.

        Parameters
        ----------
        params: dict
            A dictionary that contains input parameters corresponding to the 
            bounding box to be computed.

        Returns
        -------
        numpy.ndarray: A 2x2 numpy array that contains the bounding box 
            coordinates corresponding to the input parameters.

        Example
        -------
        >>> import numpy as np
        >>> from laygo2.object.template import ParameterizedInstanceTemplate
        >>> from laygo2.object.physical import Pin
        >>> # bbox computation function.
        >>> def pcell_bbox_func(params):
                return np.array([[0, 0], [100 * params["mult"], 100]])
        >>> # pin generation function.
        >>> def pcell_pins_func(params):
        >>>     template_pins = dict()
        >>>     for i in range(params["mult"]):
        >>>         template_pins["in" + str(i)] = Pin(
        >>>             xy=[[i * 100 + 0, 0], [i * 100 + 10, 10]],
        >>>             layer=["M1", "drawing"],
        >>>             netname="in" + str(i),
        >>>         )
        >>>         template_pins["out" + str(i)] = Pin(
        >>>             xy=[[i * 100 + 90, 90], [i * 100 + 90, 100]],
        >>>             layer=["M1", "drawing"],
        >>>             netname="out" + str(i),
        >>>         )
        >>>     return template_pins
        >>> # Create a template
        >>> pcell_temp = ParameterizedInstanceTemplate(
        >>>     libname="mylib",
        >>>     cellname="mypcelltemplate",
        >>>     bbox_func=pcell_bbox_func,
        >>>     pins_func=pcell_pins_func,
        >>> )
        >>> # Compute bbox for input parameters
        >>> pcell_inst_params = {"mult": 4}
        >>> pcell_temp.bbox(params=pcell_inst_params)
        array([[  0,   0],
               [400, 100]])

        .. image:: ../assets/img/object_template_ParameterizedInstanceTemplate_bbox.png
          :height: 250

        """
        return self._bbox(params=params)

    def pins(self, params=None):
        """
        Return pin dictionary of ParameterizedInstanceTemplate object.

        Parameters
        ----------
        params: dict
            A dictionary that contains input parameters corresponding to the 
            pin objects to be produced.

        Returns
        -------
        dict: A dictionary that contains pin object corresponding to the 
            input parameters.

        Example
        -------
        >>> import numpy as np
        >>> from laygo2.object.template import ParameterizedInstanceTemplate
        >>> from laygo2.object.physical import Pin
        >>> # bbox computation function.
        >>> def pcell_bbox_func(params):
                return np.array([[0, 0], [100 * params["mult"], 100]])
        >>> # pin generation function.
        >>> def pcell_pins_func(params):
        >>>     template_pins = dict()
        >>>     for i in range(params["mult"]):
        >>>         template_pins["in" + str(i)] = Pin(
        >>>             xy=[[i * 100 + 0, 0], [i * 100 + 10, 10]],
        >>>             layer=["M1", "drawing"],
        >>>             netname="in" + str(i),
        >>>         )
        >>>         template_pins["out" + str(i)] = Pin(
        >>>             xy=[[i * 100 + 90, 90], [i * 100 + 90, 100]],
        >>>             layer=["M1", "drawing"],
        >>>             netname="out" + str(i),
        >>>         )
        >>>     return template_pins
        >>> # Create a template
        >>> pcell_temp = ParameterizedInstanceTemplate(
        >>>     libname="mylib",
        >>>     cellname="mypcelltemplate",
        >>>     bbox_func=pcell_bbox_func,
        >>>     pins_func=pcell_pins_func,
        >>> )
        >>> # Compute bbox for input parameters
        >>> pcell_inst_params = {"mult": 4}
        >>> pcell_temp.pins(params=pcell_inst_params)
            {'in0': <laygo2.object.physical.Pin object at 0x000001B1BA91F7C0>, 
             'out0': <laygo2.object.physical.Pin object at 0x000001B1BA91E830>,
             'in1': <laygo2.object.physical.Pin object at 0x000001B1BA91DAB0>, 
             'out1': <laygo2.object.physical.Pin object at 0x000001B1BA91E860>, 
             'in2': <laygo2.object.physical.Pin object at 0x000001B1BA91E560>, 
             'out2': <laygo2.object.physical.Pin object at 0x000001B1BA91E800>, 
             'in3': <laygo2.object.physical.Pin object at 0x000001B1BA91E770>, 
             'out3': <laygo2.object.physical.Pin object at 0x000001B1BA91EA40>}

        .. image:: ../assets/img/object_template_ParameterizedInstanceTemplate_pins.png
          :height: 250

        """
        return self._pins(params=params)

    def generate(self, name=None, shape=None, pitch=None, transform="R0", netmap=None, params=None):
        """
        Generate an Instance object corresponding to the template and its 
        input parameters.

        Parameters
        ----------
        name : str
            name of the instance to be generated.
        shape : numpy.ndarray, optional.
            shape of the object to be generated.
        pitch : numpy.ndarray, optional.
            pitch of the object to be generated.
        transform : str, optional.
            transformation attribute of the entity to be generated.
        netmap : dict, optional.
            dictionary containing netmap conversion information of pins.
        params : dict, optional.
            dictionary having the entity attributes.

        Returns
        -------
        laygo2.object.physical.Instance: The generated Instance object

        Example
        -------
        >>> import numpy as np
        >>> from laygo2.object.template import ParameterizedInstanceTemplate
        >>> from laygo2.object.physical import Pin
        >>> # bbox computation function.
        >>> def pcell_bbox_func(params):
                return np.array([[0, 0], [100 * params["mult"], 100]])
        >>> # pin generation function.
        >>> def pcell_pins_func(params):
        >>>     template_pins = dict()
        >>>     for i in range(params["mult"]):
        >>>         template_pins["in" + str(i)] = Pin(
        >>>             xy=[[i * 100 + 0, 0], [i * 100 + 10, 10]],
        >>>             layer=["M1", "drawing"],
        >>>             netname="in" + str(i),
        >>>         )
        >>>         template_pins["out" + str(i)] = Pin(
        >>>             xy=[[i * 100 + 90, 90], [i * 100 + 90, 100]],
        >>>             layer=["M1", "drawing"],
        >>>             netname="out" + str(i),
        >>>         )
        >>>     return template_pins
        >>> # Create a template
        >>> pcell_temp = ParameterizedInstanceTemplate(
        >>>     libname="mylib",
        >>>     cellname="mypcelltemplate",
        >>>     bbox_func=pcell_bbox_func,
        >>>     pins_func=pcell_pins_func,
        >>> )
        >>> # Generate an instance for input parameters.
        >>> pcell_inst_params = {"mult": 4}
        >>> pcell_inst = pcell_temp.generate(
        >>>    name="mypcellinst",
        >>>    transform="R0",
        >>>    params=pcell_inst_params,
        >>> )
        >>> # display
        >>> print(pcell_temp)
            <laygo2.object.template.ParameterizedInstanceTemplate object at 0x000001B1BA91D9C0>
            name: mypcelltemplate, 
            class: ParameterizedInstanceTemplate,
        >>> print(pcell_inst)
            xy: [0, 0],
            params: {'mult': 4},
            size: [400, 100],
            shape: None,
            pitch: [400, 100],
            transform: R0,
            pins: {'in0': <laygo2.object.physical.Pin object at 0x000001B1BA91FCA0>, 
                   'out0': <laygo2.object.physical.Pin object at 0x000001B1BA91FC70>, 
                   'in1': <laygo2.object.physical.Pin object at 0x000001B1BA91FC10>, 
                   'out1': <laygo2.object.physical.Pin object at 0x000001B1BA91E8C0>, 
                   'in2': <laygo2.object.physical.Pin object at 0x000001B1BA91E890>, 
                   'out2': <laygo2.object.physical.Pin object at 0x000001B1BA91FBE0>, 
                   'in3': <laygo2.object.physical.Pin object at 0x000001B1BA91E7D0>, 
                   'out3': <laygo2.object.physical.Pin object at 0x000001B1BA91E710>},

        .. image:: ../assets/img/object_template_ParameterizedInstanceTemplate_generate.png
          :height: 250

        """
        # xy = xy + np.dot(self.xy(params)[0], tf.Mt(transform).T)
        inst = laygo2.object.physical.Instance(
            libname=self.libname,
            cellname=self.cellname,
            xy=np.array([0, 0]),
            shape=shape,
            pitch=pitch,
            unit_size=self.size(params),
            pins=self.pins(params),
            transform=transform,
            name=name,
            params=params,
        )
        # update netnames if netmap is provided.
        if netmap is not None:
            inst.update_netname(netmap=netmap)  
        return inst


class UserDefinedTemplate(Template):
    """
    UserDefinedTemplate class implements the template that generates 
    a VirtualInstance object corresponding to the template and input 
    parameters.

    """

    _bbox = None
    """The internal pointer to the bbox computing function."""

    _pins = None
    """The internal pointer to the pin creation function."""

    _generate = None
    """The internal pointer to the instance generation function."""

    def __init__(self, bbox_func, pins_func, generate_func, name=None):
        """
        Constructor function of UserDefinedTemplate class.

        Parameters
        ----------
        bbox_func: callable
            The function that computes the bounding box of the template from 
            its input parameters.
        pins_func: callable
            The function that returns a dictionary that contains its pin 
            objects for its input parameters.
        generate_func: callable
            The function that returns a generated VirtualInstance object 
            for its input parameters.
        name : str
            The name of the template.

        Returns
        -------
        laygo2.object.template.UserDefinedTemplate

        Example
        -------
        >>> import numpy as np
        >>> from laygo2.object.template import UserDefinedTemplate
        >>> from laygo2.object.physical import Pin, Rect, VirtualInstance
        >>> # bbox computation function.
        >>> def user_bbox_func(params):
        >>>     return np.array([[0, 0], [100 * params["mult"], 100]])
        >>> # pin generation function.
        >>> def user_pins_func(params):
        >>>     template_pins = dict()
        >>>     for i in range(params["mult"]):
        >>>         template_pins["in" + str(i)] = Pin(
        >>>             xy=[[i * 100 + 0, 0], [i * 100 + 10, 10]],
        >>>             layer=["M1", "drawing"],
        >>>             netname="in" + str(i),
        >>>         )
        >>>         template_pins["out" + str(i)] = Pin(
        >>>             xy=[[i * 100 + 90, 90], [i * 100 + 90, 100]],
        >>>             layer=["M1", "drawing"],
        >>>             netname="out" + str(i),
        >>>         )
        >>>     return template_pins
        >>> # instance generation function.
        >>> def user_generate_func(
        >>>     name=None, shape=None, pitch=np.array([0, 0]), transform="R0", 
        >>>     params=None):
        >>>     m = params["mult"]
        >>>     shape = np.array([1, 1]) if shape is None else np.asarray(shape)
        >>>     inst_pins = user_pins_func(params)
        >>>     inst_native_elements = dict()
        >>>     for i in range(m):
        >>>         ofst = i * 100
        >>>         inst_native_elements["R0_" + str(i)] = Rect(
        >>>             xy=[[ofst, 0], [ofst + 10, 10]], layer=["M1", "drawing"]
        >>>         )
        >>>         inst_native_elements["R1_" + str(i)] = Rect(
        >>>             xy=[[ofst + 90, 90], [ofst + 100, 100]], layer=["M1", "drawing"]
        >>>         )
        >>>     inst_native_elements["R2"] = Rect(
        >>>         xy=[[0, 0], [m * 100, 100]], layer=["prBoundary", "drawing"]
        >>>     )
        >>>     inst = VirtualInstance(
        >>>         name=name,
        >>>         libname="mylib",
        >>>         cellname="myvinst",
        >>>         xy=np.array([0, 0]),
        >>>         native_elements=inst_native_elements,
        >>>         shape=shape,
        >>>         pitch=pitch,
        >>>         unit_size=[m * 100, 100],
        >>>         pins=inst_pins,
        >>>         transform=transform,
        >>>         params=params,
        >>>     )
        >>>     return inst
        >>> # UserDefinedTemplate construction.
        >>> user_temp = UserDefinedTemplate(
        >>>     name="myusertemplate",
        >>>     bbox_func=user_bbox_func,
        >>>     pins_func=user_pins_func,
        >>>     generate_func=user_generate_func,
        >>> )
        >>> # VirtualInstance generation.
        >>> user_inst = user_temp.generate(name="myinst", params={"mult": 5})
        >>> # Display
        >>> print(user_temp)
            <laygo2.object.template.UserDefinedTemplate object at 0x00000192BF990130> 
            name: myusertemplate, class: UserDefinedTemplate,
        >>> print(user_inst)
            <laygo2.object.physical.VirtualInstance object at 0x00000192BF990280>
                name: myinst,
                class: VirtualInstance,
                xy: [0, 0],
                params: {'mult': 5},
                size: [500, 100],
                shape: [1, 1],
                pitch: [500, 100],
                transform: R0,
                pins: {'in0': <laygo2.object.physical.Pin object at 0x00000192BF9930D0>, 
                       'out0': <laygo2.object.physical.Pin object at 0x00000192BF9931C0>, 
                       'in1': <laygo2.object.physical.Pin object at 0x00000192BF993760>, 
                       'out1': <laygo2.object.physical.Pin object at 0x00000192BF9936A0>, 
                       'in2': <laygo2.object.physical.Pin object at 0x00000192BF993610>, 
                       'out2': <laygo2.object.physical.Pin object at 0x00000192BF9935B0>, 
                       'in3': <laygo2.object.physical.Pin object at 0x00000192BF9932E0>, 
                       'out3': <laygo2.object.physical.Pin object at 0x00000192BF9931F0>, 
                       'in4': <laygo2.object.physical.Pin object at 0x00000192BF993130>, 
                       'out4': <laygo2.object.physical.Pin object at 0x00000192BF9930A0>},
                native elements: {'R0_0': <laygo2.object.physical.Rect object at 0x0...>, 
                                  'R1_0': <laygo2.object.physical.Rect object at 0x0...>, 
                                  'R0_1': <laygo2.object.physical.Rect object at 0x0...>, 
                                  'R1_1': <laygo2.object.physical.Rect object at 0x0...>, 
                                  'R0_2': <laygo2.object.physical.Rect object at 0x0...>, 
                                  'R1_2': <laygo2.object.physical.Rect object at 0x0...>, 
                                  'R0_3': <laygo2.object.physical.Rect object at 0x0...>, 
                                  'R1_3': <laygo2.object.physical.Rect object at 0x0...>, 
                                  'R0_4': <laygo2.object.physical.Rect object at 0x0...>, 
                                  'R1_4': <laygo2.object.physical.Rect object at 0x0...>, 
                                  'R2': <laygo2.object.physical.Rect object at 0x000...>}
        >>> print(user_inst.bbox)
            [[  0   0]
             [500 100]]

        .. image:: ../assets/img/object_template_UserDefinedTemplate_init.png
          :height: 250

        """
        self._bbox = bbox_func
        self._pins = pins_func
        self._generate = generate_func
        Template.__init__(self, name=name)

    # Core template functions
    def bbox(self, params=None):
        """
        Return bbox of UserDefinedTemplate object.

        Parameters
        ----------
        params: dict
            A dictionary that contains input parameters corresponding to the 
            bounding box to be computed.

        Returns
        -------
        numpy.ndarray: A 2x2 numpy array that contains the bounding box 
            coordinates corresponding to the input parameters.

        Example
        -------
        >>> import numpy as np
        >>> from laygo2.object.template import UserDefinedTemplate
        >>> from laygo2.object.physical import Pin, Rect, VirtualInstance
        >>> # bbox computation function.
        >>> def user_bbox_func(params):
        >>>     return np.array([[0, 0], [100 * params["mult"], 100]])
        >>> # pin generation function.
        >>> def user_pins_func(params):
        >>>     template_pins = dict()
        >>>     for i in range(params["mult"]):
        >>>         template_pins["in" + str(i)] = Pin(
        >>>             xy=[[i * 100 + 0, 0], [i * 100 + 10, 10]],
        >>>             layer=["M1", "drawing"],
        >>>             netname="in" + str(i),
        >>>         )
        >>>         template_pins["out" + str(i)] = Pin(
        >>>             xy=[[i * 100 + 90, 90], [i * 100 + 90, 100]],
        >>>             layer=["M1", "drawing"],
        >>>             netname="out" + str(i),
        >>>         )
        >>>     return template_pins
        >>> # instance generation function.
        >>> def user_generate_func(
        >>>     name=None, shape=None, pitch=np.array([0, 0]), transform="R0", 
        >>>     params=None):
        >>>     m = params["mult"]
        >>>     shape = np.array([1, 1]) if shape is None else np.asarray(shape)
        >>>     inst_pins = user_pins_func(params)
        >>>     inst_native_elements = dict()
        >>>     for i in range(m):
        >>>         ofst = i * 100
        >>>         inst_native_elements["R0_" + str(i)] = Rect(
        >>>             xy=[[ofst, 0], [ofst + 10, 10]], layer=["M1", "drawing"]
        >>>         )
        >>>         inst_native_elements["R1_" + str(i)] = Rect(
        >>>             xy=[[ofst + 90, 90], [ofst + 100, 100]], layer=["M1", "drawing"]
        >>>         )
        >>>     inst_native_elements["R2"] = Rect(
        >>>         xy=[[0, 0], [m * 100, 100]], layer=["prBoundary", "drawing"]
        >>>     )
        >>>     inst = VirtualInstance(
        >>>         name=name,
        >>>         libname="mylib",
        >>>         cellname="myvinst",
        >>>         xy=np.array([0, 0]),
        >>>         native_elements=inst_native_elements,
        >>>         shape=shape,
        >>>         pitch=pitch,
        >>>         unit_size=[m * 100, 100],
        >>>         pins=inst_pins,
        >>>         transform=transform,
        >>>         params=params,
        >>>     )
        >>>     return inst
        >>> # UserDefinedTemplate construction.
        >>> user_temp = UserDefinedTemplate(
        >>>     name="myusertemplate",
        >>>     bbox_func=user_bbox_func,
        >>>     pins_func=user_pins_func,
        >>>     generate_func=user_generate_func,
        >>> )
        >>> user_temp.bbox(params={"mult": 5})
        array([[  0,   0],
               [500, 100]])

        .. image:: ../assets/img/object_template_UserDefinedTemplate_bbox.png
          :height: 250

        """
        return self._bbox(params=params)

    def pins(self, params=None):
        """
        Pins of UserDefinedTemplate object.

        Parameters
        ----------
        params: dict
            A dictionary that contains input parameters corresponding to the 
            pin objects to be produced.

        Returns
        -------
        dict: A dictionary that contains pin object corresponding to the 
            input parameters.

        Example
        -------
        >>> import numpy as np
        >>> from laygo2.object.template import UserDefinedTemplate
        >>> from laygo2.object.physical import Pin, Rect, VirtualInstance
        >>> # bbox computation function.
        >>> def user_bbox_func(params):
        >>>     return np.array([[0, 0], [100 * params["mult"], 100]])
        >>> # pin generation function.
        >>> def user_pins_func(params):
        >>>     template_pins = dict()
        >>>     for i in range(params["mult"]):
        >>>         template_pins["in" + str(i)] = Pin(
        >>>             xy=[[i * 100 + 0, 0], [i * 100 + 10, 10]],
        >>>             layer=["M1", "drawing"],
        >>>             netname="in" + str(i),
        >>>         )
        >>>         template_pins["out" + str(i)] = Pin(
        >>>             xy=[[i * 100 + 90, 90], [i * 100 + 90, 100]],
        >>>             layer=["M1", "drawing"],
        >>>             netname="out" + str(i),
        >>>         )
        >>>     return template_pins
        >>> # instance generation function.
        >>> def user_generate_func(
        >>>     name=None, shape=None, pitch=np.array([0, 0]), transform="R0", 
        >>>     params=None):
        >>>     m = params["mult"]
        >>>     shape = np.array([1, 1]) if shape is None else np.asarray(shape)
        >>>     inst_pins = user_pins_func(params)
        >>>     inst_native_elements = dict()
        >>>     for i in range(m):
        >>>         ofst = i * 100
        >>>         inst_native_elements["R0_" + str(i)] = Rect(
        >>>             xy=[[ofst, 0], [ofst + 10, 10]], layer=["M1", "drawing"]
        >>>         )
        >>>         inst_native_elements["R1_" + str(i)] = Rect(
        >>>             xy=[[ofst + 90, 90], [ofst + 100, 100]], layer=["M1", "drawing"]
        >>>         )
        >>>     inst_native_elements["R2"] = Rect(
        >>>         xy=[[0, 0], [m * 100, 100]], layer=["prBoundary", "drawing"]
        >>>     )
        >>>     inst = VirtualInstance(
        >>>         name=name,
        >>>         libname="mylib",
        >>>         cellname="myvinst",
        >>>         xy=np.array([0, 0]),
        >>>         native_elements=inst_native_elements,
        >>>         shape=shape,
        >>>         pitch=pitch,
        >>>         unit_size=[m * 100, 100],
        >>>         pins=inst_pins,
        >>>         transform=transform,
        >>>         params=params,
        >>>     )
        >>>     return inst
        >>> # UserDefinedTemplate construction.
        >>> user_temp = UserDefinedTemplate(
        >>>     name="myusertemplate",
        >>>     bbox_func=user_bbox_func,
        >>>     pins_func=user_pins_func,
        >>>     generate_func=user_generate_func,
        >>> )
        >>> user_temp.pins(params={"mult": 5})
        {'in0': <laygo2.object.physical.Pin object at 0x00000192BF990670>, 
        'out0': <laygo2.object.physical.Pin object at 0x00000192BF990400>, 
        'in1': <laygo2.object.physical.Pin object at 0x00000192BF993250>, 
        'out1': <laygo2.object.physical.Pin object at 0x00000192BF9903D0>, 
        'in2': <laygo2.object.physical.Pin object at 0x00000192BF9901F0>, 
        'out2': <laygo2.object.physical.Pin object at 0x00000192BF9904F0>, 
        'in3': <laygo2.object.physical.Pin object at 0x00000192BF993640>, 
        'out3': <laygo2.object.physical.Pin object at 0x00000192BF990520>, 
        'in4': <laygo2.object.physical.Pin object at 0x00000192BF9936D0>, 
        'out4': <laygo2.object.physical.Pin object at 0x00000192BF993790>}

        .. image:: ../assets/img/object_template_UserDefinedTemplate_pins.png
          :height: 250

        """
        return self._pins(params=params)

    def generate(self, name=None, shape=None, pitch=None, transform="R0", netmap=None, params=None):
        """
        Generate a VirtualInstance object by calling generate_func() bound to the template.

        Parameters
        ----------
        name : str
            name of the instance to be generated.
        shape : numpy.ndarray, optional.
            shape of the object to be generated.
        pitch : numpy.ndarray, optional.
            pitch of the object to be generated.
        transform : str, optional.
            transformation attribute of the entity to be generated.
        netmap : dict, optional.
            dictionary containing netmap conversion information of pins.
        params : dict, optional.
            dictionary having the entity attributes.

        Returns
        -------
        laygo2.object.physical.VirtualInstance: The generated VirtualInstance object.

        Example
        -------
        >>> import numpy as np
        >>> from laygo2.object.template import UserDefinedTemplate
        >>> from laygo2.object.physical import Pin, Rect, VirtualInstance
        >>> # bbox computation function.
        >>> def user_bbox_func(params):
        >>>     return np.array([[0, 0], [100 * params["mult"], 100]])
        >>> # pin generation function.
        >>> def user_pins_func(params):
        >>>     template_pins = dict()
        >>>     for i in range(params["mult"]):
        >>>         template_pins["in" + str(i)] = Pin(
        >>>             xy=[[i * 100 + 0, 0], [i * 100 + 10, 10]],
        >>>             layer=["M1", "drawing"],
        >>>             netname="in" + str(i),
        >>>         )
        >>>         template_pins["out" + str(i)] = Pin(
        >>>             xy=[[i * 100 + 90, 90], [i * 100 + 90, 100]],
        >>>             layer=["M1", "drawing"],
        >>>             netname="out" + str(i),
        >>>         )
        >>>     return template_pins
        >>> # instance generation function.
        >>> def user_generate_func(
        >>>     name=None, shape=None, pitch=np.array([0, 0]), transform="R0", 
        >>>     params=None):
        >>>     m = params["mult"]
        >>>     shape = np.array([1, 1]) if shape is None else np.asarray(shape)
        >>>     inst_pins = user_pins_func(params)
        >>>     inst_native_elements = dict()
        >>>     for i in range(m):
        >>>         ofst = i * 100
        >>>         inst_native_elements["R0_" + str(i)] = Rect(
        >>>             xy=[[ofst, 0], [ofst + 10, 10]], layer=["M1", "drawing"]
        >>>         )
        >>>         inst_native_elements["R1_" + str(i)] = Rect(
        >>>             xy=[[ofst + 90, 90], [ofst + 100, 100]], layer=["M1", "drawing"]
        >>>         )
        >>>     inst_native_elements["R2"] = Rect(
        >>>         xy=[[0, 0], [m * 100, 100]], layer=["prBoundary", "drawing"]
        >>>     )
        >>>     inst = VirtualInstance(
        >>>         name=name,
        >>>         libname="mylib",
        >>>         cellname="myvinst",
        >>>         xy=np.array([0, 0]),
        >>>         native_elements=inst_native_elements,
        >>>         shape=shape,
        >>>         pitch=pitch,
        >>>         unit_size=[m * 100, 100],
        >>>         pins=inst_pins,
        >>>         transform=transform,
        >>>         params=params,
        >>>     )
        >>>     return inst
        >>> # UserDefinedTemplate construction.
        >>> user_temp = UserDefinedTemplate(
        >>>     name="myusertemplate",
        >>>     bbox_func=user_bbox_func,
        >>>     pins_func=user_pins_func,
        >>>     generate_func=user_generate_func,
        >>> )
        >>> # VirtualInstance generation.
        >>> user_inst = user_temp.generate(name="myinst", params={"mult": 5})
        >>> # Display
        >>> print(user_temp)
            <laygo2.object.template.UserDefinedTemplate object at 0x00000192BF990130> 
            name: myusertemplate, class: UserDefinedTemplate,
        >>> print(user_inst)
            <laygo2.object.physical.VirtualInstance object at 0x00000192BF990280>
                name: myinst,
                class: VirtualInstance,
                xy: [0, 0],
                params: {'mult': 5},
                size: [500, 100],
                shape: [1, 1],
                pitch: [500, 100],
                transform: R0,
                pins: {'in0': <laygo2.object.physical.Pin object at 0x00000192BF9930D0>, 
                       'out0': <laygo2.object.physical.Pin object at 0x00000192BF9931C0>, 
                       'in1': <laygo2.object.physical.Pin object at 0x00000192BF993760>, 
                       'out1': <laygo2.object.physical.Pin object at 0x00000192BF9936A0>, 
                       'in2': <laygo2.object.physical.Pin object at 0x00000192BF993610>, 
                       'out2': <laygo2.object.physical.Pin object at 0x00000192BF9935B0>, 
                       'in3': <laygo2.object.physical.Pin object at 0x00000192BF9932E0>, 
                       'out3': <laygo2.object.physical.Pin object at 0x00000192BF9931F0>, 
                       'in4': <laygo2.object.physical.Pin object at 0x00000192BF993130>, 
                       'out4': <laygo2.object.physical.Pin object at 0x00000192BF9930A0>},
                native elements: {'R0_0': <laygo2.object.physical.Rect object at 0x0...>, 
                                  'R1_0': <laygo2.object.physical.Rect object at 0x0...>, 
                                  'R0_1': <laygo2.object.physical.Rect object at 0x0...>, 
                                  'R1_1': <laygo2.object.physical.Rect object at 0x0...>, 
                                  'R0_2': <laygo2.object.physical.Rect object at 0x0...>, 
                                  'R1_2': <laygo2.object.physical.Rect object at 0x0...>, 
                                  'R0_3': <laygo2.object.physical.Rect object at 0x0...>, 
                                  'R1_3': <laygo2.object.physical.Rect object at 0x0...>, 
                                  'R0_4': <laygo2.object.physical.Rect object at 0x0...>, 
                                  'R1_4': <laygo2.object.physical.Rect object at 0x0...>, 
                                  'R2': <laygo2.object.physical.Rect object at 0x000...>}
        >>> print(user_inst.bbox)
            [[  0   0]
             [500 100]]

        .. image:: ../assets/img/object_template_UserDefinedTemplate_generate.png
          :height: 250

        """
        inst = self._generate(
            name=name, shape=shape, pitch=pitch, transform=transform, params=params
        )
        # update netnames if netmap is provided.
        if netmap is not None:
            inst.update_netname(netmap=netmap)  
        return inst


# Test
if __name__ == "__main__":
    test_native_template = True
    test_pcell_template = True
    test_user_template = True

    import laygo2.object

    if test_native_template:
        print("NativeInstanceTemplate test")
        # define pins
        nat_temp_pins = dict()
        nat_temp_pins["in"] = laygo2.object.Pin(
            xy=[[0, 0], [10, 10]], layer=["M1", "drawing"], netname="in"
        )
        nat_temp_pins["out"] = laygo2.object.Pin(
            xy=[[90, 90], [100, 100]], layer=["M1", "drawing"], netname="out"
        )
        # create a template
        nat_temp = NativeInstanceTemplate(
            libname="mylib",
            cellname="mynattemplate",
            bbox=[[0, 0], [100, 100]],
            pins=nat_temp_pins,
        )
        # generate
        nat_inst = nat_temp.generate(
            name="mynatinst", shape=[2, 2], pitch=[100, 100], transform="R0"
        )
        # display
        print(nat_temp)
        print(nat_inst)

    if test_pcell_template:
        print("ParameterizedInstanceTemplate test")

        # define the bbox computation function.
        def pcell_bbox_func(params):
            return np.array([[0, 0], [100 * params["mult"], 100]])

        # define the pin generation function.
        def pcell_pins_func(params):
            template_pins = dict()
            for i in range(params["mult"]):
                template_pins["in" + str(i)] = laygo2.object.Pin(
                    xy=[[i * 100 + 0, 0], [i * 100 + 10, 10]],
                    layer=["M1", "drawing"],
                    netname="in" + str(i),
                )
                template_pins["out" + str(i)] = laygo2.object.Pin(
                    xy=[[i * 100 + 90, 90], [i * 100 + 90, 100]],
                    layer=["M1", "drawing"],
                    netname="out" + str(i),
                )
            return template_pins

        # create a template.
        pcell_temp = ParameterizedInstanceTemplate(
            libname="mylib",
            cellname="mypcelltemplate",
            bbox_func=pcell_bbox_func,
            pins_func=pcell_pins_func,
        )
        # generate based on the parameter assigned.
        pcell_inst_params = {"mult": 4}
        pcell_inst_size = pcell_temp.size(params=pcell_inst_params)
        pcell_inst = pcell_temp.generate(
            name="mypcellinst",
            shape=[2, 2],
            pitch=pcell_inst_size,
            transform="R0",
            params=pcell_inst_params,
        )
        # display
        print(pcell_temp)
        print(pcell_inst)

    if test_user_template:
        print("UserDefinedTemplate test")

        # define the bbox computation function.
        def user_bbox_func(params):
            return np.array([[0, 0], [100 * params["mult"], 100]])

        # define the pin generation function.
        def user_pins_func(params):
            template_pins = dict()
            for i in range(params["mult"]):
                template_pins["in" + str(i)] = laygo2.object.Pin(
                    xy=[[i * 100 + 0, 0], [i * 100 + 10, 10]],
                    layer=["M1", "drawing"],
                    netname="in" + str(i),
                )
                template_pins["out" + str(i)] = laygo2.object.Pin(
                    xy=[[i * 100 + 90, 90], [i * 100 + 90, 100]],
                    layer=["M1", "drawing"],
                    netname="out" + str(i),
                )
            return template_pins

        # define the instance generation function.
        def user_generate_func(
            name=None, shape=None, pitch=np.array([0, 0]), transform="R0", params=None
        ):
            m = params["mult"]
            shape = np.array([1, 1]) if shape is None else np.asarray(shape)

            inst_pins = user_pins_func(params)
            inst_native_elements = dict()
            for i in range(m):
                ofst = i * 100
                inst_native_elements["R0_" + str(i)] = laygo2.object.Rect(
                    xy=[[ofst, 0], [ofst + 10, 10]], layer=["M1", "drawing"]
                )
                inst_native_elements["R1_" + str(i)] = laygo2.object.Rect(
                    xy=[[ofst + 90, 90], [ofst + 100, 100]], layer=["M1", "drawing"]
                )
            inst_native_elements["R2"] = laygo2.object.Rect(
                xy=[[0, 0], [m * 100, 100]], layer=["prBoundary", "drawing"]
            )
            inst = laygo2.object.VirtualInstance(
                name=name,
                libname="mylib",
                cellname="myvinst",
                xy=np.array([0, 0]),
                native_elements=inst_native_elements,
                shape=shape,
                pitch=pitch,
                unit_size=[m * 100, 100],
                pins=inst_pins,
                transform=transform,
                params=params,
            )
            return inst

        user_temp = UserDefinedTemplate(
            name="myusertemplate",
            bbox_func=user_bbox_func,
            pins_func=user_pins_func,
            generate_func=user_generate_func,
        )
        user_inst = user_temp.generate(
            name="myuserinst", shape=[2, 1], params={"mult": 5}
        )
        print(user_temp)
        print(user_inst)
        print(user_inst.bbox)
