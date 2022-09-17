# Setting up laygo2 for a new technology

To set up laygo2 for a new technology, prepare the following files first to be imported in the layout generators for template and grid generations.

* ***\_\_init\_\_.py***: The package definition file.
* **laygo2_tech_templates.py**: Contains code for template definitions.
* **laygo2_tech_grids.py**: Contains code for grid definitions.
* **laygo2_tech.yaml**: Contains technology-related parameters (used by _template.py, _grid.py files).
* **laygo2_tech.layermap**: (optional) Contains layer information defined by process-speckfic PDK.
* **laygo2_tech.lyp**: (optional) Contains layer display information for KLayout.

A simple example for the technology setup for Laygo2 can be found [here](https://github.com/niftylab/laygo2/tree/master/laygo2/examples/laygo2_tech).

Please find detailed descriptions for the listed files below:


## laygo2_tech_templates.py

The python module that contains technology-specific templates. When the load_templates() in the module is called,
it produces various template objects(MOSFET, Capacitors, Resistors) and returns them as a TemplateLibrary object.

The template can be either one of NativeInstanceTemplate(for vanilla instances), ParameterizedInstanceTemplate(for PCell instances), 
UserDefinedTemplate(for user-defined virtual instances), or newly defined types by users.


## laygo2_tech_grids.py

This module contains various placement/routing grids for the target process. When the load_girds() function is called,
it produces various grid objects for the target technology and returns them as a GridLibrary object.


## laygo2_tech.yaml

This yaml file contains various parameters for template and grid generation functions in the _template.py and _grid.py files.
This file is not a mendatory one, as the py files can be described without using any external parameter files. 
The example file contains various technology parameters such as unit resolutons, pin locations, structure sizes, and grid parameters.


## \_\_init\_\_.py

This initialization file contains code to read out load_templates / load_grids funtions when the package is loaded.


## laygo2_tech.layermap

Contains layer mapping information for the target technology, which is used for layout object generation, gds export, and skill export.

The layermap file is normally provided by the technology ventor.
If the layermap file is created manually by users, please use the following format to define layer information.

*layername layerpurpose stream_layer_number datatype*

(please find the example layermap file for reference).


