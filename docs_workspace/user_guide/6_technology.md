# Setting up Laygo2 for a new technology

To set up Laygo2 for a new technology, prepare the following files first to be imported in the layout generators for template and grid generations.

* ***(technology_name)*_example.layermap**: the layer information defined by process-speckfic PDK.
* ***(technology_name)*_example.lyp**: (optional) layer display information for KLayout.
* ***(technology_name)*_templates.py**: Python code for template definitions.
* ***(technology_name)*_grids.py**: Python code for grid definitions.
* ***(technology_name)*_example.yaml**: (if used by _template.py, _grid.py files) an yaml file that contains layer information.
* ***__init__.py***: Package definition file.

A simple example for the technology setup for Laygo2 can be found [here](../../examples/technology_example).

Please find detailed descriptions for the listed files below:


## *(technology_name)*_example.layermap

Contains layer mapping information for the target technology, which is used for layout object generation, gds export, and skill export.

The layermap file is normally provided by the technology ventor.
If the layermap file is created manually by users, please use the following format to define layer information.

*layername layerpurpose stream_layer_number datatype*

(please find the example layermap file for reference).


## *(technology_name)*_templates.py

The python module that contains technology-specific templates. When the load_templates() in the module is called,
it produces various template objects(MOSFET, Capacitors, Resistors) and returns them as a TemplateLibrary object.

The template can be either one of NativeInstanceTemplate(for vanilla instances), ParameterizedInstanceTemplate(for PCell instances), 
UserDefinedTemplate(for user-defined virtual instances), or newly defined types by users.


## *(technology_name)*_grids.py

This module contains various placement/routing grids for the target process. When the load_girds() function is called,
it produces various grid objects for the target technology and returns them as a GridLibrary object.


## *(technology_name)*_example.yaml

This yaml file contains various parameters for template and grid generation functions in the _template.py and _grid.py files.
This file is not a mendatory one, as the py files can be described without using any external parameter files. 
The example file contains various technology parameters such as unit resolutons, pin locations, structure sizes, and grid parameters.

## __init__.py

This initialization file contains code to read out load_templates / load_grids funtions when the package is loaded.
