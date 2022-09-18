# Tech Setup

To set up laygo2 for a new technology, prepare the following files first to be imported in the layout generators for template and grid generations.

* ***\_\_init\_\_.py***: The package definition file.
* **laygo2_tech_templates.py**: Contains code for template definitions.
* **laygo2_tech_grids.py**: Contains code for grid definitions.
* **laygo2_tech.yaml**: Contains technology-related parameters (used by _template.py, _grid.py files).
* **laygo2_tech.layermap**: (optional) Contains layer information defined by process-speckfic PDK.
* **laygo2_tech.lyp**: (optional) Contains layer display information for KLayout.

A simple example for the technology setup for Laygo2 can be found [here](https://github.com/niftylab/laygo2/tree/master/laygo2/examples/laygo2_tech).

Please find detailed descriptions for the listed files below:


## \_\_init\_\_.py

Contains import statements for technlogy-specific functions such as load_templates() and load_grids().


## laygo2_tech_templates.py

Implements technology-specific templates and their functions. It also contains load_template() function, 
which returns a TemplateLibrary object containing template objects (MOSFET, Capacitors, Resistors) of core devices.

The templates are typically implemented by utilizing on of the predefined template classes, 
such as NativeInstanceTemplate(for vanilla instances), ParameterizedInstanceTemplate(for PCell instances), 
UserDefinedTemplate(for user-defined virtual instances). Or users can define their own template classes.


## laygo2_tech_grids.py

Implements technology-specific placement/routing grids and their function. It also contains load_grid() function,
which returns a GridLibrary object containing grid objects.


## laygo2_tech.yaml

This yaml file contains technology-specific parameters (such as unit resolution, pin location, structure size, and grid parameters) 
for template and grid generation functions in the _template.py and _grid.py files.
Instead of using the yaml file to store technology information, users can hard-code all required numeric files in the template and grid files.


## laygo2_tech.layermap

Contains layer mapping information for the target technology, which is used for layout object generation, gds export, and skill export.

The layermap file is normally provided by the technology ventor.
If the layermap file is created manually by users, please use the following format to define layer parameters.

*layername layerpurpose stream_layer_number datatype*

(please find the example layermap file for reference).


