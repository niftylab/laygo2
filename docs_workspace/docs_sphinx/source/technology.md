# Tech Setup

The following files needs to be prepared in order to set up **laygo2** 
for a new technology:

* ***\_\_init\_\_.py***: Package definition file.
* **laygo2_tech_templates.py**: Code for template definitions.
* **laygo2_tech_grids.py**: Code for grid definitions.
* **laygo2_tech.yaml**: Technology parameters used by _template.py, _grid.py files.
* **laygo2_tech.layermap**: (optional) Layer information defined by technology-speckfic PDK.
* **laygo2_tech.lyp**: (optional) Layer display information for KLayout.

A simple technology setup example for **laygo2** is available [here](https://github.com/niftylab/laygo2/tree/master/laygo2/examples/laygo2_tech).

Please find detailed descriptions for the listed files below:


## \_\_init\_\_.py

Contains import statements for technlogy-specific functions such as **load_templates()** and **load_grids()**.


## laygo2_tech_templates.py

This file implements technology-specific templates and their 
associated functions. It also contains the **load_template()** function 
which returns a **TemplateLibrary** object that holds template objects 
(such as MOSFETs, capacitors, and resistors) of core devices. 

The templates are typically created using predefined template classes 
such as **NativeInstanceTemplate**(for vanilla instances), 
**ParameterizedInstanceTemplate**(for PCell instances), 
or **UserDefinedTemplate**(for user-defined virtual instances),
or users can define their own template classes.


## laygo2_tech_grids.py

This file implements the technology-specific placement and routing grids 
along with their functions. The **load_grid()** function, in this file, 
returns a **GridLibrary** object which contains grid objects.


## laygo2_tech.yaml

This yaml file contains technology-specific parameters required for 
template and grid generation (such as unit resolution, pin location, 
structure size, and grid parameters).
This file serves as a central repository for all technology-related 
information for template and grid generation functions in the **laygo2_tech_template.py** and **laygo2_tech_grid.py** files.
Instead of using this yaml file, users can also hard-code the required 
the numeric parameters.


## laygo2_tech.layermap

Contains layer mapping information for the target technology.
The layer information is used for layout generation ation, gds export, and skill export.

The layermap file is normally provided by the technology ventor.
If the layermap file is created manually by users, please use the following format to define layer parameters.

*layername layerpurpose stream_layer_number datatype*

(please find the example layermap file in the **laygo2_tech** directiory
 for reference).


