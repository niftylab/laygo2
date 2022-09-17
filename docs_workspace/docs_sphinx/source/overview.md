# laygo2 Overview


## Introduction

**The LAYout with Gridded Object 2 (laygo2)** package is a Python-based layout generation framework with the following goals:

* Automation and reuse of layout generation process.

* Parametrized layout generation.

* Layout automation in advanced technologies such as FinFET.

* Code-based layout generation. 

**LAYGO**, the previous version of laygo2, is orignally developed as one of layout generation engines in 
**Berkeley Analog Generator2 (BAG2)**, and laygo2 extends the origial LAYGO's functions and capabilities.
laygo2 can be launched either in standalone mode or in combination with BAG2.

**The major features supported by both LAYGO and laygo2 are summarized below:**

1. **Template and grid-based device placements**: LAYGO/laygo2 enhances the portability and reusability of layout 
generation process by using abstract grids and instances (templates), without directly dealing with physical 
instances and coordinates.
1. **Instance placements based on relative information**: The process portability is maximized by enhancing 
the level of abstraction by utilizing relative information between instances for their placements.
1. **Grid-based wire routing**: The wire routing process is abstracted as well, by placing the interconnects 
on process-independent routing grids. 

**laygo2 has the following improvement items compared to the original laygo.**
1. **Enhanced object-oriented features**: The module dependency is minimized to enhance the process portability 
and reusability. The layout generation process is fully described in object-oriented ways.
For examples, classes defined in the physical module can be used independently, and new template types can be 
easily implemented by inheriting the base Template class.
1. **Integer-based grid computations**: laygo originally used real-number-based grid systems. laygo2 converted 
all grid systems to integer-based ones to avoid floating-point operations. The integer number coordinates are 
converted to real numbers during the final export phase.
1. **Advanced template generation**: In additional to the single-instance-based template (NativeInstanceTemplate),
laygo2 implements various ways of template types, including parameterized-instance-based template
(ParameterizedInstanceTemplate) and user-defined templates (UserDefinedTemplate). laygo2 supports inheritance 
of the primitive template class (Template) for new template definitions.
1. **Advanced instanced and grid indexing**: Instance and grid objects are tightly integrated to Numpy 
array objects, to support advanced indexing and slicing functions. The grid objects especially extend 
the Numpy array to implement unlimited indexing over their defined ranges.
1. **Code quality enhancement and refactoring**.
1. (on-going) **More documents and tests added**.
1. (on-going) **Generator code separated from the main framework**.

## Quick trial
Example procedures to install and generate simple structures are introduced in the **[Trial](trial.md)** page to help users to understand the basic dynamics of laygo2.

## laygo2 structure
The structures of packages and modules in laygo2 are described in the **[Structure](structure.md)** page.

## Typical design flow
1. Load technology parameters, primitive templates, and grids 
(check [laygo2_tech.load_templates()](https://github.com/niftylab/laygo2/blob/master/laygo2/examples/laygo2_tech/laygo2_tech_templates.py) 
and [laygo2_tech.load_grids()](https://github.com/niftylab/laygo2/blob/master/laygo2/examples/laygo2_tech/laygo2_tech_grids.py)).
1. Generate instances from templates (check [object.Template.generate()](https://laygo2.github.io/laygo2.object.template.Template.html#laygo2.object.template.Template.generate)).
2. Place the generated instances (check [object.database.Design.place()](https://laygo2.github.io/laygo2.object.database.Design.html#laygo2.object.database.Design.place)).
3. Route wires and vias between the instances' terminals (check [object.database.Design.route()](https://laygo2.github.io/laygo2.object.database.Design.html#laygo2.object.database.Design.route)).
4. Pin creation (check [object.database.Design.pin()](https://laygo2.github.io/laygo2.object.database.Design.html#laygo2.object.database.Design.pin)).
5. Export the generated design in proper formats 
(check [interface.skill.export()](https://laygo2.github.io/laygo2.interface.html#laygo2.interface.skill.export)).
1.  _(Optional)_ export the design as a new template (check [interface.yaml.export_template()](https://laygo2.github.io/laygo2.interface.html#laygo2.interface.yaml.export_template)).

## List of developers
See the [github repository README](https://github.com/niftylab/laygo2) for the full list of developers and contributors.

## License
laygo2 is distributed under the BSD 3-Clause License.

