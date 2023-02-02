# Overview


## Introduction

The **LAYout with Gridded Object 2 (laygo2)** package is a comprehensive 
framework for IC layout generation, written in the Python programming 
language. Its primary objects are to:

* **Automate** and streamline the layout generation process, promoting 
reuse and efficiency.

* Enable **Parametrized** layout generation, allowing for greater 
flexibility and customization.

* Provide support for **advanced technologies** suc as FinFET, 
ensuring compatibility with the latest CMOS fabrication processes.

* Implement dynamic **template**-and-**grid**-based layout generation,
facilitating the rapid and effective creation of layouts. 

**laygo2** builds upon the capabilities of its predecessor, 
**[laygo](https://ieeexplore.ieee.org/document/9314047)**, 
which was orignally developed as one of layout generation 
engines in the **Berkeley Analog Generator2 (BAG2)** platform.

## Quick trial
For step-by-step instructions on how to install the software and generate simple structures, refer to the **[Trial](trial.md)** page.

## API documentation
For an in-depth understanding of the **laygo2** software architecture 
and componenets, refer to the latest **[API reference](laygo2.rst)**. 
This reference provides a comprehensive overview of the package's 
functions, modules, and classes, allowing users to effectively utilize 
the full range of **laygo2**'s capabilities.

## Typical design flow
The following steps outline the standard procedure for constructing 
a generator code for a custom layout using the **laygo2** package:

1. Load the necessary **technology** parameters, primitive templates, and grids from the _laygo2_tech_ module.
    - Check [laygo2_tech.load_templates()](https://github.com/niftylab/laygo2/blob/master/laygo2/examples/laygo2_tech/laygo2_tech_templates.py)
    and [laygo2_tech.load_grids()](https://github.com/niftylab/laygo2/blob/master/laygo2/examples/laygo2_tech/laygo2_tech_grids.py) for further details.
1. **Generate instances** from the loaded templates.
    - Check [object.Template.generate()](https://laygo2.github.io/laygo2.object.template.Template.html#laygo2.object.template.Template.generate) for further details.
1. Perform **placement** of the generated instances.
    - Check [object.database.Design.place()](https://laygo2.github.io/laygo2.object.database.Design.html#laygo2.object.database.Design.place) for further details.
1. **Route** wires and vias between the instances' terminals.
    - Check [object.database.Design.route()](https://laygo2.github.io/laygo2.object.database.Design.html#laygo2.object.database.Design.route), [object.database.Design.route_via_track()](https://laygo2.github.io/laygo2.object.database.Design.html#laygo2.object.database.Design.route_via_track), and [object.routing.RoutingMesh](https://laygo2.github.io/laygo2.object.routing.RoutingMesh.html) for further details.
2. Create the necessary **Pins**. 
    - Check [object.database.Design.pin()](https://laygo2.github.io/laygo2.object.database.Design.html#laygo2.object.database.Design.pin) for further details.
3. **Export** the completed design in the desired format.
    - Check [interface.skill.export()](https://laygo2.github.io/laygo2.interface.skill.html#laygo2.interface.skill.export), [interface.skillbridge.export()](https://laygo2.github.io/laygo2.interface.skillbridge.html#laygo2.interface.skillbridge.export), [interface.bag.export()](https://laygo2.github.io/laygo2.interface.bag.html#laygo2.interface.bag.export), [interface.gdspy.export()](https://laygo2.github.io/laygo2.interface.gdspy.html#laygo2.interface.gdspy.export), [interface.magic.export()](https://laygo2.github.io/laygo2.interface.magic.html#laygo2.interface.magic.export), and [interface.mpl.export()](https://laygo2.github.io/laygo2.interface.mpl.html#laygo2.interface.mpl.export) for further details.
4.  _(Optional)_ Save the design as a new **template** for future reuse. 
    - Check [interface.yaml.export_template()](https://laygo2.github.io/laygo2.interface.html#laygo2.interface.yaml.export_template) for details.

## Technology setup
See the **[Tech Setup](technology.md)** section for detailed instructions on setting up the technology configurations.

## List of developers
See the **[README](https://github.com/niftylab/laygo2)** file in the Github repository for a complete list of developers and contributors to the project.

## License
**laygo2** is distributed under the BSD 3-Clause License.

