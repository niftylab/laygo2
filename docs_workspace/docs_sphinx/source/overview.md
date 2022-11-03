# Overview


## Introduction

**The LAYout with Gridded Object 2 (laygo2)** package is a Python-based layout generation framework to achieve the following goals:

* **Automation** and reuse of layout generation process.

* **Parametrized** layout generation.

* **Advanced technology** support (such as FinFET).

* (Dynamic) **template**-and-**grid**-based layout generation. 

laygo2 extends the capability of its previous version (**[laygo](https://ieeexplore.ieee.org/document/9314047)**), which is orignally developed as one of layout generation engines in 
**Berkeley Analog Generator2 (BAG2)**.

## Quick trial
For example procedures to install and generate simple structures, see the **[Trial](trial.md)** page.

## API docmentation
Please see the latest **[API reference](laygo2.rst)** for the anatomy of laygo2.

## Typical design flow
1. **Load technology** parameters, primitive templates, and grids from _laygo2_tech_.
    - Check [laygo2_tech.load_templates()](https://github.com/niftylab/laygo2/blob/master/laygo2/examples/laygo2_tech/laygo2_tech_templates.py)
    and [laygo2_tech.load_grids()](https://github.com/niftylab/laygo2/blob/master/laygo2/examples/laygo2_tech/laygo2_tech_grids.py).
1. **Generate instances** from templates.
    - Check [object.Template.generate()](https://laygo2.github.io/laygo2.object.template.Template.html#laygo2.object.template.Template.generate).
1. **Place** the generated instances.
    - Check [object.database.Design.place()](https://laygo2.github.io/laygo2.object.database.Design.html#laygo2.object.database.Design.place).
1. **Route** wires and vias between the instances' terminals.
    - Check [object.database.Design.route()](https://laygo2.github.io/laygo2.object.database.Design.html#laygo2.object.database.Design.route).
1. Create **Pins**. 
    - Check [object.database.Design.pin()](https://laygo2.github.io/laygo2.object.database.Design.html#laygo2.object.database.Design.pin).
1. **Export** the generated design in proper format.
    - Check [interface.skill.export()](https://laygo2.github.io/laygo2.interface.skill.html#laygo2.interface.skill.export).
1.  _(Optional)_ export the design as a new template. 
    - Check [interface.yaml.export_template()](https://laygo2.github.io/laygo2.interface.html#laygo2.interface.yaml.export_template).

## List of developers
See the **[github repository README](https://github.com/niftylab/laygo2)** for the full list of developers and contributors.

## License
**laygo2** is distributed under the BSD 3-Clause License.

