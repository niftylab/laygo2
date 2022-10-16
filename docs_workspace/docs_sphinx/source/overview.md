# Overview


## Introduction

**The LAYout with Gridded Object 2 (laygo2)** package is a Python-based layout generation framework with the following goals:

* **Automation** and reuse of layout generation process.

* **Parametrized** layout generation.

* **Advanced technology** support (such as FinFET).

* (Dynamic) **template**-and-**grid**-based layout generation. 

laygo2 extends the capability of its previous version (**[laygo](https://ieeexplore.ieee.org/document/9314047)**), which is orignally developed as one of layout generation engines in 
**Berkeley Analog Generator2 (BAG2)**.

## Quick trial
Example procedures to install and generate simple structures are introduced in the **[Trial](trial.md)** page to help users to understand the basic dynamics of laygo2.

## API docmentation
The anotomy of laygo2 is described in the **[API reference](laygo2.rst)** page.

## Typical design flow
1. Load technology parameters, primitive templates, and grids from tech-specific setup _laygo2_tech_
(check [laygo2_tech.load_templates()](https://github.com/niftylab/laygo2/blob/master/laygo2/examples/laygo2_tech/laygo2_tech_templates.py) 
and [laygo2_tech.load_grids()](https://github.com/niftylab/laygo2/blob/master/laygo2/examples/laygo2_tech/laygo2_tech_grids.py)).
1. Generate instances from templates (check [object.Template.generate()](https://laygo2.github.io/laygo2.object.template.Template.html#laygo2.object.template.Template.generate)).
2. Place the generated instances (check [object.database.Design.place()](https://laygo2.github.io/laygo2.object.database.Design.html#laygo2.object.database.Design.place)).
3. Route wires and vias between the instances' terminals (check [object.database.Design.route()](https://laygo2.github.io/laygo2.object.database.Design.html#laygo2.object.database.Design.route)).
4. Create Pins (check [object.database.Design.pin()](https://laygo2.github.io/laygo2.object.database.Design.html#laygo2.object.database.Design.pin)).
5. Export the generated design in proper formats 
(check [interface.skill.export()](https://laygo2.github.io/laygo2.interface.html#laygo2.interface.skill.export)).
1.  _(Optional)_ export the design as a new template (check [interface.yaml.export_template()](https://laygo2.github.io/laygo2.interface.html#laygo2.interface.yaml.export_template)).

## List of developers
See the [github repository README](https://github.com/niftylab/laygo2) for the full list of developers and contributors.

## License
laygo2 is distributed under the BSD 3-Clause License.

