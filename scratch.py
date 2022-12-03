# -*- coding: utf-8 -*-
import laygo2
import laygo2.interface
import laygo2_tech_quick_start as tech

# Parameter definitions #############
# Design Variables
cellname = "test"
# Design hierarchy
libname = "layout_generated"

# (optional) matplotlib export
mpl_params = tech.tech_params["mpl"]
# End of parameter definitions ######
print(mpl_params)

"""## Create a design hierarchy

Simliar to conventional design frameworks, laygo2 supports hierarchical management of multiple designs with libraries and design entities. Run the following code cell to create a [Library](https://laygo2.github.io/laygo2.object.database.Library.html) and [Design](https://laygo2.github.io/laygo2.object.database.Design.html) object for the layout to be generated.
"""

# 2. Create a design hierarchy
lib = laygo2.object.database.Library(name=libname)
dsn = laygo2.object.database.Design(name=cellname, libname=libname)
lib.append(dsn)

"""## Create and place rectangles

The first step to create a layout after setting up the design entity is generating instances from template. By running the [generate()](https://laygo2.github.io/laygo2.object.template.UserDefinedTemplate.html#laygo2.object.template.UserDefinedTemplate.generate) function with design parameters as its arguements, you can create and backstage an instance (or virtual instance) corresponding to the template and parameters.
"""

rect0 = laygo2.object.physical.Rect(xy=[[0, 0], [100, 100]], layer=['metal1', 'drawing'])
print(rect0)
dsn.append(rect0)
rect1 = laygo2.object.physical.Rect(xy=[[100, 100], [200, 200]], layer=['metal1', 'drawing'])
print(rect1)
dsn.append(rect1)
print(dsn)
# plot
#fig = laygo2.interface.mpl.export(lib, cellname=None, colormap=mpl_params["colormap"], order=mpl_params["order"], xlim=[-100, 600], ylim=[-100, 600])

text0 = laygo2.object.physical.Text(xy=[0, 200], layer=['text', 'drawing'], text='test', params=None)
print(text0)
dsn.append(text0)
print(dsn)
# plot
#fig = laygo2.interface.mpl.export(lib, cellname=None, colormap=mpl_params["colormap"], order=mpl_params["order"], xlim=[-100, 600], ylim=[-100, 600])

pin0 = laygo2.object.physical.Pin(xy=[[200, 200], [300, 300]], layer=['metal1', 'pin'],
               netname='net0', params={'direction': 'input'})
print(pin0)
dsn.append(pin0)
print(dsn)
# plot
#fig = laygo2.interface.mpl.export(lib, cellname=None, colormap=mpl_params["colormap"], order=mpl_params["order"], xlim=[-100, 600], ylim=[-100, 600])

"""## Create and place instances

Call the [place()](https://laygo2.github.io/laygo2.object.database.Design.place.html) function to place the generated (and backstaged) instances on grid. Note that the placement grid **pg** and its grid conversion function ([pg.mn([x, y])](https://laygo2.github.io/laygo2.object.grid.PlacementGrid.html#laygo2.object.grid.PlacementGrid.mn)) are used in combination with various positional functions (such as [top_left()](https://laygo2.github.io/laygo2.object.grid.Grid.html#laygo2.object.grid.Grid.top_left) and [height_vec()](https://laygo2.github.io/laygo2.object.grid.Grid.html#laygo2.object.grid.Grid.height_vec)) to enable grid-based, relative placements.

The placed instances contain all geometries to compose the transistor structure with base interconnects.
"""

inst0_pins = dict()
inst0_pins['in'] = laygo2.object.physical.Pin(xy=[[0, 0], [10,10]],
        layer = ['metal1', 'pin'], netname = 'in')
inst0_pins['out']= laygo2.object.physical.Pin(xy=[[90, 90], [100, 100]],
        layer=['metal1', 'pin'], netname='out')
inst0 = laygo2.object.physical.Instance(name="I0", xy=[300, 0],
        libname="mylib", cellname="mycell", shape=[3, 2], pitch=[200,200],
        unit_size=[100, 100], pins=inst0_pins, transform='R0')

print(inst0)
#print(inst0.elements)
#print(inst0.elements[0, 0])
#print(inst0.elements[1, 0])
import numpy as np
#for i in np.ndenumerate(inst0.elements):
#        print(i)
#print(inst0.pins)
#print(inst0.pins['in'])
#print(inst0.pins['in'][0, 1])

dsn.append(inst0)
#print(dsn)
# plot
fig = laygo2.interface.mpl.export(lib, cellname=None, colormap=mpl_params["colormap"], order=mpl_params["order"], xlim=[-100, 1100], ylim=[-100, 500])
