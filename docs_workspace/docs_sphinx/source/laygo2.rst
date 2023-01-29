API Reference
========================

**LAYout with Gridded Object 2 (laygo2)** is an open-source project 
that offers libraries for automating the creation of custom circuit 
layouts, which satisfy design rules, performance and area efficiency 
in advanced CMOS processes. **laygo2** aims to 1) streamline the layout 
design process through automation, while 2) provide rich parameterization 
capability and 3) compability with cutting-edge technologies such as 
FinFET, making it easier for circuit/mask designers to enhance their 
design productivity.

**laygo2** is implemented with a comprehensive collection of layout 
object classes and methods, written in the Python programmanging language, 
rending it suitible for code-based layout generation.

**laygo2** is cimposed of the following subpackages, each serving 
a distinct purpose:

* **laygo2.object**: houses the core object classes, responsible for defining
                     physical layout structures and design hierarchies.

* **laygo2.interface**: implements the interfaces to external data structures 
                        and EDA tools (**yaml**, **skill**, **BAG**, **GDS**, 
                        **matplotlib** and others).

* **laygo2.util**: provides a range of utility functions for internal use.

For further information regarding the subpackages and modules in **laygo**,
please refer the provided link:

.. toctree::
   :maxdepth: 6

   laygo2.object
   laygo2.interface
   laygo2.util

