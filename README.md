# laygo_refactor
This repository is temporarily built to reconstruct the [laygo](https://ucb-art.github.io/laygo/) package in private 
mode. 

The LAYout with Gridded Object (LAYGO) is a set of API (Application Programming
 Interface) which is used to construct physical design (or layout) of fully 
 customized integrated circuits. The LAYGO provides a rich and productive layout 
 design environment for circuit designers, by utilizing the Python programming 
 language as well as new design concepts such as template and grid based layout
 generations.
 
The Python programming language is selected to build the LAYGO API for the 
 following reasons:
 
1. Its high productivity and readability help users to build their layout 
 generation scripts easily.
1. Python supports advanced programming concepts such as object-oriented 
 programming, lambda, slicing and list comprehensions, which are very helpful 
 describing parameterized, flexible, abstracted, and reusable generators.
1. Its comprehensive libraries in scientific computing (eg. NumPy) and 
 vectorized computations can be widely utilized for various data processing and 
 computations in the layout generation process.
 
The LAYGO utilized various advanced layout generation concepts as well, to 
 enhance the layout design productivity, especially for advanced CMOS 
 technologies, and they are summarized below:
 
1. Layout objects are abstracted as templates to be reused across different 
 components and designs with different technologies.
1. The templates and other layout objects are placed on technology-specific 
 grids, to abstract complicated design rules underlying the object placements.
1. The layout objects are often placed based on relative information between 
 objects, to further abstract the placement information. 
1. Wires connecting the layout objects are also routed on grids to abstract 
 design rules and make the routing functions reusable across different 
 technologies.
  
The LAYGO API was initially developed by Jaeduk Han and after several 
 modifications and improvements, LAYGO2, a new version of LAYGO, is released.

The major improvements of LAYGO2 from the original LAYGO are as follows:

1. The framework structure is reorganized to be more object-oriented and 
distributed, to enhance its modularity and reusability.

1. Used integer-based coordinate systems even for physical coordinates instead 
of floating-pointed numbers, to avoid conversion errors.
    * Note: the use of transformation matrices (Mt) and direction matrices (Md)
    are still relying on intermediate 0.5x scaling and floating point 
    operations, which will be updated in later updates.

1. Supports more advance template constructions in addition to the native 
instance based one in the original LAYGO. In LAYGO2, templates can produce 
either parameterized instances (PCells) or more complex layout structures from 
user-defined generator functions or user-defined template classes.

1. Improved instance and grid indexing systems.

    * Arrayed objects can access their child elements directly using numpy-based 
    indexing and slicing methods.
    * Grid conversion can be done in a similar way, while their indices span 
    to infinity (beyond the range where the grid is defined).

1. Code are refactored and cleaned up. 

1. Added more documentations and tests. 

1. Separated the generator part from the repo.


