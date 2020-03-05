# Gaussian Accumulator

A Gaussian Accumulator refers to the notion of discretizing the **surface** of the unit sphere (a gaussian surface) into buckets/cells. One can then integrate/accumulate a list of **points** (aka unit normals) into these buckets.
The end result is then a histogram of the sphere. There are many choices for the discretization process, however this library uses equilateral triangles because each cell will have the exact same **area** and **shape**. This process done by *refining* an icosahedron. The following image shows our discretization strategy. The first object discretizes a sphere with uniform spacing of phi/theta (not small cells at poles), the second object is an icosahedron, the third object is the first level of refinement for an icosahdron, the last object is the second level of refinement of an icosahedron.

![Icosahedron](/assets/imgs/refined_icosahedron.png)

Once a level of refinement is chosen, one can then integrate point vectors into the cells/buckets. For example integrating the normals of the following mesh would in *half* a gaussian sphere would look like this:

![GaussianAccumulator](/assets/imgs/gaussian_accumulator_example.png)

To do this one must **find** the cell that corresponds to the point. This is a search process that has been implemented in several fashions in this repo. The main ways are as follows:

* 3D KD Tree - Do a nearest neighbor search using a binary tree.
    - `GaussianAccumulatorKDPY` - One implementation using scipy kdtree.
    - `GaussianAccumulatorKD` One implementation uses C++ nanoflann.
* Global Index and Local Search - A 3D point is transformed to a unique integer id. The closest id is found corresponding to triangle cell. A local search of triangle neighbors is performed to find closest triangle cell to point.
    - `GaussianAccumulatorOpt` - Works good on top hemisphere. Projects 3D point to plane using Azimuth Equidistant projection. Convert 2D point to int32 index using Hilbert Curve.
    - `GaussianAccumulatorKDS2` -Uses Googles S2 Global index. 3D point is projected to unit cube, assigned to a face, and then a Hilbert curve index is created.

## Installation

This project uses CMake. You can build using the provided Makefile which will call CMake commands for you. For example to build just `make` and it will perform the following steps:

1. `mkdir cmake-build && cd cmake-build` 
2. `cd cmake-build` 
3. `cmake ../ -DCMAKE_BUILD_TYPE=Release -DWERROR=0` 
4. `cmake --build . -j$(nproc)` 

### Python

1. Install [conda](https://conda.io/projects/conda/en/latest/) or create a python virtual envrionment ([Why?](https://medium.freecodecamp.org/why-you-need-python-environments-and-how-to-manage-them-with-conda-85f155f4353c)). I recommend conda for Windows users.
2. Perform `CMake` build as described above
3. `cd cmake-build && cmake --build . --target python-package --config Release` 
4. `cd lib/python_package &&  pip install -e .` 

If you want to run the examples then you need to install the following (from main directory):

1. `pip install -r requirements-dev.txt` 

