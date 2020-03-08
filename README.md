# Gaussian Accumulator

A Gaussian Accumulator refers to the notion of discretizing the **surface** of the unit sphere (a gaussian surface) into buckets/cells. One can then integrate/accumulate a list of **points** (aka unit normals) into these buckets.
The end result is then a histogram of the sphere. There are many choices for the discretization process, however this library uses equilateral triangles because each cell will have the exact same **area** and **shape**. This process is done by *refining* an icosahedron. The following image shows our discretization strategy. The first object discretizes a sphere with uniform spacing of phi/theta (note small cells at poles), the second object is an icosahedron, the third object is the first level of refinement for an icosahdron, the last object is the second level of refinement of an icosahedron.

![Icosahedron](/assets/imgs/refined_icosahedron.png)

Once a level of refinement is chosen, one can then integrate point vectors into the cells/buckets. For example integrating the normals of the following mesh would into a *half* gaussian sphere would look like this:

![GaussianAccumulator](/assets/imgs/gaussian_accumulator_example.png)

To do this one must **find** the cell that corresponds to the point. This is a search process that has been implemented in several fashions in this repo. The main ways are as follows:

* 3D KD Tree - Do a nearest neighbor search using a binary tree.
    - `GaussianAccumulatorKDPY` - One implementation using scipy kdtree.
    - `GaussianAccumulatorKD` One implementation uses C++ nanoflann.
* Global Index and Local Search - A 3D point is transformed to a unique integer id. The unique ids have the property that ids close to each other will be close to each other in 3D space. The closest id is found corresponding to a triangle cell. A local search of triangle neighbors is performed to find closest triangle cell to the point.
    - `GaussianAccumulatorOpt` - Works good on top hemisphere. Projects 3D point to plane using Azimuth Equidistant projection. Convert 2D point to int32 index using Hilbert Curve.
    - `GaussianAccumulatorKDS2`- Works on full sphere! Uses Googles S2 Global index. 3D point is projected to unit cube, assigned to a face of the cube, and then a Hilbert curve index is found for that cube face.

## Installation

This project uses CMake. You can build using the provided Makefile which will call CMake commands for you. For example to build just `make` and it will perform the following steps:

1. `mkdir cmake-build && cd cmake-build` 
2. `cd cmake-build` 
3. `cmake ../ -DCMAKE_BUILD_TYPE=Release -DWERROR=0`. For windows also add `-DCMAKE_GENERATOR_PLATFORM=x64`
4. `cmake --build . -j$(nproc)` 

### Python

1. Install [conda](https://conda.io/projects/conda/en/latest/) or create a python virtual envrionment ([Why?](https://medium.freecodecamp.org/why-you-need-python-environments-and-how-to-manage-them-with-conda-85f155f4353c)). I recommend conda for Windows users.
2. Perform `CMake` build as described above
3. `cd cmake-build && cmake --build . --target python-package --config Release` 
4. `cd lib/python_package &&  pip install -e .` 

If you want to run the examples then you need to install the following (from main directory):

1. `pip install -r requirements-dev.txt` 


## Build with S2

To build with S2 you must apply this patch first.

```
diff --git a/CMakeLists.txt b/CMakeLists.txt
index 5ecd280..d67bf76 100644
--- a/CMakeLists.txt
+++ b/CMakeLists.txt
@@ -411,7 +411,7 @@ install(TARGETS s2 s2testing DESTINATION lib)

 message("GTEST_ROOT: ${GTEST_ROOT}")
 if (GTEST_ROOT)
-  add_subdirectory(${GTEST_ROOT} build_gtest)
+#   add_subdirectory(${GTEST_ROOT} build_gtest)
   include_directories(${GTEST_ROOT}/include)

   set(S2TestFiles
```

Then enable the option for CMake.


## Profiling

1. `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libprofiler.so CPUPROFILE=prof.prof CPUPROFILE_FREQUENCY=1000`
2. `google-pprof --cum --web ./cmake-build/bin/example-kd prof.prof`
