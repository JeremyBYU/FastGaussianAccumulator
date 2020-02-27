

#include "pybind11/pybind11.h" // Pybind11 import to define Python bindings
#include "pybind11/stl.h"      // Pybind11 import for STL containers
#include <pybind11/stl_bind.h> // Pybind11 stl bindings
#include "pybind11/numpy.h"

#include "FastGA.hpp"
namespace py = pybind11;

std::tuple<std::vector<double>, std::vector<uint32_t>> convert_normals_to_hilbert(py::array_t<double> nparray)
{
    // This function allows us to convert keyword arguments into a configuration struct
    auto info = nparray.request();
    std::vector<size_t> shape({(size_t)info.shape[0], (size_t)info.shape[1]});
    auto data_ptr = (double*)info.ptr;
    // std::cout << "Size: (" << shape[0] << ", " << shape[1] << "); Ptr: " << data_ptr << std::endl;
    std::vector<double> normals(data_ptr, data_ptr + shape[0] * shape[1]);
    return FastGA::ConvertNormalsToHilbert(normals);
}
