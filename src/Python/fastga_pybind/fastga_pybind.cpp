
#include "fastga_pybind/fastga_pybind.hpp"
#include "fastga_pybind/fastga_glue.hpp"

PYBIND11_MODULE(fastga, m) {
    m.doc() = "Python binding of FastGA";

    py::bind_vector<std::vector<std::size_t>>(m, "VectorLongInt", py::buffer_protocol());
    py::bind_vector<std::vector<double>>(m, "VectorDouble", py::buffer_protocol());
    py::bind_vector<std::vector<int>>(m, "VectorInt", py::buffer_protocol());

    py::class_<FastGA::Ico::Vertices>(m, "VerticesDouble", py::buffer_protocol())
    .def_buffer([](FastGA::Ico::Vertices &m) -> py::buffer_info {
        return py::buffer_info(
            m.data(),                                   /* Pointer to buffer */
            sizeof(double),                          /* Size of one scalar */
            py::format_descriptor<double>::format(), /* Python struct-style format descriptor */
            2L,                                       /* Number of dimensions */
            {m.size(), 3UL},                        /* Buffer dimensions */
            {sizeof(double) * 3,                /* Strides (in bytes) for each index */
                sizeof(double)});
    });
    py::class_<FastGA::Ico::Triangles>(m, "TrianglesUInt", py::buffer_protocol())
    .def_buffer([](FastGA::Ico::Triangles &m) -> py::buffer_info {
        return py::buffer_info(
            m.data(),                                   /* Pointer to buffer */
            sizeof(size_t),                          /* Size of one scalar */
            py::format_descriptor<size_t>::format(), /* Python struct-style format descriptor */
            2L,                                       /* Number of dimensions */
            {m.size(), 3UL},                        /* Buffer dimensions */
            {sizeof(size_t) * 3,                /* Strides (in bytes) for each index */
                sizeof(size_t)});
    });

    py::class_<FastGA::Ico::IcoMesh>(m, "IcoMesh")
        .def(py::init<>())
        .def("__repr__",
            [](const FastGA::Ico::IcoMesh &a) {
                return "<FastGA::Ico::IcoMesh; # Triangles: '" + std::to_string(a.triangles.size()) + "'>";
            }
        )
        .def_readonly("triangles", &FastGA::Ico::IcoMesh::triangles)
        .def_readonly("vertices", &FastGA::Ico::IcoMesh::vertices);

    py::class_<FastGA::GaussianAccumulator>(m, "GaussianAccumulator")
        .def(py::init<const int>())
        .def("__repr__",
            [](const FastGA::GaussianAccumulator &a) {
                return "<FastGA::GA; # Triangles: '" + std::to_string(a.mesh.triangles.size()) + "'>";
            }
        )
        .def_readonly("mesh", &FastGA::GaussianAccumulator::mesh);

    m.def("convert_normals_to_hilbert", &convert_normals_to_hilbert, "normals"_a);
#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}