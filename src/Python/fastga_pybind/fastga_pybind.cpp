
#include "fastga_pybind/fastga_pybind.hpp"
#include "fastga_pybind/fastga_glue.hpp"

// Makes a copy
template <typename T, int dim>
std::vector<std::array<T, dim>> py_array_to_vectors(
        py::array_t<double, py::array::c_style | py::array::forcecast> array) {
    // return std::vector<std::array<T, dim>>();
    if (array.ndim() != 2 || array.shape(1) != dim) {
        throw py::cast_error();
    }
    std::vector<std::array<T, dim>> vectors_T(array.shape(0));
    auto array_unchecked = array.mutable_unchecked<2>();
    for (auto i = 0; i < array_unchecked.shape(0); ++i) {
        for (auto j = 0; j < dim; j++)
        {
            vectors_T[i][j] = array_unchecked(i, j);
        }
    }
    return vectors_T;
}

PYBIND11_MODULE(fastga, m)
{
    m.doc() = "Python binding of FastGA";

    py::bind_vector<std::vector<std::size_t>>(m, "VectorULongInt", py::buffer_protocol());
    py::bind_vector<std::vector<uint8_t>>(m, "VectorUInt8", py::buffer_protocol());
    py::bind_vector<std::vector<double>>(m, "VectorDouble", py::buffer_protocol());
    py::bind_vector<std::vector<int>>(m, "VectorInt", py::buffer_protocol());


    py::class_<FastGA::MatX3d>(m, "MatX3d", py::buffer_protocol())
        // .def(py::init([](py::array_t<double, py::array::c_style> my_array) {return FastGA::MatX3d();} ))
        .def(py::init(&py_array_to_vectors<double, 3>))
        .def_buffer([](FastGA::MatX3d& m) -> py::buffer_info {
            return py::buffer_info(
                m.data(),                                /* Pointer to buffer */
                sizeof(double),                          /* Size of one scalar */
                py::format_descriptor<double>::format(), /* Python struct-style format descriptor */
                2L,                                      /* Number of dimensions */
                {m.size(), 3UL},                         /* Buffer dimensions */
                {sizeof(double) * 3,                     /* Strides (in bytes) for each index */
                 sizeof(double)});
        })
        .def("__copy__", [](FastGA::MatX3d &v) {
            return FastGA::MatX3d(v);
        })
        .def("__deepcopy__", [](FastGA::MatX3d &v, py::dict &memo) {
            return FastGA::MatX3d(v);
        });

    py::class_<FastGA::MatX3I>(m, "MatX3I", py::buffer_protocol())
        .def_buffer([](FastGA::MatX3I& m) -> py::buffer_info {
            return py::buffer_info(
                m.data(),                                /* Pointer to buffer */
                sizeof(size_t),                          /* Size of one scalar */
                py::format_descriptor<size_t>::format(), /* Python struct-style format descriptor */
                2L,                                      /* Number of dimensions */
                {m.size(), 3UL},                         /* Buffer dimensions */
                {sizeof(size_t) * 3,                     /* Strides (in bytes) for each index */
                 sizeof(size_t)});
        });

    py::class_<FastGA::MatX2d>(m, "MatX2d", py::buffer_protocol())
        .def_buffer([](FastGA::MatX2d& m) -> py::buffer_info {
            return py::buffer_info(
                m.data(),                                /* Pointer to buffer */
                sizeof(double),                          /* Size of one scalar */
                py::format_descriptor<double>::format(), /* Python struct-style format descriptor */
                2L,                                      /* Number of dimensions */
                {m.size(), 2UL},                         /* Buffer dimensions */
                {sizeof(double) * 2,                     /* Strides (in bytes) for each index */
                 sizeof(double)});
        });

    // Classes

    py::class_<FastGA::Bucket<uint32_t>>(m, "BucketUInt32")
        .def(py::init<>())
        .def_readonly("normal", &FastGA::Bucket<uint32_t>::normal)
        .def_readonly("hilbert_value", &FastGA::Bucket<uint32_t>::hilbert_value)
        .def_readonly("count", &FastGA::Bucket<uint32_t>::count)
        .def("__repr__",
             [](const FastGA::Bucket<uint32_t>& a) {
                 return ("<Bucket Normal: " + FastGA::Helper::ArrayToString<double, 3>(a.normal) + "; HV: " + std::to_string(a.hilbert_value) + "; CNT: " + std::to_string(a.count) +  "'>");
             });

    py::class_<FastGA::Ico::IcoMesh>(m, "IcoMesh")
        .def(py::init<>())
        .def("__repr__",
             [](const FastGA::Ico::IcoMesh& a) {
                 return "<FastGA::Ico::IcoMesh; # Triangles: '" + std::to_string(a.triangles.size()) + "'>";
             })
        .def_readonly("triangles", &FastGA::Ico::IcoMesh::triangles)
        .def_readonly("vertices", &FastGA::Ico::IcoMesh::vertices)
        .def_readonly("triangle_normals", &FastGA::Ico::IcoMesh::triangle_normals);

    py::class_<FastGA::GaussianAccumulator<uint32_t>>(m, "GaussianAccumulator")
        .def(py::init<const int, const double>(), "level"_a=FastGA_LEVEL, "max_phi"_a=FastGA_MAX_PHI)
        .def("__repr__",
             [](const FastGA::GaussianAccumulator<uint32_t>& a) {
                 return "<FastGA::GA; # Triangles: '" + std::to_string(a.mesh.triangles.size()) + "'>";
             })
        .def_readonly("mesh", &FastGA::GaussianAccumulator<uint32_t>::mesh)
        .def_readonly("buckets", &FastGA::GaussianAccumulator<uint32_t>::buckets)
        .def_readonly("mask", &FastGA::GaussianAccumulator<uint32_t>::mask)
        .def_readonly("num_buckets", &FastGA::GaussianAccumulator<uint32_t>::num_buckets)
        .def("get_bucket_normals", &FastGA::GaussianAccumulator<uint32_t>::GetBucketNormals)
        .def("get_normalized_bucket_counts", &FastGA::GaussianAccumulator<uint32_t>::GetNormalizedBucketCounts)
        .def("get_bucket_indices", &FastGA::GaussianAccumulator<uint32_t>::GetBucketIndices)
        .def("get_bucket_projection", &FastGA::GaussianAccumulator<uint32_t>::GetBucketProjection)
        .def("clear_count", &FastGA::GaussianAccumulator<uint32_t>::ClearCount);

    py::class_<FastGA::GaussianAccumulatorKD,FastGA::GaussianAccumulator<uint32_t>>(m, "GaussianAccumulatorKD")
        .def(py::init<const int, const double, const size_t>(), "level"_a=FastGA_LEVEL, "max_phi"_a=FastGA_MAX_PHI, "max_leaf_size"_a=FastGA_MAX_LEAF_SIZE)
        .def("integrate", &FastGA::GaussianAccumulatorKD::Integrate, "normals"_a, "eps"_a=FastGA_KDTREE_EPS)
        .def("__repr__",
             [](const FastGA::GaussianAccumulatorKD& a) {
                 return "<FastGA::GAKD; # Triangles: '" + std::to_string(a.mesh.triangles.size()) + "'>";
             });

    py::class_<FastGA::GaussianAccumulatorOpt,FastGA::GaussianAccumulator<uint32_t>>(m, "GaussianAccumulatorOpt")
        .def(py::init<const int, const double>(), "level"_a=FastGA_LEVEL, "max_phi"_a=FastGA_MAX_PHI)
        .def_readonly("bucket_neighbors", &FastGA::GaussianAccumulatorOpt::bucket_neighbors)
        .def("integrate", &FastGA::GaussianAccumulatorOpt::Integrate, "normals"_a, "exhaustive"_a=FastGA_EXHAUSTIVE)
        .def("__repr__",
             [](const FastGA::GaussianAccumulatorOpt& a) {
                 return "<FastGA::GAOPT; # Triangles: '" + std::to_string(a.mesh.triangles.size()) + "'>";
             });

    // Functions
    // m.def("convert_normals_to_hilbert", &convert_normals_to_hilbert, "normals"_a);
#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}