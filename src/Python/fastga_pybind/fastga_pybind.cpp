
#include "fastga_pybind/fastga_pybind.hpp"
#include "fastga_pybind/docstring/docstring.hpp"

using namespace FastGA;
// Makes a copy
template <typename T, int dim>
std::vector<std::array<T, dim>> py_array_to_vectors(
    py::array_t<double, py::array::c_style | py::array::forcecast> array)
{
    // return std::vector<std::array<T, dim>>();
    if (array.ndim() != 2 || array.shape(1) != dim)
    {
        throw py::cast_error();
    }
    std::vector<std::array<T, dim>> vectors_T(array.shape(0));
    auto array_unchecked = array.mutable_unchecked<2>();
    for (auto i = 0; i < array_unchecked.shape(0); ++i)
    {
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
            const size_t cols = 3;
            return py::buffer_info(
                m.data(),                                /* Pointer to buffer */
                sizeof(double),                          /* Size of one scalar */
                py::format_descriptor<double>::format(), /* Python struct-style format descriptor */
                2UL,                                     /* Number of dimensions */
                {m.size(), cols},                        /* Buffer dimensions */
                {sizeof(double) * cols,                  /* Strides (in bytes) for each index */
                 sizeof(double)});
        })
        .def("__copy__", [](FastGA::MatX3d& v) {
            return FastGA::MatX3d(v);
        })
        .def("__deepcopy__", [](FastGA::MatX3d& v, py::dict& memo) {
            return FastGA::MatX3d(v);
        });

    py::class_<FastGA::MatX3I>(m, "MatX3I", py::buffer_protocol())
        .def_buffer([](FastGA::MatX3I& m) -> py::buffer_info {
            const size_t cols = 3;
            return py::buffer_info(
                m.data(),                                /* Pointer to buffer */
                sizeof(size_t),                          /* Size of one scalar */
                py::format_descriptor<size_t>::format(), /* Python struct-style format descriptor */
                2UL,                                     /* Number of dimensions */
                {m.size(), cols},                        /* Buffer dimensions */
                {sizeof(size_t) * cols,                  /* Strides (in bytes) for each index */
                 sizeof(size_t)});
        });
    py::class_<FastGA::MatX2I>(m, "MatX2I", py::buffer_protocol())
        .def_buffer([](FastGA::MatX2I& m) -> py::buffer_info {
            const size_t cols = 2;
            return py::buffer_info(
                m.data(),                                /* Pointer to buffer */
                sizeof(size_t),                          /* Size of one scalar */
                py::format_descriptor<size_t>::format(), /* Python struct-style format descriptor */
                2UL,                                     /* Number of dimensions */
                {m.size(), cols},                        /* Buffer dimensions */
                {sizeof(size_t) * cols,                  /* Strides (in bytes) for each index */
                 sizeof(size_t)});
        });

    py::class_<FastGA::MatX2d>(m, "MatX2d", py::buffer_protocol())
        .def_buffer([](FastGA::MatX2d& m) -> py::buffer_info {
            const size_t cols = 3;
            return py::buffer_info(
                m.data(),                                /* Pointer to buffer */
                sizeof(double),                          /* Size of one scalar */
                py::format_descriptor<double>::format(), /* Python struct-style format descriptor */
                2UL,                                     /* Number of dimensions */
                {m.size(), cols},                        /* Buffer dimensions */
                {sizeof(double) * cols,                  /* Strides (in bytes) for each index */
                 sizeof(double)});
        });

    // Classes
    py::class_<FastGA::Ico::Image>(m, "Image", py::buffer_protocol())
        .def_buffer([](FastGA::Ico::Image& m) -> py::buffer_info {
            const size_t cols = m.cols_;
            const size_t rows = m.rows_;
            std::string format;
            switch (m.bytes_per_channel_)
            {
            case 1:
                format = py::format_descriptor<uint8_t>::format();
                break;
            case 2:
                format = py::format_descriptor<uint16_t>::format();
                break;
            case 4:
                if (m.is_float_)
                {
                    format = py::format_descriptor<float>::format();
                }
                else
                {
                    format = py::format_descriptor<int>::format();
                }
                break;
            default:
                throw std::runtime_error(
                    "Image has unrecognized bytes_per_channel.");
                break;
            }
            return py::buffer_info(
                m.buffer_.data(),             /* Pointer to buffer */
                m.bytes_per_channel_,         /* Size of one scalar */
                format,                       /* Python struct-style format descriptor */
                2UL,                          /* Number of dimensions */
                {rows, cols},                 /* Buffer dimensions */
                {m.bytes_per_channel_ * cols, /* Strides (in bytes) for each index */
                 static_cast<size_t>(m.bytes_per_channel_)});
        })
        .def("__repr__",
             [](const FastGA::Ico::Image& img) { return std::string("Image of size ") +
                                                        std::to_string(img.cols_) + std::string("x") +
                                                        std::to_string(img.rows_); });

    py::class_<FastGA::Bucket<uint32_t>>(m, "BucketUInt32")
        .def(py::init<>())
        .def_readonly("normal", &FastGA::Bucket<uint32_t>::normal)
        .def_readonly("hilbert_value", &FastGA::Bucket<uint32_t>::hilbert_value)
        .def_readonly("count", &FastGA::Bucket<uint32_t>::count)
        .def("__repr__",
             [](const FastGA::Bucket<uint32_t>& a) {
                 return ("<Bucket Normal: " + FastGA::Helper::ArrayToString<double, 3>(a.normal) + "; HV: " + std::to_string(a.hilbert_value) + "; CNT: " + std::to_string(a.count) + "'>");
             });

    py::class_<FastGA::Helper::BBOX>(m, "BBOX", "Contains extents for a projection")
        .def(py::init<>())
        .def("__repr__",
             [](const FastGA::Bucket<uint32_t>& a) {
                 return ("<BBOX>");
             });

    py::class_<FastGA::Ico::IcoMesh>(m, "IcoMesh", "A Triangle Mesh of Icosphere")
        .def(py::init<>())
        .def("__repr__",
             [](const FastGA::Ico::IcoMesh& a) {
                 return "<FastGA::Ico::IcoMesh; # Triangles: '" + std::to_string(a.triangles.size()) + "'>";
             })
        .def_readonly("triangles", &FastGA::Ico::IcoMesh::triangles)
        .def_readonly("vertices", &FastGA::Ico::IcoMesh::vertices)
        .def_readonly("triangle_normals", &FastGA::Ico::IcoMesh::triangle_normals)
        .def_readonly("adjacency_matrix", &FastGA::Ico::IcoMesh::adjacency_matrix);

    py::class_<FastGA::GaussianAccumulator<uint32_t>>(m, "GaussianAccumulatorUI32")
        .def(py::init<const int, const double>(), "level"_a = FastGA_LEVEL, "max_phi"_a = FastGA_MAX_PHI)
        .def("__repr__",
             [](const FastGA::GaussianAccumulator<uint32_t>& a) {
                 return "<FastGA::GA; # Triangles: '" + std::to_string(a.mesh.triangles.size()) + "'>";
             })
        .def_readonly("mesh", &FastGA::GaussianAccumulator<uint32_t>::mesh)
        .def_readonly("buckets", &FastGA::GaussianAccumulator<uint32_t>::buckets)
        .def_readonly("sort_idx", &FastGA::GaussianAccumulator<uint32_t>::sort_idx)
        .def_readonly("mask", &FastGA::GaussianAccumulator<uint32_t>::mask)
        .def_readonly("num_buckets", &FastGA::GaussianAccumulator<uint32_t>::num_buckets)
        .def_readonly("projected_bbox", &FastGA::GaussianAccumulator<uint32_t>::projected_bbox)
        .def("get_bucket_normals", &FastGA::GaussianAccumulator<uint32_t>::GetBucketNormals, "reverse_sort"_a = false)
        .def("get_normalized_bucket_counts", &FastGA::GaussianAccumulator<uint32_t>::GetNormalizedBucketCounts, "reverse_sort"_a = false)
        .def("get_normalized_bucket_counts_by_vertex", &FastGA::GaussianAccumulator<uint32_t>::GetNormalizedBucketCountsByVertex, "reverse_sort"_a = false)
        .def("get_bucket_indices", &FastGA::GaussianAccumulator<uint32_t>::GetBucketIndices)
        .def("get_bucket_projection", &FastGA::GaussianAccumulator<uint32_t>::GetBucketProjection)
        .def("clear_count", &FastGA::GaussianAccumulator<uint32_t>::ClearCount)
        .def("copy_ico_mesh", &FastGA::GaussianAccumulator<uint32_t>::CopyIcoMesh, "reverse_sort"_a = false);

    py::class_<FastGA::GaussianAccumulator<uint64_t>>(m, "GaussianAccumulatorUI64")
        .def(py::init<const int, const double>(), "level"_a = FastGA_LEVEL, "max_phi"_a = FastGA_MAX_PHI)
        .def("__repr__",
             [](const FastGA::GaussianAccumulator<uint64_t>& a) {
                 return "<FastGA::GA; # Triangles: '" + std::to_string(a.mesh.triangles.size()) + "'>";
             })
        .def_readonly("mesh", &FastGA::GaussianAccumulator<uint64_t>::mesh)
        .def_readonly("buckets", &FastGA::GaussianAccumulator<uint64_t>::buckets)
        .def_readonly("sort_idx", &FastGA::GaussianAccumulator<uint64_t>::sort_idx)
        .def_readonly("mask", &FastGA::GaussianAccumulator<uint64_t>::mask)
        .def_readonly("num_buckets", &FastGA::GaussianAccumulator<uint64_t>::num_buckets)
        .def_readonly("projected_bbox", &FastGA::GaussianAccumulator<uint64_t>::projected_bbox)
        .def("get_bucket_normals", &FastGA::GaussianAccumulator<uint64_t>::GetBucketNormals, "reverse_sort"_a = false)
        .def("get_normalized_bucket_counts", &FastGA::GaussianAccumulator<uint64_t>::GetNormalizedBucketCounts, "reverse_sort"_a = false)
        .def("get_normalized_bucket_counts_by_vertex", &FastGA::GaussianAccumulator<uint64_t>::GetNormalizedBucketCountsByVertex, "reverse_sort"_a = false)
        .def("get_bucket_indices", &FastGA::GaussianAccumulator<uint64_t>::GetBucketIndices)
        .def("get_bucket_projection", &FastGA::GaussianAccumulator<uint64_t>::GetBucketProjection)
        .def("clear_count", &FastGA::GaussianAccumulator<uint64_t>::ClearCount)
        .def("copy_ico_mesh", &FastGA::GaussianAccumulator<uint64_t>::CopyIcoMesh, "reverse_sort"_a = false);

    py::class_<FastGA::GaussianAccumulatorKD, FastGA::GaussianAccumulator<uint32_t>>(m, "GaussianAccumulatorKD", "A Fast Gaussian Accumulator. Works on Full Sphere using KD Trees")
        .def(py::init<const int, const double, const size_t>(), "level"_a = FastGA_LEVEL, "max_phi"_a = FastGA_MAX_PHI, "max_leaf_size"_a = FastGA_MAX_LEAF_SIZE, "Will intergrate the normals into the S2 Historgram")
        .def("integrate", &FastGA::GaussianAccumulatorKD::Integrate, "normals"_a, "eps"_a = FastGA_KDTREE_EPS)
        .def("__repr__",
             [](const FastGA::GaussianAccumulatorKD& a) {
                 return "<FastGA::GAKD; # Triangles: '" + std::to_string(a.mesh.triangles.size()) + "'>";
             });

    py::class_<FastGA::GaussianAccumulatorOpt, FastGA::GaussianAccumulator<uint32_t>>(m, "GaussianAccumulatorOpt", "A Fast Gaussian Accumulator on S2. Only works on top hemisphere.")
        .def(py::init<const int, const double>(), "level"_a = FastGA_LEVEL, "max_phi"_a = FastGA_MAX_PHI)
        .def_readonly("bucket_neighbors", &FastGA::GaussianAccumulatorOpt::bucket_neighbors)
        .def("integrate", &FastGA::GaussianAccumulatorOpt::Integrate, "normals"_a, "num_nbr"_a = FastGA_TRI_NBRS, "Will intergrate the normals into the S2 Historgram")
        .def("__repr__",
             [](const FastGA::GaussianAccumulatorOpt& a) {
                 return "<FastGA::GAOPT; # Triangles: '" + std::to_string(a.mesh.triangles.size()) + "'>";
             });

    py::class_<FastGA::GaussianAccumulatorS2, FastGA::GaussianAccumulator<uint64_t>>(m, "GaussianAccumulatorS2", "A Fast Gaussian Accumulator on S2. Works on full sphere.")
        .def(py::init<const int, const double>(), "level"_a = FastGA_LEVEL, "max_phi"_a = FastGA_MAX_PHI)
        .def_readonly("bucket_neighbors", &FastGA::GaussianAccumulatorS2::bucket_neighbors, "Fast lookup matrix to find neighbors of a bucket")
        .def("integrate", &FastGA::GaussianAccumulatorS2::Integrate, "normals"_a, "num_nbr"_a = FastGA_TRI_NBRS, "Will intergrate the normals into the S2 Historgram")
        .def("__repr__",
             [](const FastGA::GaussianAccumulatorS2& a) {
                 return "<FastGA::GAS2; # Triangles: '" + std::to_string(a.mesh.triangles.size()) + "'>";
             });

    py::class_<FastGA::Ico::IcoCharts>(m, "IcoCharts", "Contains charts of an unwrapped Icosahedron")
        .def(py::init<const int>(), "level"_a = FastGA_LEVEL)
        // .def_readonly("point_idx_to_image_idx", &FastGA::Ico::IcoChart::point_idx_to_image_idx)
        // .def_readonly("local_to_global_point_idx_map", &FastGA::Ico::IcoChart::local_to_global_point_idx_map)
        .def_readonly("image", &FastGA::Ico::IcoCharts::image, "Returns an unwrapped image of the IcoCharts")
        .def_readonly("image_to_vertex_idx", &FastGA::Ico::IcoCharts::image_to_vertex_idx, "Fast lookup matrix for image creation. Each pixel hold the icosahedron vertex index it corresponds to")
        .def_readonly("mask", &FastGA::Ico::IcoCharts::mask, "Boolean mask corresponding to valid cells. False(0) corresponds to ghost/halo cells")
        .def_readonly("sphere_mesh", &FastGA::Ico::IcoCharts::sphere_mesh, "The full icosahedron the IcoChart is unwrapping")
        .def("fill_image", &FastGA::Ico::IcoCharts::FillImage, "normalized_vertex_count"_a, "Fills the the image using the normalized vertex counts")
        .def("__repr__",
             [](const FastGA::Ico::IcoCharts& a) {
                 return "<IcoChart; Level: '" + std::to_string(a.level) + "'>";
             });

    // Functions
    m.def("convert_normals_to_hilbert", &FastGA::Helper::ConvertNormalsToHilbert, "normals"_a, "bbox"_a,
          "Not recommended. Converts a numpy array of normals to uint32 Hilbert Values"
          "Assumes EqualArea Azimuth Projection centered at north pole. Only good on for northern hemisphere.");
    docstring::FunctionDocInject(m, "convert_normals_to_hilbert", {{"normals", "MatX3d; NX3 Array"}, {"bbox", "BBOX; bounding box of AzimuthProjection projection"}});

    m.def("convert_normals_to_s2id", &FastGA::Helper::ConvertNormalsToS2ID, "normals"_a, "Converts unit normals to uint64 S2 ids");
    docstring::FunctionDocInject(m, "convert_normals_to_s2id", {{"normals", "MatX3d; NX3 Array"}});

    m.def("refine_icosahedron", &FastGA::Ico::RefineIcosahedron, "level"_a, "Creates a refined icosahedron mesh");
    docstring::FunctionDocInject(m, "refine_icosahedron", {{"level", "The level of refinement of the icosahedron. Each level recursively subdived triangles"}});

    m.def("refine_icochart", &FastGA::Ico::RefineIcoChart, "level"_a = 0, "square"_a = false, "Return an refined icochart");


#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}