
#include "fastga_pybind/fastga_pybind.hpp"
#include "fastga_pybind/docstring/docstring.hpp"

using namespace FastGA;
// Makes a copy
template <typename T, int dim>
std::vector<std::array<T, dim>>
py_array_to_vectors(py::array_t<double, py::array::c_style | py::array::forcecast> array)
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

    py::bind_vector<std::vector<std::size_t>>(
        m, "VectorULongInt", py::buffer_protocol(),
        "Contiguous buffer of Uint64. Use np.asarray() to get to get numpy array.");
    py::bind_vector<std::vector<uint8_t>>(m, "VectorUInt8", py::buffer_protocol(),
                                          "Contiguous buffer of Uint8. Use np.asarray() to get to get numpy array.");
    py::bind_vector<std::vector<double>>(m, "VectorDouble", py::buffer_protocol(),
                                         "Contiguous buffer of Float64. Use np.asarray() to get to get numpy array.");
    py::bind_vector<std::vector<int>>(m, "VectorInt", py::buffer_protocol(),
                                      "Contiguous buffer of Int32. Use np.asarray() to get to get numpy array.");

    py::class_<FastGA::MatX3d>(
        m, "MatX3d", py::buffer_protocol(),
        "NX3 Matrix (Double) representation of numpy array. Use np.asarray() to get numpy array.")
        // .def(py::init([](py::array_t<double, py::array::c_style> my_array) {return FastGA::MatX3d();} ))
        .def(py::init(&py_array_to_vectors<double, 3>))
        .def_buffer([](FastGA::MatX3d& m) -> py::buffer_info {
            const size_t cols = 3;
            return py::buffer_info(m.data(),                                /* Pointer to buffer */
                                   sizeof(double),                          /* Size of one scalar */
                                   py::format_descriptor<double>::format(), /* Python struct-style format descriptor */
                                   2UL,                                     /* Number of dimensions */
                                   {m.size(), cols},                        /* Buffer dimensions */
                                   {sizeof(double) * cols,                  /* Strides (in bytes) for each index */
                                    sizeof(double)});
        })
        .def("__copy__", [](FastGA::MatX3d& v) { return FastGA::MatX3d(v); })
        .def("__deepcopy__", [](FastGA::MatX3d& v, py::dict& memo) { return FastGA::MatX3d(v); });

    py::class_<FastGA::MatX3I>(
        m, "MatX3I", py::buffer_protocol(),
        "NX3 Matrix (Uint64) representation of numpy array. Use np.asarray() to get numpy array.")
        .def_buffer([](FastGA::MatX3I& m) -> py::buffer_info {
            const size_t cols = 3;
            return py::buffer_info(m.data(),                                /* Pointer to buffer */
                                   sizeof(size_t),                          /* Size of one scalar */
                                   py::format_descriptor<size_t>::format(), /* Python struct-style format descriptor */
                                   2UL,                                     /* Number of dimensions */
                                   {m.size(), cols},                        /* Buffer dimensions */
                                   {sizeof(size_t) * cols,                  /* Strides (in bytes) for each index */
                                    sizeof(size_t)});
        });

    py::class_<FastGA::MatX6I>(
        m, "MatX6I", py::buffer_protocol(),
        "NX6 Matrix (Uint64) representation of numpy array. Use np.asarray() to get numpy array.")
        .def_buffer([](FastGA::MatX6I& m) -> py::buffer_info {
            const size_t cols = 6;
            return py::buffer_info(m.data(),                                /* Pointer to buffer */
                                   sizeof(size_t),                          /* Size of one scalar */
                                   py::format_descriptor<size_t>::format(), /* Python struct-style format descriptor */
                                   2UL,                                     /* Number of dimensions */
                                   {m.size(), cols},                        /* Buffer dimensions */
                                   {sizeof(size_t) * cols,                  /* Strides (in bytes) for each index */
                                    sizeof(size_t)});
        });
        
    py::class_<FastGA::MatX2I>(
        m, "MatX2I", py::buffer_protocol(),
        "NX2 Matrix (Uint64) representation of numpy array. Use np.asarray() to get numpy array.")
        .def_buffer([](FastGA::MatX2I& m) -> py::buffer_info {
            const size_t cols = 2;
            return py::buffer_info(m.data(),                                /* Pointer to buffer */
                                   sizeof(size_t),                          /* Size of one scalar */
                                   py::format_descriptor<size_t>::format(), /* Python struct-style format descriptor */
                                   2UL,                                     /* Number of dimensions */
                                   {m.size(), cols},                        /* Buffer dimensions */
                                   {sizeof(size_t) * cols,                  /* Strides (in bytes) for each index */
                                    sizeof(size_t)});
        });

    py::class_<FastGA::MatX12I>(
        m, "MatX12I", py::buffer_protocol(),
        "NX12 Matrix (Uint64) representation of numpy array. Use np.asarray() to get numpy array.")
        .def_buffer([](FastGA::MatX12I& m) -> py::buffer_info {
            const size_t cols = 12;
            return py::buffer_info(m.data(),                                /* Pointer to buffer */
                                   sizeof(size_t),                          /* Size of one scalar */
                                   py::format_descriptor<size_t>::format(), /* Python struct-style format descriptor */
                                   2UL,                                     /* Number of dimensions */
                                   {m.size(), cols},                        /* Buffer dimensions */
                                   {sizeof(size_t) * cols,                  /* Strides (in bytes) for each index */
                                    sizeof(size_t)});
        });

    py::class_<FastGA::MatX2d>(
        m, "MatX2d", py::buffer_protocol(),
        "NX2 Matrix (Double) representation of numpy array. Use np.asarray() to get numpy array.")
        .def_buffer([](FastGA::MatX2d& m) -> py::buffer_info {
            const size_t cols = 3;
            return py::buffer_info(m.data(),                                /* Pointer to buffer */
                                   sizeof(double),                          /* Size of one scalar */
                                   py::format_descriptor<double>::format(), /* Python struct-style format descriptor */
                                   2UL,                                     /* Number of dimensions */
                                   {m.size(), cols},                        /* Buffer dimensions */
                                   {sizeof(double) * cols,                  /* Strides (in bytes) for each index */
                                    sizeof(double)});
        });

    // Classes
    py::class_<FastGA::Ico::Image>(m, "Image", py::buffer_protocol(),
                                   "An image class. In this library it stores unwrapped values of the icosahedron as a "
                                   "2D image. Use np.asarray() to get numpy array.")
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
                throw std::runtime_error("Image has unrecognized bytes_per_channel.");
                break;
            }
            return py::buffer_info(m.buffer_.data(),             /* Pointer to buffer */
                                   m.bytes_per_channel_,         /* Size of one scalar */
                                   format,                       /* Python struct-style format descriptor */
                                   2UL,                          /* Number of dimensions */
                                   {rows, cols},                 /* Buffer dimensions */
                                   {m.bytes_per_channel_ * cols, /* Strides (in bytes) for each index */
                                    static_cast<size_t>(m.bytes_per_channel_)});
        })
        .def("__repr__", [](const FastGA::Ico::Image& img) {
            return std::string("Image of size ") + std::to_string(img.cols_) + std::string("x") +
                   std::to_string(img.rows_);
        });

    py::class_<FastGA::Bucket<uint32_t>>(m, "BucketUInt32",
                                         "The bucket class describes a cell on the S2 Histogram. It unfortunately has "
                                         "one member that should not really exist (projection).")
        .def(py::init<>())
        .def_readonly("normal", &FastGA::Bucket<uint32_t>::normal, "The surface **unit** normal of the triangle cell")
        .def_readonly("hilbert_value", &FastGA::Bucket<uint32_t>::hilbert_value,
                      "Space Filling Curve value, may be Int32 or Uint64")
        .def_readonly("count", &FastGA::Bucket<uint32_t>::count, "Count variable for histogram")
        .def("__repr__", [](const FastGA::Bucket<uint32_t>& a) {
            return ("<Bucket Normal: " + FastGA::Helper::ArrayToString<double, 3>(a.normal) +
                    "; HV: " + std::to_string(a.hilbert_value) + "; CNT: " + std::to_string(a.count) + "'>");
        });

    py::class_<FastGA::Bucket<uint64_t>>(m, "BucketUInt64",
                                         "The bucket class describes a cell on the S2 Histogram. It unfortunately has "
                                         "one member that should not really exist (projection).")
        .def(py::init<>())
        .def_readonly("normal", &FastGA::Bucket<uint64_t>::normal, "The surface **unit** normal of the triangle cell")
        .def_readonly("hilbert_value", &FastGA::Bucket<uint64_t>::hilbert_value,
                      "Space Filling Curve value, may be Int32 or Uint64")
        .def_readonly("count", &FastGA::Bucket<uint64_t>::count, "Count variable for histogram")
        .def("__repr__", [](const FastGA::Bucket<uint64_t>& a) {
            return ("<Bucket Normal: " + FastGA::Helper::ArrayToString<double, 3>(a.normal) +
                    "; HV: " + std::to_string(a.hilbert_value) + "; CNT: " + std::to_string(a.count) + "'>");
        });

    py::class_<FastGA::Helper::BBOX>(m, "BBOX", "Contains extents for a projection")
        .def(py::init<>())
        .def("__repr__", [](const FastGA::Bucket<uint32_t>& a) { return ("<BBOX>"); });

    py::class_<FastGA::Ico::IcoMesh>(
        m, "IcoMesh",
        "This stores the mesh of a refined icosahedron. Note that the ordering of the vertices and triangles are very "
        "particular. This ordering must be maintained for use during an unwrapping procedure.")
        .def(py::init<>())
        .def("__repr__",
             [](const FastGA::Ico::IcoMesh& a) {
                 return "<FastGA::Ico::IcoMesh; # Triangles: '" + std::to_string(a.triangles.size()) + "'>";
             })
        .def_readonly("triangles", &FastGA::Ico::IcoMesh::triangles, "The vetrices in the mesh, NX3")
        .def_readonly("vertices", &FastGA::Ico::IcoMesh::vertices, "The triangles in the mesh, KX3")
        .def_readonly("triangle_normals", &FastGA::Ico::IcoMesh::triangle_normals,
                      "The triangle normals in the mesh, KX3")
        .def_readonly("adjacency_matrix", &FastGA::Ico::IcoMesh::adjacency_matrix, "The vertex adjacency matrix, NX6");

    py::class_<FastGA::GaussianAccumulator<uint32_t>>(
        m, "GaussianAccumulatorUI32",
        "This is the base class of the Gaussian Accumulator. GaussianAccumulatorKD, GaussianAccumulatorOpt, and "
        "GaussianAccumulatorS2 will derive from this class. Unfortunately those classes have small differences causing "
        "some unnecessary members in here that basically occurred as these classes were created and changed over time. "
        "Eventually I will rewrite this whole thing such that only the bare essentials are in this class.")
        .def(py::init<const int, const double>(), "level"_a = FASTGA_LEVEL, "max_phi"_a = FASTGA_MAX_PHI)
        .def("__repr__",
             [](const FastGA::GaussianAccumulator<uint32_t>& a) {
                 return "<FastGA::GA; # Triangles: '" + std::to_string(a.mesh.triangles.size()) + "'>";
             })
        .def_readonly("mesh", &FastGA::GaussianAccumulator<uint32_t>::mesh,
                      "The underlying sphere-like mesh of the Gaussian Accumulator")
        .def_readonly("buckets", &FastGA::GaussianAccumulator<uint32_t>::buckets,
                      "The buckets in the histogram, corresponding to cells/triangles on the mesh")
        // .def_readonly("sort_idx", &FastGA::GaussianAccumulator<uint32_t>::sort_idx)
        .def_readonly("mask", &FastGA::GaussianAccumulator<uint32_t>::mask,
                      "A mask which indicates which triangles in the mesh are included in the buckets. By default its "
                      "every one (mask = ones). This was added because I thought a user might want to limit the "
                      "histogram to only include triangles a max_phi from the north pole.")
        .def_readonly("num_buckets", &FastGA::GaussianAccumulator<uint32_t>::num_buckets,
                      "The number of buckets in histogram, size(buckets)")
        .def_readonly("projected_bbox", &FastGA::GaussianAccumulator<uint32_t>::projected_bbox,
                      "Only a valid member for GaussianAccumulatorOpt, ignore for everthing else")
        .def("get_bucket_normals", &FastGA::GaussianAccumulator<uint32_t>::GetBucketNormals,
             "Gets the surface normals of the buckets in the histogram."
             "The order by default is sorted by the space filling curve value attached to each cell.",
             "mesh_order"_a = false)
        .def("get_normalized_bucket_counts", &FastGA::GaussianAccumulator<uint32_t>::GetNormalizedBucketCounts,
             "Get the normalized bucket counts in the histogram."
             "The order by default is sorted by the space filling curve value attached to each cell.",
             "mesh_order"_a = false)
        .def("get_normalized_bucket_counts_by_vertex",
             &FastGA::GaussianAccumulator<uint32_t>::GetNormalizedBucketCountsByVertex,
             "Average the normalized buckets counts (triangles) into the *vertices* of the mesh."
             "The order by default is sorted by the space filling curve value attached to each cell.",
             "mesh_order"_a = false)
        .def("get_bucket_sfc_values", &FastGA::GaussianAccumulator<uint32_t>::GetBucketSFCValues,
             "Get the space filling curve values of each bucket. Will be sorted low to high.")
        .def("get_bucket_projection", &FastGA::GaussianAccumulator<uint32_t>::GetBucketProjection,
             "Only useful for GaussianAccumulatorOpt. Return the XY projection of each bucket. ")
        .def("clear_count", &FastGA::GaussianAccumulator<uint32_t>::ClearCount,
             "Clears all the histogram counts for each cell. Useful to call after peak detection to 'reset' the mesh.")
        .def("copy_ico_mesh", &FastGA::GaussianAccumulator<uint32_t>::CopyIcoMesh, "Creates a copy of the ico mesh.",
             "mesh_order"_a = false);

    py::class_<FastGA::GaussianAccumulator<uint64_t>>(
        m, "GaussianAccumulatorUI64",
        "This is the base class of the Gaussian Accumulator. GaussianAccumulatorKD, GaussianAccumulatorOpt, and "
        "GaussianAccumulatorS2 will derive from this class. Unfortunately those classes have small differences causing "
        "some unnecessary members in here that basically occurred as these classes were created and changed over time. "
        "Eventually I will rewrite this whole thing such that only the bare essentials are in this class.")
        .def(py::init<const int, const double>(), "level"_a = FASTGA_LEVEL, "max_phi"_a = FASTGA_MAX_PHI)
        .def("__repr__",
             [](const FastGA::GaussianAccumulator<uint64_t>& a) {
                 return "<FastGA::GA; # Triangles: '" + std::to_string(a.mesh.triangles.size()) + "'>";
             })
        .def_readonly("mesh", &FastGA::GaussianAccumulator<uint64_t>::mesh,
                      "The underlying sphere-like mesh of the Gaussian Accumulator")
        .def_readonly("buckets", &FastGA::GaussianAccumulator<uint64_t>::buckets,
                      "The buckets in the histogram, corresponding to cells/triangles on the mesh")
        // .def_readonly("sort_idx", &FastGA::GaussianAccumulator<uint64_t>::sort_idx)
        .def_readonly("mask", &FastGA::GaussianAccumulator<uint64_t>::mask,
                      "A mask which indicates which triangles in the mesh are included in the buckets. By default its "
                      "every one (mask = ones). This was added because I thought a user might want to limit the "
                      "histogram to only include triangles a max_phi from the north pole.")
        .def_readonly("num_buckets", &FastGA::GaussianAccumulator<uint64_t>::num_buckets,
                      "The number of buckets in histogram, size(buckets)")
        .def_readonly("projected_bbox", &FastGA::GaussianAccumulator<uint64_t>::projected_bbox,
                      "Only a valid member for GaussianAccumulatorOpt, ignore for everthing else")
        .def("get_bucket_normals", &FastGA::GaussianAccumulator<uint64_t>::GetBucketNormals,
             "Gets the surface normals of the buckets in the histogram."
             "The order by default is sorted by the space filling curve value attached to each cell.",
             "mesh_order"_a = false)
        .def("get_normalized_bucket_counts", &FastGA::GaussianAccumulator<uint64_t>::GetNormalizedBucketCounts,
             "Get the normalized bucket counts in the histogram."
             "The order by default is sorted by the space filling curve value attached to each cell.",
             "mesh_order"_a = false)
        .def("get_normalized_bucket_counts_by_vertex",
             &FastGA::GaussianAccumulator<uint64_t>::GetNormalizedBucketCountsByVertex,
             "Average the normalized buckets counts (triangles) into the *vertices* of the mesh."
             "The order by default is sorted by the space filling curve value attached to each cell.",
             "mesh_order"_a = false)
        .def("get_bucket_sfc_values", &FastGA::GaussianAccumulator<uint64_t>::GetBucketSFCValues,
             "Get the space filling curve values of each bucket. Will be sorted low to high.")
        .def("get_bucket_projection", &FastGA::GaussianAccumulator<uint64_t>::GetBucketProjection,
             "Only useful for GaussianAccumulatorOpt. Return the XY projection of each bucket. ")
        .def("clear_count", &FastGA::GaussianAccumulator<uint64_t>::ClearCount,
             "Clears all the histogram counts for each cell. Useful to call after peak detection to 'reset' the mesh.")
        .def("copy_ico_mesh", &FastGA::GaussianAccumulator<uint64_t>::CopyIcoMesh, "Creates a copy of the ico mesh.",
             "mesh_order"_a = false);

    py::class_<FastGA::GaussianAccumulatorKD, FastGA::GaussianAccumulator<uint32_t>>(
        m, "GaussianAccumulatorKD", "A Fast Gaussian Accumulator. Works on Full Sphere using KD Trees")
        .def(py::init<const int, const double, const size_t>(), "level"_a = FASTGA_LEVEL, "max_phi"_a = FASTGA_MAX_PHI,
             "max_leaf_size"_a = FASTGA_MAX_LEAF_SIZE, "Will intergrate the normals into the S2 Historgram")
        .def("integrate", &FastGA::GaussianAccumulatorKD::Integrate,
             "Will intergrate the unit normals into the S2 Historgram", "normals"_a, "eps"_a = FASTGA_KDTREE_EPS)
        .def("__repr__", [](const FastGA::GaussianAccumulatorKD& a) {
            return "<FastGA::GAKD; # Triangles: '" + std::to_string(a.mesh.triangles.size()) + "'>";
        });

    py::class_<FastGA::GaussianAccumulatorOpt, FastGA::GaussianAccumulator<uint32_t>>(
        m, "GaussianAccumulatorOpt",
        "Construct a new GaussianAccumulatorOpt object. Do **not** use this class. It was my first design and only "
        "works well on the top hemisphere of a sphere. It uses a single projection (Azimuth Equal Area Projection) to "
        "project to a 2D plane. A hilbert curve is performed on the plane to greate the SFC on the sphere. This class "
        "is the reason that the `GaussianAccumulator` base class is such a mess because it began with the assumptions "
        "built into this class. Eventually this will be deprecated.")
        .def(py::init<const int, const double>(), "level"_a = FASTGA_LEVEL, "max_phi"_a = FASTGA_MAX_PHI)
        .def_readonly("bucket_neighbors", &FastGA::GaussianAccumulatorOpt::bucket_neighbors)
        .def("integrate", &FastGA::GaussianAccumulatorOpt::Integrate, "normals"_a, "num_nbr"_a = FASTGA_TRI_NBRS,
             "Will intergrate the normals into the S2 Historgram")
        .def("__repr__", [](const FastGA::GaussianAccumulatorOpt& a) {
            return "<FastGA::GAOPT; # Triangles: '" + std::to_string(a.mesh.triangles.size()) + "'>";
        });

    py::class_<FastGA::GaussianAccumulatorS2, FastGA::GaussianAccumulator<uint64_t>>(
        m, "GaussianAccumulatorS2",
        "This GaussianAccumulator can handle the entire sphere by using a space filling curve designed by Google's S2 "
        "Geometry library. It projects a sphere to the faces of a cube and creates six separate hilbert curves for "
        "each face. It then stitches these curves together into one continuous thread. This class does not need S2 "
        "Geometry Library. We are using a port callsed s2nano that pulls out the essential SFC routine. It basically "
        "works by converting a normal to being integrated into a s2_id (SFC unique integer). It performs a faster "
        "interpolated and branchless binary search to find the closest cell in buckets. It then performs a local "
        "neighborhood search centered around the cell which actually looks at the surface normal.")
        .def(py::init<const int, const double>(), "level"_a = FASTGA_LEVEL, "max_phi"_a = FASTGA_MAX_PHI)
        .def_readonly("bucket_neighbors", &FastGA::GaussianAccumulatorS2::bucket_neighbors,
                      "Fast lookup matrix to find neighbors of a bucket")
        .def("integrate", &FastGA::GaussianAccumulatorS2::Integrate, "normals"_a, "num_nbr"_a = FASTGA_TRI_NBRS,
             "Will intergrate the normals into the S2 Historgram")
        .def("__repr__", [](const FastGA::GaussianAccumulatorS2& a) {
            return "<FastGA::GAS2; # Triangles: '" + std::to_string(a.mesh.triangles.size()) + "'>";
        });

    py::class_<FastGA::Ico::IcoCharts>(
        m, "IcoCharts",
        "Contains charts of an unwrapped Icosahedron. This is basically my implementation of unwrapping an icosahedron "
        "as described in: Gauge Equivariant Convolutional Networks and the Icosahedral CNN - "
        "https://arxiv.org/abs/1902.04615")
        .def(py::init<const int>(), "level"_a = FASTGA_LEVEL)
        // .def_readonly("point_idx_to_image_idx", &FastGA::Ico::IcoChart::point_idx_to_image_idx)
        // .def_readonly("local_to_global_point_idx_map", &FastGA::Ico::IcoChart::local_to_global_point_idx_map)
        .def_readonly("image", &FastGA::Ico::IcoCharts::image, "Returns an unwrapped image of the IcoCharts")
        .def_readonly(
            "image_to_vertex_idx", &FastGA::Ico::IcoCharts::image_to_vertex_idx,
            "Fast lookup matrix for image creation. Each pixel hold the icosahedron vertex index it corresponds to")
        .def_readonly("mask", &FastGA::Ico::IcoCharts::mask,
                      "Boolean mask corresponding to valid cells. False(0) corresponds to ghost/halo cells")
        .def_readonly("sphere_mesh", &FastGA::Ico::IcoCharts::sphere_mesh,
                      "The full icosahedron the IcoChart is unwrapping")
        .def("fill_image", &FastGA::Ico::IcoCharts::FillImage, "normalized_vertex_count"_a,
             "Fills the the image using the normalized vertex counts")
        .def("find_peaks", &FastGA::Ico::IcoCharts::FindPeaks, "threshold_abs"_a = 25, "exclude_border"_a = false,
             "Finds all peaks inside the image")
        .def("__repr__",
             [](const FastGA::Ico::IcoCharts& a) { return "<IcoChart; Level: '" + std::to_string(a.level) + "'>"; });

    // Functions
    m.def("convert_normals_to_hilbert", &FastGA::Helper::ConvertNormalsToHilbert, "normals"_a, "bbox"_a,
          "Not recommended. Converts a numpy array of normals to uint32 Hilbert Values"
          "Assumes EqualArea Azimuth Projection centered at north pole. Only good on for northern hemisphere.");
    docstring::FunctionDocInject(
        m, "convert_normals_to_hilbert",
        {{"normals", "MatX3d; NX3 Array"}, {"bbox", "BBOX; bounding box of AzimuthProjection projection"}});

    m.def("convert_normals_to_s2id", &FastGA::Helper::ConvertNormalsToS2ID, "normals"_a,
          "Converts unit normals to uint64 S2 ids. Uses s2nano (micro port of S2 Geometry)");
    docstring::FunctionDocInject(m, "convert_normals_to_s2id", {{"normals", "MatX3d; NX3 Array"}});

    m.def("refine_icosahedron", &FastGA::Ico::RefineIcosahedron, "level"_a, "Creates a refined icosahedron mesh");
    docstring::FunctionDocInject(
        m, "refine_icosahedron",
        {{"level", "The level of refinement of the icosahedron. Each level recursively subdived triangles"}});

    m.def("refine_icochart", &FastGA::Ico::RefineIcoChart, "level"_a = 0, "square"_a = false,
          "Return an refined icochart");

    //////////////////
    //////////////////
    // Add new S2 Beta

    py::class_<FastGA::BucketS2>(m, "BucketS2",
                                        "The bucket class describes a cell on the S2 Histogram. It unfortunately has")
    .def(py::init<>())
    .def_readonly("normal", &FastGA::BucketS2::normal, "The surface **unit** normal of the triangle cell")
    .def_readonly("average_normal", &FastGA::BucketS2::average_normal, "The average surface **unit** normal of the triangle cell after integration")
    .def_readonly("hilbert_value", &FastGA::BucketS2::hilbert_value,
                    "Space Filling Curve value, may be Int32 or Uint64")
    .def_readonly("count", &FastGA::BucketS2::count, "Count variable for histogram")
    .def("__repr__", [](const FastGA::BucketS2& a) {
        return ("<Bucket Normal: " + FastGA::Helper::ArrayToString<double, 3>(a.normal) +
                "; HV: " + std::to_string(a.hilbert_value) + "; CNT: " + std::to_string(a.count) + "'>");
    });

    py::class_<FastGA::GaussianAccumulatorS2Beta>(
        m, "GaussianAccumulatorS2Beta",
        "This GaussianAccumulator can handle the entire sphere by using a space filling curve designed by Google's S2 "
        "Geometry library. It projects a sphere to the faces of a cube and creates six separate hilbert curves for "
        "each face. It then stitches these curves together into one continuous thread. This class does not need S2 "
        "Geometry Library. We are using a port callsed s2nano that pulls out the essential SFC routine. It basically "
        "works by converting a normal to being integrated into a s2_id (SFC unique integer). It performs a faster "
        "interpolated and branchless binary search to find the closest cell in buckets. It then performs a local "
        "neighborhood search centered around the cell which actually looks at the surface normal.")
        .def(py::init<const int>(), "level"_a = FASTGA_LEVEL)
        .def_readonly("bucket_neighbors", &FastGA::GaussianAccumulatorS2Beta::bucket_neighbors,
                      "Fast lookup matrix to find neighbors of a bucket")
        .def_readonly("mesh", &FastGA::GaussianAccumulatorS2Beta::mesh,
                      "The underlying sphere-like mesh of the Gaussian Accumulator")
        .def_readonly("buckets", &FastGA::GaussianAccumulatorS2Beta::buckets,
                      "The buckets in the histogram, corresponding to cells/triangles on the mesh")
        .def_readonly("num_buckets", &FastGA::GaussianAccumulatorS2Beta::num_buckets,
                      "The number of buckets in histogram, size(buckets)")
        .def("get_bucket_normals", &FastGA::GaussianAccumulatorS2Beta::GetBucketNormals,
             "Gets the surface normals of the buckets in the histogram."
             "The order by default is sorted by the space filling curve value attached to each cell.",
             "mesh_order"_a = false)
        .def("get_bucket_average_normals", &FastGA::GaussianAccumulatorS2Beta::GetBucketAverageNormals,
             "Gets the average surface normals of the buckets in the histogram after integration."
             "The order by default is sorted by the space filling curve value attached to each cell.",
             "mesh_order"_a = false)
        .def("get_normalized_bucket_counts", &FastGA::GaussianAccumulatorS2Beta::GetNormalizedBucketCounts,
             "Get the normalized bucket counts in the histogram."
             "The order by default is sorted by the space filling curve value attached to each cell.",
             "mesh_order"_a = false)
        .def("get_normalized_bucket_counts_by_vertex",
             &FastGA::GaussianAccumulatorS2Beta::GetNormalizedBucketCountsByVertex,
             "Average the normalized buckets counts (triangles) into the *vertices* of the mesh."
             "The order by default is sorted by the space filling curve value attached to each cell.",
             "mesh_order"_a = false)
        .def("get_average_normals_by_vertex",
             &FastGA::GaussianAccumulatorS2Beta::GetAverageNormalsByVertex,
             "Average the normalized buckets counts (triangles) into the *vertices* of the mesh."
             "The order by default is sorted by the space filling curve value attached to each cell.",
             "mesh_order"_a = false)
        .def("get_bucket_sfc_values", &FastGA::GaussianAccumulatorS2Beta::GetBucketSFCValues,
             "Get the space filling curve values of each bucket. Will be sorted low to high.")
        .def("clear_count", &FastGA::GaussianAccumulatorS2Beta::ClearCount,
             "Clears all the histogram counts for each cell. Useful to call after peak detection to 'reset' the mesh.")
        .def("copy_ico_mesh", &FastGA::GaussianAccumulatorS2Beta::CopyIcoMesh, "Creates a copy of the ico mesh.",
             "mesh_order"_a = false)
        .def("integrate", &FastGA::GaussianAccumulatorS2Beta::Integrate, "normals"_a, "num_nbr"_a = FASTGA_TRI_NBRS,
             "Will intergrate the normals into the S2 Historgram")
        .def("find_peaks_from_ico_charts", &FastGA::GaussianAccumulatorS2Beta::FindPeaksFromIcoCharts, "ico"_a, "threshold_abs"_a = 25, "exclude_border"_a=false,
             "Find the peaks on the Gaussian Accumulator")
        .def("__repr__", [](const FastGA::GaussianAccumulatorS2Beta& a) {
            return "<FastGA::GAS2; # Triangles: '" + std::to_string(a.mesh.triangles.size()) + "'>";
        });

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}