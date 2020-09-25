#include "FastGA/Helper.hpp"

namespace FastGA {

namespace Ico {
const static double GOLDEN_RATIO = (1.0 + sqrt(5.0)) / 2.0;
const static double EQUILATERAL_TRIANGLE_RATIO = sqrt(3.0) / 2.0;
const static double ICOSAHEDRON_TRUE_RADIUS = sqrt(1.0 + GOLDEN_RATIO * GOLDEN_RATIO);
const static double ICOSAHEDRON_SCALING = 1.0 / ICOSAHEDRON_TRUE_RADIUS;
const static int NUMBER_OF_CHARTS = 5;

inline constexpr int IcoMeshFaces(int level = 1)
{
    if (level == 0)
        return 20;
    else
        return IcoMeshFaces(level - 1) * 4;
}

inline constexpr int IcoMeshVertices(int level = 1)
{
    if (level == 1)
        return 12;
    return IcoMeshVertices(level - 1) + static_cast<int>(1.5 * IcoMeshFaces(level - 1));
}

/**
 * @brief This stores the mesh of a refined icosahedron.
 *        Note that the ordering of the vertices and triangles are very particular.
 *        This ordering must be maintained for use during an unwrapping procedure.
 */
struct IcoMesh
{
    /** @brief The vetrices in the mesh, NX3 */
    MatX3d vertices;
    /** @brief The triangles in the mesh, KX3 */
    MatX3I triangles;
    /** @brief The triangle normals in the mesh, KX3 */
    MatX3d triangle_normals;
    /** @brief The vertex adjacency matrix, NX6 */
    MatX6I adjacency_matrix;
    IcoMesh() : vertices(), triangles(), triangle_normals(), adjacency_matrix() {}
};

// An IcoChart is one of 5 face groups on an icosahedron
// This function is used to create a 2D representation of one of these charts
inline const IcoMesh CreateIcoChart(bool square = false)
{
    IcoMesh mesh;
    MatX3d& vertices = mesh.vertices;
    MatX3I& triangles = mesh.triangles;
    const double half_edge = 0.5;

    if (square)
    {
        vertices.push_back({0, 0, 0}); // 0
        vertices.push_back({0, 1, 0}); // 1
        vertices.push_back({1, 1, 0}); // 2
        vertices.push_back({1, 0, 0}); // 3
        vertices.push_back({2, 1, 0}); // 4
        vertices.push_back({2, 0, 0}); // 5
    }
    else
    {
        vertices.push_back({0, 0, 0});                                        // 0
        vertices.push_back({-half_edge, EQUILATERAL_TRIANGLE_RATIO, 0});      // 1
        vertices.push_back({half_edge, EQUILATERAL_TRIANGLE_RATIO, 0});       // 2
        vertices.push_back({1, 0, 0});                                        // 3
        vertices.push_back({half_edge + 1.0, EQUILATERAL_TRIANGLE_RATIO, 0}); // 4
        vertices.push_back({2, 0, 0});                                        // 5
    }

    // Chart
    triangles.push_back({0, 2, 1});
    triangles.push_back({0, 3, 2});
    triangles.push_back({2, 3, 4});
    triangles.push_back({5, 4, 3});

    return mesh;
}

inline const IcoMesh
CreateIcosahedron(const double scale = ICOSAHEDRON_SCALING)
{
    const double p = GOLDEN_RATIO;
    IcoMesh mesh;
    MatX3d& vertices = mesh.vertices;
    MatX3I& triangles = mesh.triangles;
    vertices.push_back({-1, 0, p});
    vertices.push_back({1, 0, p});
    vertices.push_back({1, 0, -p});
    vertices.push_back({-1, 0, -p});
    vertices.push_back({0, -p, 1});
    vertices.push_back({0, p, 1});
    vertices.push_back({0, p, -1});
    vertices.push_back({0, -p, -1});
    vertices.push_back({-p, -1, 0});
    vertices.push_back({p, -1, 0});
    vertices.push_back({p, 1, 0});
    vertices.push_back({-p, 1, 0});

    FastGA::Helper::ScaleArrayInPlace<double, 3>(vertices, scale);

    // Specifically arranged into face groups (aka Charts)
    // Each chart start near the top (high z) and goes down
    // Chart One
    triangles.push_back({0, 4, 1});
    triangles.push_back({0, 8, 4});
    triangles.push_back({4, 8, 7});
    triangles.push_back({3, 7, 8});
    // Chart Two
    triangles.push_back({4, 9, 1});
    triangles.push_back({4, 7, 9});
    triangles.push_back({9, 7, 2});
    triangles.push_back({3, 2, 7});
    // Chart Three
    triangles.push_back({9, 10, 1});
    triangles.push_back({9, 2, 10});
    triangles.push_back({10, 2, 6});
    triangles.push_back({3, 6, 2});
    // Chart Four
    triangles.push_back({10, 5, 1});
    triangles.push_back({10, 6, 5});
    triangles.push_back({5, 6, 11});
    triangles.push_back({3, 11, 6});
    // Chart Five
    triangles.push_back({5, 0, 1});
    triangles.push_back({5, 11, 0});
    triangles.push_back({0, 11, 8});
    triangles.push_back({3, 8, 11});

    return mesh;
}

inline std::unordered_map<size_t, std::unordered_set<size_t>> ComputeAdjacencySet(const MatX3I& triangles)
{
    std::unordered_map<size_t, std::unordered_set<size_t>> adjacency_set;
    for (size_t i = 0; i < triangles.size(); i++)
    {
        auto& pi0 = triangles[i][0];
        auto& pi1 = triangles[i][1];
        auto& pi2 = triangles[i][2];

        auto& tri_set_p0 = adjacency_set[pi0];
        auto& tri_set_p1 = adjacency_set[pi1];
        auto& tri_set_p2 = adjacency_set[pi2];

        tri_set_p0.insert(i);
        tri_set_p1.insert(i);
        tri_set_p2.insert(i);
    }
    return adjacency_set;
}

inline MatX6I ComputeAdjacencyMatrix(const MatX3I& triangles, const MatX3d& vertices)
{
    size_t max_limit_ = std::numeric_limits<size_t>::max();
    std::unordered_map<size_t, std::unordered_set<size_t>> pi_to_triset = ComputeAdjacencySet(triangles);
    MatX6I adjacency_matrix(vertices.size(), {max_limit_, max_limit_, max_limit_, max_limit_, max_limit_, max_limit_});

    for (size_t p_idx = 0; p_idx < vertices.size(); p_idx++)
    {
        auto& tri_set = pi_to_triset[p_idx];
        int counter = 0;
        for (const auto& tri_idx : tri_set)
        {
            adjacency_matrix[p_idx][counter] = tri_idx;
            counter++;
        }
    }
    return adjacency_matrix;
}

inline size_t CantorMapping(const size_t k1, const size_t k2)
{
    auto dk1 = static_cast<double>(k1);
    auto dk2 = static_cast<double>(k2);
    auto mapping = static_cast<size_t>(((dk1 + dk2) * (dk1 + dk2 + 1)) / 2.0 + dk2);
    return mapping;
}

inline size_t GenerateKeyFromPoint(const size_t p1_idx, const size_t p2_idx)
{
    size_t lower_idx = p1_idx;
    size_t higher_idx = p2_idx;
    if (p1_idx > p2_idx)
    {
        lower_idx = p2_idx;
        higher_idx = p1_idx;
    }
    return CantorMapping(lower_idx, higher_idx);
}

inline std::array<double, 3> ConstructMidPoint(const size_t p1_idx, const size_t p2_idx, MatX3d& vertices, bool scale = true)
{
    auto& p1 = vertices[p1_idx];
    auto& p2 = vertices[p2_idx];
    auto mean = FastGA::Helper::Mean<double, 3>(p1, p2);
    if (scale)
    {
        auto norm = FastGA::Helper::L2Norm<double, 3>(mean);
        auto scaling = 1 / norm;
        FastGA::Helper::ScaleItemInPlace(mean, scaling);
    }
    return mean;
}

inline size_t GetPointIdx(const size_t p1_idx, const size_t p2_idx, std::map<size_t, size_t>& point_to_idx, MatX3d& vertices, bool scale = true)
{
    auto point_key = GenerateKeyFromPoint(p1_idx, p2_idx);
    if (point_to_idx.find(point_key) != point_to_idx.end())
    {
        return point_to_idx[point_key];
    }
    else
    {
        point_to_idx[point_key] = vertices.size();
        auto midpoint_on_sphere = ConstructMidPoint(p1_idx, p2_idx, vertices, scale);
        vertices.push_back(midpoint_on_sphere);
        return point_to_idx[point_key];
    }
}

inline void RefineMesh(IcoMesh& mesh, const int level = 1, bool scale = true)
{
    auto& vertices = mesh.vertices;
    auto& triangles = mesh.triangles;
    std::map<size_t, size_t> point_to_idx;

    for (int i = 0; i < level; i++)
    {
        MatX3I triangles_refined;
        for (auto& triangle : triangles)
        {
            auto& p1_idx = triangle[0];
            auto& p2_idx = triangle[1];
            auto& p3_idx = triangle[2];
            // Create new points (if not existing) and return point index
            auto p4_idx = GetPointIdx(p1_idx, p2_idx, point_to_idx, vertices, scale);
            auto p5_idx = GetPointIdx(p2_idx, p3_idx, point_to_idx, vertices, scale);
            auto p6_idx = GetPointIdx(p3_idx, p1_idx, point_to_idx, vertices, scale);

            // Create the four new triangles
            std::array<size_t, 3> t1 = {{p3_idx, p6_idx, p5_idx}};
            std::array<size_t, 3> t2 = {{p6_idx, p4_idx, p5_idx}};
            std::array<size_t, 3> t3 = {{p5_idx, p4_idx, p2_idx}};
            std::array<size_t, 3> t4 = {{p6_idx, p1_idx, p4_idx}};
            // Append triangles to the new array
            triangles_refined.push_back(t1);
            triangles_refined.push_back(t2);
            triangles_refined.push_back(t3);
            triangles_refined.push_back(t4);
        }
        // copy constructor
        triangles = triangles_refined;
    }
    mesh.adjacency_matrix = ComputeAdjacencyMatrix(triangles, vertices);
}

inline const IcoMesh RefineIcosahedron(const int level = 1)
{
    auto mesh = CreateIcosahedron();
    RefineMesh(mesh, level, true);
    FastGA::Helper::ComputeTriangleNormals(mesh.vertices, mesh.triangles, mesh.triangle_normals);

    return mesh;
}

inline std::vector<size_t> ExtractHalfEdges(const MatX3I& triangles)
{
    // auto before = std::chrono::high_resolution_clock::now();
    size_t max_limit_ = std::numeric_limits<size_t>::max();

    std::vector<size_t> halfedges(triangles.size() * 3, max_limit_);
    MatX2I halfedges_pi(triangles.size() * 3);
    std::unordered_map<size_t, size_t> vertex_indices_to_half_edge_index;
    vertex_indices_to_half_edge_index.reserve(triangles.size() * 3);

    for (size_t triangle_index = 0; triangle_index < triangles.size(); triangle_index++)
    {

        const std::array<size_t, 3>& triangle = triangles[triangle_index];
        size_t num_half_edges = triangle_index * 3;

        size_t he_0_index = num_half_edges;
        size_t he_1_index = num_half_edges + 1;
        size_t he_2_index = num_half_edges + 2;
        size_t he_0_mapped = CantorMapping(triangle[0], triangle[1]);
        size_t he_1_mapped = CantorMapping(triangle[1], triangle[2]);
        size_t he_2_mapped = CantorMapping(triangle[2], triangle[0]);

        std::array<size_t, 2>& he_0 = halfedges_pi[he_0_index];
        std::array<size_t, 2>& he_1 = halfedges_pi[he_1_index];
        std::array<size_t, 2>& he_2 = halfedges_pi[he_2_index];

        he_0[0] = triangle[0];
        he_0[1] = triangle[1];
        he_1[0] = triangle[1];
        he_1[1] = triangle[2];
        he_2[0] = triangle[2];
        he_2[1] = triangle[0];

        vertex_indices_to_half_edge_index[he_0_mapped] = he_0_index;
        vertex_indices_to_half_edge_index[he_1_mapped] = he_1_index;
        vertex_indices_to_half_edge_index[he_2_mapped] = he_2_index;
    }

    for (size_t this_he_index = 0; this_he_index < halfedges.size(); this_he_index++)
    {
        size_t& that_he_index = halfedges[this_he_index];
        std::array<size_t, 2>& this_he = halfedges_pi[this_he_index];
        size_t that_he_mapped = CantorMapping(this_he[1], this_he[0]);
        if (that_he_index == max_limit_ &&
            vertex_indices_to_half_edge_index.find(that_he_mapped) !=
                vertex_indices_to_half_edge_index.end())
        {
            size_t twin_he_index =
                vertex_indices_to_half_edge_index[that_he_mapped];
            halfedges[this_he_index] = twin_he_index;
            halfedges[twin_he_index] = this_he_index;
        }
    }

    return halfedges;
}

inline MatX12I ComputeTriangleNeighbors(const MatX3I& triangles, const MatX3d& triangle_normals, const size_t max_idx)
{
    size_t max_limit_ = std::numeric_limits<size_t>::max();
    // Must compute adjacency set again because triangle indices may been sorted by hilbert values
    std::unordered_map<size_t, std::unordered_set<size_t>> pi_to_triset = ComputeAdjacencySet(triangles);
    MatX12I neighbors(triangles.size(), {max_limit_, max_limit_, max_limit_, max_limit_, max_limit_, max_limit_, max_limit_, max_limit_, max_limit_, max_limit_, max_limit_, max_limit_});

    for (size_t i = 0; i < triangles.size(); i++)
    {

        if (i >= max_idx)
            continue;

        auto this_normal = triangle_normals[i];
        std::unordered_set<size_t> neighbors_set;

        auto& pi0 = triangles[i][0];
        auto& pi1 = triangles[i][1];
        auto& pi2 = triangles[i][2];

        auto& tri_set_p0 = pi_to_triset[pi0];
        auto& tri_set_p1 = pi_to_triset[pi1];
        auto& tri_set_p2 = pi_to_triset[pi2];

        neighbors_set.insert(tri_set_p0.begin(), tri_set_p0.end());
        neighbors_set.insert(tri_set_p1.begin(), tri_set_p1.end());
        neighbors_set.insert(tri_set_p2.begin(), tri_set_p2.end());

        std::vector<std::tuple<size_t, double>> neighbor_idx_and_dist;
        std::transform(neighbors_set.begin(), neighbors_set.end(), std::back_inserter(neighbor_idx_and_dist),
                       [&this_normal, &triangle_normals](const size_t& nbr_idx) { return std::make_tuple(nbr_idx, Helper::SquaredDistance(this_normal, triangle_normals[nbr_idx])); });

        std::sort(neighbor_idx_and_dist.begin(), neighbor_idx_and_dist.end(),
                  [](const std::tuple<size_t, double>& a, const std::tuple<size_t, double>& b) { return std::get<1>(a) < std::get<1>(b); });

        for (size_t nbr_list_idx = 0; nbr_list_idx < neighbor_idx_and_dist.size(); nbr_list_idx++)
        {
            if (nbr_list_idx == 0)
                continue;
            auto& nbr_idx_dist = neighbor_idx_and_dist[nbr_list_idx];
            auto& nbr_tri_idx = std::get<0>(nbr_idx_dist);
            neighbors[i][nbr_list_idx - 1] = nbr_tri_idx;
        }
    }

    return neighbors;
}

inline const IcoMesh RefineIcoChart(const int level = 1, bool square = false)
{
    auto mesh = CreateIcoChart(square);
    RefineMesh(mesh, level, false);
    return mesh;
}

inline MatX2I CreateChartImageIndices(const int level, IcoMesh& chart_template)
{
    double scale_to_integer = std::pow(2, level);
    auto& vertices = chart_template.vertices;
    // Convert doubles to integer
    MatX2I point_idx_to_image_idx;
    point_idx_to_image_idx.reserve(vertices.size());
    std::transform(vertices.begin(), vertices.end(), std::back_inserter(point_idx_to_image_idx), [scale_to_integer](std::array<double, 3>& a) -> std::array<size_t, 2> {
        auto u = static_cast<size_t>(scale_to_integer * a[0]);
        auto v = static_cast<size_t>(scale_to_integer * a[1]);
        return {{u, v}};
    });

    return point_idx_to_image_idx;
}

inline std::vector<size_t> Create_Local_To_Global_Point_Idx_Map(IcoMesh& sphere_mesh, IcoMesh& chart_template, int chart_idx)
{
    MatX3I& sphere_triangles = sphere_mesh.triangles;   // all triangles on S2
    MatX3I& chart_triangles = chart_template.triangles; //
    auto num_chart_triangles = chart_triangles.size();
    auto num_chart_vertices = chart_template.vertices.size();
    std::vector<size_t> local_to_global_point_idx_map(num_chart_vertices);

    auto chart_tri_start_idx = chart_idx * num_chart_triangles;

    for (size_t i = 0; i < num_chart_triangles; ++i)
    {
        auto& local_tri = chart_triangles[i];
        auto& global_tri = sphere_triangles[chart_tri_start_idx + i];
        for (size_t idx = 0; idx < 3; ++idx)
        {
            local_to_global_point_idx_map[local_tri[idx]] = global_tri[idx];
        }
    }
    return local_to_global_point_idx_map;
}

/**
 * An image class. This stores unwrapped values of the icosahedron as a 2D image
 * 
 */
class Image
{
  public:
    /** @brief The underlying memory buffer of the data */
    std::vector<uint8_t> buffer_;
    /** @brief the rows in the image */
    int rows_;
    /** @brief the columns in the image */
    int cols_;
    /** @brief the bytes per pixel in a chanel, e.g, Float32 = 4 */
    int bytes_per_channel_;
    /** @brief Weather the data is a float or an int */
    bool is_float_;
    // Image() = default;
    /**
     * @brief Construct a new Image object.
     * 
     * @param rows              Rows in image
     * @param cols              Columns in image    
     * @param bytes_per_channel Bytes per channel (e.g., Float32 = 4)
     * @param is_float          Weather the data is a float or an int
     */
    Image(int rows, int cols, int bytes_per_channel, bool is_float = false) : buffer_(), rows_(rows), cols_(cols), bytes_per_channel_(bytes_per_channel), is_float_(is_float)
    {
        AllocateBuffer();
    }

    /**
     * @brief Access the data of the image
     * 
     * @tparam T                Data type
     * @param u                 Index of first dimension, row
     * @param v                 Index of second dimension, column
     * @return T*               Pointer to data
     */
    template <typename T>
    T* PointerAt(int u, int v)
    {
        return reinterpret_cast<T*>(buffer_.data() + (v * cols_ + u) * sizeof(T));
    }

  private:
    void AllocateBuffer()
    {
        buffer_.resize(cols_ * rows_ * bytes_per_channel_);
    }
};

inline int get_chart_width(int level, int padding)
{
    return static_cast<int>(std::pow(2, level + 1)) + (2 * padding);
}

inline int get_chart_height(int level, int padding)
{
    return static_cast<int>(std::pow(2, level)) + (2 * padding);
}

template int* Image::PointerAt<int>(int u, int v);
template uint8_t* Image::PointerAt<uint8_t>(int u, int v);

/**
 * @brief This is basically my implementation of unwrapping an 
 *        icosahedron as described in:
 *        Gauge Equivariant Convolutional Networks and the Icosahedral CNN - https://arxiv.org/abs/1902.04615
 * 
 */
class IcoCharts
{
  public:
    /** @brief The refinement level. Iterations of recursive subdivision. */
    int level;
    /** @brief padding around the image, padding = 1 for 3X3 kernel */
    int padding; //
    /** @brief Unwrapped refined icosahedron as an image */
    Image image;
    /** @brief each pixel on the image is directly mapped to a vertex idx on the refined icosahedron */
    Image image_to_vertex_idx; // 
    /** @brief mask of image which indcates which cells are valid, useful to know what pixels are ghost cells */
    Image mask;    
    /** @brief Full Refined Icosahedron Mesh on S2 */      
    IcoMesh sphere_mesh;

    /**
     * @brief Construct a new Ico Charts object.
     * 
     * @param level_         The refinement level. Iterations of recursive subdivision.   
     * @param padding_       Padding around the image, padding = 1 for 3X3 kernel. Please leave at default level, have not correctly made it general.
     */
    IcoCharts(const int level_ = 1, const int padding_ = 1) : level(level_), padding(padding_), image(get_chart_height(level, padding) * NUMBER_OF_CHARTS, get_chart_width(level, padding), 1), image_to_vertex_idx(get_chart_height(level, padding) * NUMBER_OF_CHARTS, get_chart_width(level, padding), 4), mask(get_chart_height(level, padding) * NUMBER_OF_CHARTS, get_chart_width(level, padding), 1), sphere_mesh(), chart_template(), point_idx_to_image_idx(), local_to_global_point_idx_map(NUMBER_OF_CHARTS)
    {
        sphere_mesh = RefineIcosahedron(level);
        chart_template = RefineIcoChart(level, true);
        point_idx_to_image_idx = CreateChartImageIndices(level, chart_template);
        for (int i = 0; i < NUMBER_OF_CHARTS; ++i)
        {
            local_to_global_point_idx_map[i] = Create_Local_To_Global_Point_Idx_Map(sphere_mesh, chart_template, i);
        }
        ConstructImageToVertexIdx();
        ConstructImageMask();
    }

    /**
     * @brief Will take the normalized counts of vertices of the refined icosahedron at copy them to the unwrapped image.
     * 
     * @param normalized_vertex_count       These are the normalized counts of the histogram of the **vertices** of the icosahedron
     */
    void FillImage(std::vector<double> normalized_vertex_count)
    {
        for (int row = 0; row < image.rows_; ++row)
        {
            for (int col = 0; col < image.cols_; ++col)
            {
                // get vertex index that corresponds to this image position
                auto ico_vertex_index = *image_to_vertex_idx.PointerAt<int>(col, row);
                // get pointer to our image at this position
                auto img_pointer = image.PointerAt<uint8_t>(col, row);
                // set value of the image
                *img_pointer = static_cast<uint8_t>(255.0 * normalized_vertex_count[ico_vertex_index]);
            }
        }
    }

  private:
    IcoMesh chart_template;                                         // Single 2D Chart Template
    MatX2I point_idx_to_image_idx;                                  // Maps a point index to an image index
    std::vector<std::vector<size_t>> local_to_global_point_idx_map; // 5 vectors for each chart, maps a charts local point index to the global point index
    void ConstructImageMask()
    {
        // Start off with every pixel being valid
        auto chart_height = get_chart_height(level, padding);
        std::fill(mask.buffer_.begin(), mask.buffer_.end(), 255);
        // Set first column to invalid (ghost column)
        for (int row = 0; row < mask.rows_; ++row)
        {
            auto img_pointer = mask.PointerAt<uint8_t>(0, row);
            *img_pointer = 0;
        }
        // Iterate through the 5 ghost rows, set all cols of these rows to 0
        for (int chart_idx = 0; chart_idx < NUMBER_OF_CHARTS; ++chart_idx)
        {
            int row = (chart_idx + 1) * chart_height - 1;
            for (int col = 0; col < mask.cols_; ++col)
            {
                auto img_pointer = mask.PointerAt<uint8_t>(col, row);
                *img_pointer = 0;
            }
        }
    }
    void ConstructImageToVertexIdx()
    {
        auto chart_height = get_chart_height(level, padding);
        for (int chart_idx = 0; chart_idx < NUMBER_OF_CHARTS; ++chart_idx)
        {
            // Our y axis for our image will be growing DOWN. Basically we are inverting the y-axis
            int chart_height_offset = (NUMBER_OF_CHARTS - chart_idx - 1) * chart_height;
            // Get the mapping from the local point idx to global point index for this chart
            auto& local_to_global_point_idx_map_chart = local_to_global_point_idx_map[chart_idx];
            for (size_t local_point_idx = 0; local_point_idx < point_idx_to_image_idx.size(); ++local_point_idx)
            {
                const auto& global_point_idx = local_to_global_point_idx_map_chart[local_point_idx];
                const auto& img_coords = point_idx_to_image_idx[local_point_idx];
                // Once again we need to flip the y-axis
                const auto img_row = chart_height_offset + (chart_height - static_cast<int>(img_coords[1]) - 2);
                const auto img_col = static_cast<int>(img_coords[0]) + 1;
                auto img_ptr = image_to_vertex_idx.PointerAt<int>(img_col, img_row);
                *img_ptr = static_cast<int>(global_point_idx);
            }
        }

        // Fill in Ghost Cells (Copy Operations)
        MatX2i to_flattened_indices;
        MatX2i from_flattened_indices;
        std::tie(to_flattened_indices, from_flattened_indices) = GetGhostCellIndices();
        for (size_t i = 0; i < to_flattened_indices.size(); i++)
        {
            auto& to_index = to_flattened_indices[i];
            auto& from_index = from_flattened_indices[i];
            auto img_ptr = image_to_vertex_idx.PointerAt<int>(to_index[1], to_index[0]);
            *img_ptr = *(image_to_vertex_idx.PointerAt<int>(from_index[1], from_index[0]));
        }
    }
    std::tuple<MatX2i, MatX2i> GetGhostCellIndices()
    {
        auto chart_height = get_chart_height(level, padding);
        auto sub_block_width = static_cast<int>(std::pow(2, level));
        auto block_width = static_cast<int>(std::pow(2, level + 1));

        // Create a range iterator....
        std::vector<int> nums(NUMBER_OF_CHARTS);
        std::iota(nums.begin(), nums.end(), 0);

        MatX2i to_flattened_indices;
        MatX2i from_flattened_indices;

        MatX4i to_indices;
        MatX4i from_indices;
        // The ghost cells here assume 1 padding
        // TODO make generic for padding
        // Copies for first ghost column
        std::for_each(nums.begin(), nums.end(), [&](int i) {
            auto start_row = i * chart_height + 1;
            auto end_row = i * chart_height + 1 + sub_block_width;
            to_indices.emplace_back(std::array<int, 4>{start_row, end_row, 0, 1});
        });

        std::for_each(nums.begin(), nums.end(), [&](int i) {
            auto start_row = ((i + 1) % NUMBER_OF_CHARTS) * chart_height + 1;
            auto end_row = start_row + 1;
            from_indices.emplace_back(std::array<int, 4>{start_row, end_row, 1, sub_block_width + 1});
        });

        // Copies for ghost row, left block
        std::for_each(nums.begin(), nums.end(), [&](int i) {
            auto start_row = (i + 1) * chart_height - 1;
            auto end_row = start_row + 1;
            auto start_col = 1;
            auto end_col = sub_block_width + 1;
            to_indices.emplace_back(std::array<int, 4>{start_row, end_row, start_col, end_col});
        });

        std::for_each(nums.begin(), nums.end(), [&](int i) {
            auto start_row = ((i + 1) % NUMBER_OF_CHARTS) * chart_height + 1;
            auto end_row = start_row + 1;
            auto start_col = 1 + sub_block_width;
            auto end_col = 1 + 2 * sub_block_width;
            from_indices.emplace_back(std::array<int, 4>{start_row, end_row, start_col, end_col});
        });

        // Copies for ghost row, right block
        std::for_each(nums.begin(), nums.end(), [&](int i) {
            auto start_row = (i + 1) * chart_height - 1;
            auto end_row = start_row + 1;
            auto start_col = 1 + sub_block_width;
            auto end_col = 1 + 2 * sub_block_width;
            to_indices.emplace_back(std::array<int, 4>{start_row, end_row, start_col, end_col});
        });

        std::for_each(nums.begin(), nums.end(), [&](int i) {
            auto start_row = ((i + 1) % NUMBER_OF_CHARTS) * chart_height + 1;
            auto end_row = start_row + sub_block_width;
            auto start_col = block_width;
            auto end_col = block_width + 1;
            from_indices.emplace_back(std::array<int, 4>{start_row, end_row, start_col, end_col});
        });

        flatten_indices(to_indices, to_flattened_indices);
        flatten_indices(from_indices, from_flattened_indices);

        return std::make_tuple(to_flattened_indices, from_flattened_indices);
    }
    void flatten_indices(std::vector<std::array<int, 4>>& indices, std::vector<std::array<int, 2>>& flattened_indices)
    {
        for (auto& row_col_idx : indices)
        {
            for (auto row = row_col_idx[0]; row < row_col_idx[1]; ++row)
            {
                for (auto col = row_col_idx[2]; col < row_col_idx[3]; ++col)
                {
                    flattened_indices.emplace_back(std::array<int, 2>{row, col});
                }
            }
        }
    }
};

} // namespace Ico
} // namespace FastGA
