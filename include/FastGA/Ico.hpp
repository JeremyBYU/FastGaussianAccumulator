#include "FastGA/Helper.hpp"

namespace FastGA {

namespace Ico {
const static double GOLDEN_RATIO = (1.0 + sqrt(5.0)) / 2.0;
const static double EQUILATERAL_TRIANGLE_RATIO = sqrt(3.0) / 2.0;
const static double ICOSAHEDRON_TRUE_RADIUS = sqrt(1.0 + GOLDEN_RATIO * GOLDEN_RATIO);
const static double ICOSAHEDRON_SCALING = 1.0 / ICOSAHEDRON_TRUE_RADIUS;
const static int NUMBER_OF_CHARTS = 5;
// using Triangles = std::vector<std::array<size_t, 3>>;
// using Vertices = std::vector<std::array<double, 3>>;

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

struct IcoMesh
{
    MatX3d vertices;
    MatX3I triangles;
    MatX3d triangle_normals;
    MatX6I adjacency_matrix;
    IcoMesh() : vertices(), triangles(), triangle_normals(), adjacency_matrix() {}
};

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
    size_t max_limit = std::numeric_limits<size_t>::max();
    std::unordered_map<size_t, std::unordered_set<size_t>> pi_to_triset = ComputeAdjacencySet(triangles);
    MatX6I adjacency_matrix(vertices.size(), {max_limit, max_limit, max_limit, max_limit, max_limit, max_limit});

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
            // std::array<size_t, 3> t1 = {{p1_idx, p4_idx, p6_idx}};
            // std::array<size_t, 3> t2 = {{p4_idx, p2_idx, p5_idx}};
            // std::array<size_t, 3> t3 = {{p6_idx, p5_idx, p3_idx}};
            // std::array<size_t, 3> t4 = {{p6_idx, p4_idx, p5_idx}};
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
    size_t max_limit = std::numeric_limits<size_t>::max();

    std::vector<size_t> halfedges(triangles.size() * 3, max_limit);
    MatX2I halfedges_pi(triangles.size() * 3);
    std::unordered_map<size_t, size_t> vertex_indices_to_half_edge_index;
    vertex_indices_to_half_edge_index.reserve(triangles.size() * 3);
    // auto after = std::chrono::high_resolution_clock::now();
    // auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(after - before);
    // std::cout << "Create Datastructures took " << elapsed.count() << " milliseconds" << std::endl;

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

    // auto after2 = std::chrono::high_resolution_clock::now();
    // elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(after2 - after);
    // std::cout << "Triangle loop took " << elapsed.count() << " milliseconds" << std::endl;
    // Fill twin half-edge. In the previous step, it is already guaranteed that
    // each half-edge can have at most one twin half-edge.
    for (size_t this_he_index = 0; this_he_index < halfedges.size(); this_he_index++)
    {
        size_t& that_he_index = halfedges[this_he_index];
        std::array<size_t, 2>& this_he = halfedges_pi[this_he_index];
        size_t that_he_mapped = CantorMapping(this_he[1], this_he[0]);
        if (that_he_index == max_limit &&
            vertex_indices_to_half_edge_index.find(that_he_mapped) !=
                vertex_indices_to_half_edge_index.end())
        {
            size_t twin_he_index =
                vertex_indices_to_half_edge_index[that_he_mapped];
            halfedges[this_he_index] = twin_he_index;
            halfedges[twin_he_index] = this_he_index;
        }
    }
    // auto after3 = std::chrono::high_resolution_clock::now();
    // elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(after3 - after2);
    // std::cout << "Half Edge loop took " << elapsed.count() << " milliseconds" << std::endl;

    // elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(after3 - before);
    // std::cout << "Total time took " << elapsed.count() << " milliseconds" << std::endl;

    return halfedges;
}

inline MatX12I ComputeTriangleNeighbors(const MatX3I& triangles, const MatX3d& triangle_normals, const size_t max_idx)
{
    size_t max_limit = std::numeric_limits<size_t>::max();
    // Must compute adjacency set again because triangle indices may been sorted by hilbert values
    std::unordered_map<size_t, std::unordered_set<size_t>> pi_to_triset = ComputeAdjacencySet(triangles);
    MatX12I neighbors(triangles.size(), {max_limit, max_limit, max_limit, max_limit, max_limit, max_limit, max_limit, max_limit, max_limit, max_limit, max_limit, max_limit});

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

class Image
{
  public:
    std::vector<uint8_t> buffer_;
    int rows_;
    int cols_;
    int bytes_per_channel_;
    bool is_float_;
    // Image() = default;
    Image(int rows, int cols, int bytes_per_channel, bool is_float = false) : buffer_(), rows_(rows), cols_(cols), bytes_per_channel_(bytes_per_channel), is_float_(is_float)
    {
        AllocateBuffer();
    }
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

constexpr int get_chart_width(int level, int padding)
{
    return static_cast<int>(std::pow(2, level + 1)) + (2 * padding);
}

constexpr int get_chart_height(int level, int padding)
{
    return static_cast<int>(std::pow(2, level)) + (2 * padding);
}

template int *Image::PointerAt<int>(int u, int v);
template uint8_t *Image::PointerAt<uint8_t>(int u, int v);


class IcoChart
{
  public:
    int level;
    int padding;
    MatX2I point_idx_to_image_idx;
    std::vector<std::vector<size_t>> local_to_global_point_idx_map;
    // int chart_width_padded;
    // int chart_height_padded;
    Image image;
    // each position/pixel on the image is directly mapped to a vertex on the refined icosahedron
    Image image_to_vertex_idx;
    IcoChart(const int level_ = 1, const int padding_ = 1) : level(level_), padding(padding_), point_idx_to_image_idx(), local_to_global_point_idx_map(NUMBER_OF_CHARTS), image(get_chart_height(level, padding) * 5, get_chart_width(level, padding), 1), image_to_vertex_idx(get_chart_height(level, padding) * 5, get_chart_width(level, padding), 4), sphere_mesh(), chart_template()
    {
        sphere_mesh = RefineIcosahedron(level);
        chart_template = RefineIcoChart(level, true);
        point_idx_to_image_idx = CreateChartImageIndices(level, chart_template);
        for (int i = 0; i < NUMBER_OF_CHARTS; ++i)
        {
            local_to_global_point_idx_map[i] = Create_Local_To_Global_Point_Idx_Map(sphere_mesh, chart_template, i);
        }
        ConstructImageToVertexIdx();

    }
    void FillImage(std::vector<double> normalized_vertex_count)
    {
        for(int row = 0; row < image.rows_; ++row)
        {
            for(int col = 0; col < image.cols_; ++col)
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

  protected:
  private:
    IcoMesh sphere_mesh;    // Full Refined Icosahedron Mesh on S2
    IcoMesh chart_template; // Single 2D Chart Template

    void ConstructImageToVertexIdx()
    {
        auto chart_height = get_chart_height(level, padding);
        for (int chart_idx = 0; chart_idx < NUMBER_OF_CHARTS; ++chart_idx)
        {
            int chart_height_offset = (NUMBER_OF_CHARTS - chart_idx - 1) * chart_height;
            // Get the mapping from the local point idx to global point index for this chart
            auto &local_to_global_point_idx_map_chart = local_to_global_point_idx_map[chart_idx];
            for (size_t local_point_idx = 0; local_point_idx < point_idx_to_image_idx.size(); ++local_point_idx)
            {
                const auto &global_point_idx = local_to_global_point_idx_map_chart[local_point_idx];
                const auto &img_coords = point_idx_to_image_idx[local_point_idx];
                const auto img_row = chart_height_offset + (chart_height - static_cast<int>(img_coords[1]) - 1);
                const auto img_col = static_cast<int>(img_coords[0]) + 1;
                auto img_ptr = image_to_vertex_idx.PointerAt<int>(img_col, img_row);
                *img_ptr = static_cast<int>(global_point_idx);
            }
        }
    }
};

} // namespace Ico
} // namespace FastGA
