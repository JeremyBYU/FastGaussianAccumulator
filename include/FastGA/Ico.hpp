#include "FastGA/Helper.hpp"

namespace FastGA {
namespace Ico {
const static double GOLDEN_RATIO = (1.0 + sqrt(5.0)) / 2.0;
const static double ICOSAHEDRON_TRUE_RADIUS = sqrt(1.0 + GOLDEN_RATIO * GOLDEN_RATIO);
const static double ICOSAHEDRON_SCALING = 1.0 / ICOSAHEDRON_TRUE_RADIUS;

// using Triangles = std::vector<std::array<size_t, 3>>;
// using Vertices = std::vector<std::array<double, 3>>;

inline constexpr int IcoMeshFaces(int level = 1)
{
    return 20 * level;
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
    IcoMesh() : vertices(), triangles(), triangle_normals() {}
};

// using TriangleMesh = std::tuple<Vertices<T>, Triangles<T>>;

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

    triangles.push_back({0, 4, 1});
    triangles.push_back({0, 1, 5});
    triangles.push_back({1, 4, 9});
    triangles.push_back({1, 9, 10});
    triangles.push_back({1, 10, 5});
    triangles.push_back({0, 8, 4});
    triangles.push_back({0, 11, 8});
    triangles.push_back({0, 5, 11});
    triangles.push_back({5, 6, 11});
    triangles.push_back({5, 10, 6});
    triangles.push_back({4, 8, 7});
    triangles.push_back({4, 7, 9});
    triangles.push_back({3, 6, 2});
    triangles.push_back({3, 2, 7});
    triangles.push_back({2, 6, 10});
    triangles.push_back({2, 10, 9});
    triangles.push_back({2, 9, 7});
    triangles.push_back({3, 11, 6});
    triangles.push_back({3, 8, 11});
    triangles.push_back({3, 7, 8});
    return mesh;
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

inline std::array<double, 3> ConstructMidPoint(const size_t p1_idx, const size_t p2_idx, MatX3d& vertices)
{
    auto& p1 = vertices[p1_idx];
    auto& p2 = vertices[p2_idx];
    auto mean = FastGA::Helper::Mean<double, 3>(p1, p2);
    auto norm = FastGA::Helper::L2Norm<double, 3>(mean);
    auto scaling = 1 / norm;
    FastGA::Helper::ScaleItemInPlace(mean, scaling);
    return mean;
}

inline size_t GetPointIdx(const size_t p1_idx, const size_t p2_idx, std::map<size_t, size_t>& point_to_idx, MatX3d& vertices)
{
    auto point_key = GenerateKeyFromPoint(p1_idx, p2_idx);
    if (point_to_idx.find(point_key) != point_to_idx.end())
    {
        return point_to_idx[point_key];
    }
    else
    {
        point_to_idx[point_key] = vertices.size();
        auto midpoint_on_sphere = ConstructMidPoint(p1_idx, p2_idx, vertices);
        vertices.push_back(midpoint_on_sphere);
        return point_to_idx[point_key];
    }
}

inline const IcoMesh RefineIcosahedron(const int level = 1)
{
    auto mesh = CreateIcosahedron();
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
            auto p4_idx = GetPointIdx(p1_idx, p2_idx, point_to_idx, vertices);
            auto p5_idx = GetPointIdx(p2_idx, p3_idx, point_to_idx, vertices);
            auto p6_idx = GetPointIdx(p3_idx, p1_idx, point_to_idx, vertices);

            // Create the four new triangles
            std::array<size_t, 3> t1 = {{p1_idx, p4_idx, p6_idx}};
            std::array<size_t, 3> t2 = {{p4_idx, p2_idx, p5_idx}};
            std::array<size_t, 3> t3 = {{p6_idx, p5_idx, p3_idx}};
            std::array<size_t, 3> t4 = {{p6_idx, p4_idx, p5_idx}};
            // Append triangles to the new array
            triangles_refined.push_back(t1);
            triangles_refined.push_back(t2);
            triangles_refined.push_back(t3);
            triangles_refined.push_back(t4);
        }
        // copy constructor
        triangles = triangles_refined;
    }

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
    std::unordered_map<size_t, std::unordered_set<size_t>> pi_to_triset;
    MatX12I neighbors(triangles.size(), {max_limit, max_limit, max_limit, max_limit, max_limit, max_limit, max_limit, max_limit, max_limit, max_limit, max_limit, max_limit});
    for (size_t i = 0; i < triangles.size(); i++)
    {

        if (i >= max_idx)
            continue;
        auto& pi0 = triangles[i][0];
        auto& pi1 = triangles[i][1];
        auto& pi2 = triangles[i][2];

        auto& tri_set_p0 = pi_to_triset[pi0];
        auto& tri_set_p1 = pi_to_triset[pi1];
        auto& tri_set_p2 = pi_to_triset[pi2];

        tri_set_p0.insert(i);
        tri_set_p1.insert(i);
        tri_set_p2.insert(i);
    }

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

} // namespace Ico
} // namespace FastGA
