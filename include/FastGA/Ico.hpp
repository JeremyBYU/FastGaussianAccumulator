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
    IcoMesh() : vertices(), triangles(), triangle_normals(){}
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
} // namespace Ico
} // namespace FastGA
