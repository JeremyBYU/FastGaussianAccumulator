#include "FastGA/Helper.hpp"

namespace FastGA {
namespace Ico {
const static double GOLDEN_RATIO = (1.0 + sqrt(5.0)) / 2.0;
const static double ICOSAHEDRON_TRUE_RADIUS = sqrt(1.0 + GOLDEN_RATIO * GOLDEN_RATIO);
const static double ICOSAHEDRON_SCALING = 1.0 / ICOSAHEDRON_TRUE_RADIUS;

using Triangles = std::vector<std::array<size_t, 3>>;
using Vertices = std::vector<std::array<double, 3>>;

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
    Vertices vertices;
    Triangles triangles;
    IcoMesh() : vertices(), triangles() {}
};

// using TriangleMesh = std::tuple<Vertices<T>, Triangles<T>>;

inline const IcoMesh
CreateIcosahedron(const double scale = ICOSAHEDRON_SCALING)
{
    const double p = GOLDEN_RATIO;
    IcoMesh mesh;
    Vertices& vertices = mesh.vertices;
    Triangles& triangles = mesh.triangles;
    // auto test = triangles[0].data();
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

    FastGA::ScaleArrayInPlace<double, 3>(vertices, scale);

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

inline std::array<double, 3> ConstructMidPoint(const size_t p1_idx, const size_t p2_idx, Vertices vertices)
{
    auto &p1 = vertices[p1_idx];
    auto &p2 = vertices[p2_idx];
    auto mean = FastGA::Mean<double, 3>(p1, p2);
    auto norm = FastGA::L2Norm<double, 3>(mean);
    auto scaling = 1 / norm;
    FastGA::ScaleItemInPlace(mean, scaling);
    return mean;
}

// def cantor_mapping(k1, k2):
//     return int(((k1 + k2) * (k1 + k2 + 1)) / 2.0 + k2)


// def generate_key_from_point(p1_idx, p2_idx):
//     lower_idx, higher_idx = (
//         p1_idx, p2_idx) if p1_idx < p2_idx else (p2_idx, p1_idx)
//     return cantor_mapping(lower_idx, higher_idx)


// def construct_midpoint(p1_idx, p2_idx, vertices):
//     p1 = vertices[p1_idx]
//     p2 = vertices[p2_idx]
//     midpoint_on_plane = (p2 + p1) / 2.0
//     scaling = 1 / np.linalg.norm(midpoint_on_plane)
//     midpoint_on_sphere = midpoint_on_plane * scaling
//     return midpoint_on_sphere


// def get_point_idx(p1_idx, p2_idx, point_to_idx_map, vertices):
//     point_key = generate_key_from_point(p1_idx, p2_idx)
//     if point_to_idx_map.get(point_key):
//         # Existing point Idx
//         return point_to_idx_map[point_key]
//     else:
//         # New point idx
//         point_to_idx_map[point_key] = len(vertices)
//         midpoint_on_sphere = construct_midpoint(p1_idx, p2_idx, vertices)
//         vertices.append(midpoint_on_sphere)
//         return point_to_idx_map[point_key]


// def refine_icosahedron(triangles, vertices, level=2):
//     vertices = list(vertices)
//     triangles = triangles.tolist()
//     # HashMap that maps a midpoint (identified by two point indexes) to its own point index
//     point_to_idx_map = dict()
//     for i in range(level):
//         triangles_refined = []
//         # loop through every triangle and create 4 new ones based upon midpoints
//         for triangle in triangles:
//             p1_idx = triangle[0]
//             p2_idx = triangle[1]
//             p3_idx = triangle[2]

//             # Create new points (if not existing) and return point index
//             p4_idx = get_point_idx(p1_idx, p2_idx, point_to_idx_map, vertices)
//             p5_idx = get_point_idx(p2_idx, p3_idx, point_to_idx_map, vertices)
//             p6_idx = get_point_idx(p3_idx, p1_idx, point_to_idx_map, vertices)
//             # Create the four new triangles
//             t1 = [p1_idx, p4_idx, p6_idx]
//             t2 = [p4_idx, p2_idx, p5_idx]
//             t3 = [p6_idx, p5_idx, p3_idx]
//             t4 = [p6_idx, p4_idx, p5_idx]
//             # Append triangles to the new refined triangle array
//             triangles_refined.extend([t1, t2, t3, t4])
//         # overwrite existing triangles with this new array
//         triangles = triangles_refined
//     vertices = np.array(vertices)
//     triangles = np.array(triangles)
//     return vertices, triangles

inline const IcoMesh RefineIcosahedron(const int level = 1)
{
    std::cout << level << std::endl;
    return CreateIcosahedron();
}
} // namespace Ico
} // namespace FastGA
