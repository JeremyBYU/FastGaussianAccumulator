#include "FastGA.hpp"
#include <algorithm>

namespace FastGA {
const std::string test()
{
    return "test";
}

GaussianAccumulator::GaussianAccumulator(int level, double max_phi) : mesh(), buckets()
{
    mesh = FastGA::Ico::RefineIcosahedron(level);
    auto max_phi_rad = degreesToRadians(max_phi);
    auto min_z = std::cos(max_phi_rad);

    // Create buckets from refined icosahedron mesh
    std::transform(mesh.triangle_normals.begin(), mesh.triangle_normals.end(), std::back_inserter(buckets),
                   [](std::array<double, 3>& normal) -> Bucket { return {normal, 0, 0}; });
    // Remove buckets whose phi (related to min_z) is too great
    buckets.erase(std::remove_if(buckets.begin(), buckets.end(),
                                 [&min_z](Bucket& b) { return b.normal[2] < min_z; }),
                  buckets.end());

}

GaussianAccumulatorKD::GaussianAccumulatorKD(int level, double max_phi) : GaussianAccumulator(level, max_phi)
{

}

} // namespace FastGA