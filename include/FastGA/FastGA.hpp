#ifndef FASTGA_GA_HPP
#define FASTGA_GA_HPP

#include <string>
#include "FastGA/Ico.hpp"
#include "nanoflann.hpp"

namespace FastGA {
const std::string test();


class GaussianAccumulator
{

  public:
    struct Bucket
    {
      std::array<double, 3> normal;
      uint32_t hilbert_value;
      uint32_t count;
    };
    Ico::IcoMesh mesh;
    std::vector<Bucket> buckets;
    GaussianAccumulator(int level = 1, double max_phi = 100);

  private:
};

class GaussianAccumulatorKD: public GaussianAccumulator
{

  public:
    GaussianAccumulatorKD(int level = 1, double max_phi = 100);

  private:
};

// GaussianAccumulator::GaussianAccumulator(int level) : mesh()
// {
//     mesh = FastGA::Ico::RefineIcosahedron(level);
// }

} // namespace FastGA
#endif
