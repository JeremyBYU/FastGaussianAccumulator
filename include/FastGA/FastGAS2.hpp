#ifndef FASTGA_S2_HPP
#define FASTGA_S2_HPP

#include "FastGA/FastGA.hpp"
#include "s2/s2point_index.h"

namespace FastGA {
class GaussianAccumulatorS2 : public GaussianAccumulator<uint64_t>
{

  public:
    MatX12I bucket_neighbors;
    GaussianAccumulatorS2(const int level = FastGA_LEVEL, const double max_phi = FastGA_MAX_PHI);
    std::vector<size_t> Integrate(const MatX3d& normals, const int num_nbr = FastGA_TRI_NBRS);

  protected:
};

} // namespace FastGA

#endif
