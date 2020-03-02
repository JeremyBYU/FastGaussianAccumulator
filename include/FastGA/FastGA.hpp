#ifndef FASTGA_GA_HPP
#define FASTGA_GA_HPP

#include <string>
#include "FastGA/Ico.hpp"
#include "FastGA/Helper.hpp"
#include "FastGA/NanoFlannAdaptors.hpp"
#include <memory>

#define FastGA_LEVEL 1
#define FastGA_MAX_PHI 100
#define FastGA_MAX_LEAF_SIZE 10
#define FastGA_KDTREE_EPS 0.0
namespace FastGA {
template<class T>
class GaussianAccumulator
{

  public:
    Ico::IcoMesh mesh;
    std::vector<Bucket<T>> buckets;
    std::vector<uint8_t> mask;
    Helper::BBOX projected_bbox;
    GaussianAccumulator();
    GaussianAccumulator(const int level = FastGA_LEVEL, const double max_phi = FastGA_MAX_PHI);
    std::vector<double> GetNormalizedBucketCounts();
    std::vector<T> GetBucketIndices();
    std::vector<std::array<double,2>> GetBucketProjection();

  protected:
    void SortBucketsByIndices();
};

class GaussianAccumulatorKD : public GaussianAccumulator<uint32_t>
{

  public:
    GaussianAccumulatorKD(const int level = FastGA_LEVEL, const double max_phi = FastGA_MAX_PHI, const size_t max_leaf_size = FastGA_MAX_LEAF_SIZE);
    std::vector<size_t> Integrate(const MatX3d &normals, const float eps = FastGA_KDTREE_EPS);

  protected:
    const NFA::BUCKET2KD bucket2kd; // The adaptor
    const nanoflann::KDTreeSingleIndexAdaptorParams index_params;
    std::unique_ptr<NFA::nano_kd_tree_t> kd_tree_ptr;
};

} // namespace FastGA
#endif
