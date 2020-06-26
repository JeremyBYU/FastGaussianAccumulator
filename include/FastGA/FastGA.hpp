#ifndef FASTGA_GA_HPP
#define FASTGA_GA_HPP

#include <string>
#include "FastGA/Ico.hpp"
#include "FastGA/Helper.hpp"
#include "FastGA/FastBinarySearch.hpp"
#include "FastGA/NanoFlannAdaptors.hpp"
#include <memory>

#define FASTGA_LEVEL 1
#define FASTGA_MAX_PHI 180.0
#define FASTGA_MAX_LEAF_SIZE 10
#define FASTGA_KDTREE_EPS 0.0
#define FASTGA_EXHAUSTIVE 0
#define FASTGA_TRI_NBRS 12
namespace FastGA {
// TODO This should be like an Abstract class
/**
 * This is the base class of the Gaussian Accumulator.
 * GaussianAccumulatorKD, GaussianAccumulatorOpt, and GaussianAccumualtorS2
 * will derive from this class. Unfortunately those classes have small differences
 * causing some unnecessary members in here that basically occurred as these classes
 * were created and changed over time. Eventually I will rewrite this whole thing such
 * that only the bare essentials are in this class.
 * @tparam T 
 */
template <class T>
class GaussianAccumulator
{

  public:
    /** @brief The underlying sphere-like mesh of the Gaussian Accumulator */
    Ico::IcoMesh mesh;
    /** @brief The buckets in the histogram, corresponding to cells/triangles on the mesh */
    std::vector<Bucket<T>> buckets;
    /** @brief A mask which indicates which triangles in the mesh are included in the buckets
     * By default its every one (mask = ones). This was added because I thought a user might want to limit
     * the histogram to only include triangles a max_phi from the north pole. 
     */
    std::vector<uint8_t> mask;
    /** @brief Only a valid member for GaussianAccumulatorOpt */
    Helper::BBOX projected_bbox;
    size_t num_buckets;
    GaussianAccumulator();
    GaussianAccumulator(const int level = FASTGA_LEVEL, const double max_phi = FASTGA_MAX_PHI);
    MatX3d GetBucketNormals(const bool reverse_sort=false);
    std::vector<double> GetNormalizedBucketCounts(const bool reverse_sort=false);
    std::vector<double> GetNormalizedBucketCountsByVertex(const bool reverse_sort=false);
    std::vector<T> GetBucketIndices();
    MatX2d GetBucketProjection();
    Ico::IcoMesh CopyIcoMesh(const bool reverse_sort=false);
    void ClearCount();

  protected:
    std::vector<size_t> sort_idx;
    void SortBucketsByIndices();
};

class GaussianAccumulatorKD : public GaussianAccumulator<uint32_t>
{

  public:
    GaussianAccumulatorKD(const int level = FASTGA_LEVEL, const double max_phi = FASTGA_MAX_PHI, const size_t max_leaf_size = FASTGA_MAX_LEAF_SIZE);
    std::vector<size_t> Integrate(const MatX3d& normals, const float eps = FASTGA_KDTREE_EPS);

  protected:
    const NFA::BUCKET2KD bucket2kd; // The adaptor
    const nanoflann::KDTreeSingleIndexAdaptorParams index_params;
    std::unique_ptr<NFA::nano_kd_tree_t> kd_tree_ptr;
};

// This Class is not very useful anymore
class GaussianAccumulatorOpt : public GaussianAccumulator<uint32_t>
{

  public:
    std::vector<uint32_t> bucket_hv;
    MatX12I bucket_neighbors;
    Regression regression;
    GaussianAccumulatorOpt(const int level = FASTGA_LEVEL, const double max_phi = FASTGA_MAX_PHI);
    std::vector<size_t> Integrate(const MatX3d& normals, const int num_nbr = FASTGA_TRI_NBRS);
    // std::vector<size_t> Integrate2(const MatX3d &normals, const int num_nbr = FASTGA_TRI_NBRS);

  protected:
};

class GaussianAccumulatorS2 : public GaussianAccumulator<uint64_t>
{

  public:
    std::vector<uint64_t> bucket_hv;
    MatX12I bucket_neighbors;
    Regression regression;
    GaussianAccumulatorS2(const int level = FASTGA_LEVEL, const double max_phi = FASTGA_MAX_PHI);
    std::vector<size_t> Integrate(const MatX3d& normals, const int num_nbr = FASTGA_TRI_NBRS);

  protected:
};

} // namespace FastGA
#endif
