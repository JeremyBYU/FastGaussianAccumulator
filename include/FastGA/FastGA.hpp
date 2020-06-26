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
 * GaussianAccumulatorKD, GaussianAccumulatorOpt, and GaussianAccumulatorS2
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
    /** @brief Only a valid member for GaussianAccumulatorOpt, ignore for everthing else */
    Helper::BBOX projected_bbox;
    /** @brief The number of buckets in histogram, size(buckets) */
    size_t num_buckets;
    GaussianAccumulator();
    /**
     * @brief Construct a new Gaussian Accumulator Object
     * 
     * @param level         The refinement level. Iterations of recursive subdivision.
     * @param max_phi       Latitude degrees from north pole to south pole to include cells in the mesh. Leave at default
     *                      which includes all of the mesh.
     */
    GaussianAccumulator(const int level = FASTGA_LEVEL, const double max_phi = FASTGA_MAX_PHI);
    /**
     * @brief Gets the surface normals of the buckets in the histogram.
     *        The order by default is sorted by the space filling curve value attached to each cell.
     * 
     * @param mesh_order    If true will return in the *actual* order of triangles of the underlying original mesh object. Useful for visualization.       
     * @return MatX3d 
     */
    MatX3d GetBucketNormals(const bool mesh_order=false);

    /**
     * @brief Get the normalized bucket counts in the histogram.
     *        The order by default is sorted by the space filling curve value attached to each cell.
     * 
     * @param mesh_order    If true will return in the *actual* order of triangles of the underlying original mesh object. Useful for visualization.   
     * @return std::vector<double> 
     */
    std::vector<double> GetNormalizedBucketCounts(const bool mesh_order=false);

    /**
     * @brief Average the normalized buckets counts (triangles) into the *vertices* of the mesh.
     *        The order by default is sorted by the space filling curve value attached to each cell.
     * 
     * @param mesh_order    If true will return in the *actual* order of vertices of the underlying original mesh object. Useful for visualization.   
     * @return std::vector<double> 
     */
    std::vector<double> GetNormalizedBucketCountsByVertex(const bool mesh_order=false);

    /**
     * @brief Get the space filling curve values of each bucket. Will be sorted low to high.
     * 
     * @return std::vector<T> 
     */
    std::vector<T> GetBucketSFCValues();

    /**
     * @brief Only useful for GaussianAccumulatorOpt. Return the XY projection of each bucket. 
     * 
     * @return MatX2d 
     */
    MatX2d GetBucketProjection();

    /**
     * @brief Creates a copy of the ico mesh.
     * 
     * @param mesh_order    If true will return in the original ico mesh before any sortign occurred. 
     * @return Ico::IcoMesh 
     */
    Ico::IcoMesh CopyIcoMesh(const bool mesh_order=false);

    /**
     * @brief Clears all the histogram counts for each cell. Useful to call after peak detection to "reset" the mesh.
     * 
     */
    void ClearCount();

  protected:

    /**
     * @brief This member variable keeps track of any sorting that occurred on the mesh and buckets.
     *        It basically allows us to reverse any sorting performed.
     * 
     */
    std::vector<size_t> sort_idx;

    void SortBucketsByIndices();
};

/**
 * @brief This implements the Gaussian Accumulator with K-D Tree search 
 * 
 */
class GaussianAccumulatorKD : public GaussianAccumulator<uint32_t>
{

  public:
    /**
     * @brief Construct a new GaussianAccumulatorKD object. This implements the GaussianAccumulator using a k-d tree method.
     *        Its actually pretty fast and can be used, though I recommend GaussianAccumulatorS2.
     * 
     * @param level         The refinement level. Iterations of recursive subdivision.
     * @param max_phi       Latitude degrees from north pole to south pole to include cells in the mesh. Leave at default
     *                      which includes all of the mesh.
     * @param max_leaf_size The max leaf size when building the k-d tre index. Leave at default, its optimal. 
     */
    GaussianAccumulatorKD(const int level = FASTGA_LEVEL, const double max_phi = FASTGA_MAX_PHI, const size_t max_leaf_size = FASTGA_MAX_LEAF_SIZE);
    std::vector<size_t> Integrate(const MatX3d& normals, const float eps = FASTGA_KDTREE_EPS);

  protected:
    /** @brief Used for nanoflann */
    const NFA::BUCKET2KD bucket2kd;
    /** @brief Used for nanoflann */
    const nanoflann::KDTreeSingleIndexAdaptorParams index_params;
    /** @brief The nanoflann k-d tree */
    std::unique_ptr<NFA::nano_kd_tree_t> kd_tree_ptr;
};

/**
 * @brief Construct a new GaussianAccumulatorOpt object. Do **not** use this class. It was my first design
 *        and only works well on the top hemisphere of a sphere. It uses a single projection (Azimuth Equal Area Projection)
 *        to project to a 2D plane. A hilbert curve is performed on the plane to greate the SFC on the sphere.
 *        This class is the reason that the `GaussianAccumulator` base class is such a mess because it began 
 *        with the assumptions built into this class. Eventually this will be deprecated.
 */
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

/**
 * @brief This GaussianAccumulator can handle the entire sphere by using a space filling curve designed
 *        by Google's S2 Geometry library. It projects a sphere to the faces of a cube and creates
 *        six separate hilbert curves for each face. It then stitches these curves together into one
 *        continuous thread. This class does not need S2 Geometry Library. We are using a port callsed s2nano
 *        that pulls out the essential SFC routine.
 * 
 *        It basically works by converting a normal to being integrated into a s2_id (SFC unique integer). 
 *        It performs a faster interpolated and branchless binary search to find the closest cell in buckets. 
 *        It then performs a local neighborhood search centered around the cell which actually looks at the surface normal.
 */
class GaussianAccumulatorS2 : public GaussianAccumulator<uint64_t>
{

  public:
    /** @brief Sorted list of space filling curves values for each bucket */
    std::vector<uint64_t> bucket_hv;
    /** @brief A NX12 matrix that contains the indices for each triangle neighbor */
    MatX12I bucket_neighbors;

    /**
     * @brief Construct a new Gaussian Accumulator S2 object
     * 
     * @param level         The refinement level. Iterations of recursive subdivision.
     * @param max_phi       Latitude degrees from north pole to south pole to include cells in the mesh. Leave at default
     *                      which includes all of the mesh.
     */
    GaussianAccumulatorS2(const int level = FASTGA_LEVEL, const double max_phi = FASTGA_MAX_PHI);

    /**
     * @brief Integrates a list of normals into the histogram.
     * 
     * @param normals               The unit normals to integrate
     * @param num_nbr               The number of neighbors to search during local neighborhood search. Leave at default.
     * @return std::vector<size_t> 
     */
    std::vector<size_t> Integrate(const MatX3d& normals, const int num_nbr = FASTGA_TRI_NBRS);

  protected:
    /** @brief A regressed line that aids to speed up sorted integer search inside bucket_hv */
    Regression regression;
};

} // namespace FastGA
#endif
