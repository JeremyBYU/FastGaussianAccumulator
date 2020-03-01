#include "FastGA.hpp"
#include <algorithm>

namespace FastGA {

GaussianAccumulator::GaussianAccumulator(const int level, const double max_phi) : mesh(), buckets(), projected_bbox()
{
    // Create refined mesh of the icosahedron
    mesh = FastGA::Ico::RefineIcosahedron(level);
    auto max_phi_rad = degreesToRadians(max_phi);
    auto min_z = std::cos(max_phi_rad);

    // Create buckets from refined icosahedron mesh
    std::transform(mesh.triangle_normals.begin(), mesh.triangle_normals.end(), std::back_inserter(buckets),
                   [](std::array<double, 3>& normal) -> Bucket { return {normal, 0, 0, {0, 0}}; });

    // Remove buckets whose phi (related to min_z) is too great
    // This wouldn't be necessary if there was a std::transform_if, or just used for loop
    // TODO switch to basic for loop?
    buckets.erase(std::remove_if(buckets.begin(), buckets.end(),
                                 [&min_z](Bucket& b) { return b.normal[2] < min_z; }),
                  buckets.end());

    // Get projected coordinates of these buckets
    projected_bbox = Helper::InitializeProjection(buckets);
    // Compute Hilbert Values for these buckets
    auto x_range = projected_bbox.max_x - projected_bbox.min_x;
    auto y_range = projected_bbox.max_y - projected_bbox.min_y;
    std::array<uint32_t, 2> xy_int;
    for(auto &bucket: buckets)
    {
        auto &projection = bucket.projection;
        Helper::ScaleXYToUInt32(&(projection[0]), xy_int.data(), projected_bbox.min_x, projected_bbox.min_y, x_range, y_range);
        bucket.hilbert_value = Hilbert::hilbertXYToIndex(16u, xy_int[0], xy_int[1]);
    }
}

std::vector<double > GaussianAccumulator::GetNormalizedBucketCounts()
{
    std::vector<double> normalized_counts(buckets.size());
    auto max_elem = std::max_element(buckets.begin(), buckets.end(), [](const Bucket &lhs, const Bucket &rhs) {return lhs.count < rhs.count;});
    auto max_count = max_elem->count;
    for (size_t i = 0; i < buckets.size(); i++)
    {
        normalized_counts[i] = static_cast<double>(buckets[i].count / max_count);
    }
    return normalized_counts;
}


// And this is the "dataset to kd-tree" adaptor class:
// GaussianAccumulatorKD::GaussianAccumulatorKD(const int level, const double max_phi, const size_t max_leaf_size) : GaussianAccumulator(level, max_phi), index_params(max_leaf_size)
GaussianAccumulatorKD::GaussianAccumulatorKD(const int level, const double max_phi, const size_t max_leaf_size) : GaussianAccumulator(level, max_phi), bucket2kd(buckets), index_params(max_leaf_size), kd_tree_ptr()
{
    kd_tree_ptr = std::make_unique<NFA::nano_kd_tree_t>(3, bucket2kd, index_params);
    kd_tree_ptr->buildIndex();
}

std::vector<size_t> GaussianAccumulatorKD::Integrate(const MatX3d &normals, const float eps)
{
    std::vector<size_t> bucket_indexes(normals.size());
    // Parameters for KNN Search
    const size_t num_results = 1;
	size_t ret_index = 0;
	double out_dist_sqr;
	nanoflann::KNNResultSet<double> resultSet(num_results);
    auto searchParams = nanoflann::SearchParams(0, eps);
    // Loop KD Tree search for every normal
    for (size_t i = 0; i < normals.size(); i++)
    {
	    resultSet.init(&ret_index, &out_dist_sqr ); // must reset result set
	    bool complete = kd_tree_ptr->findNeighbors(resultSet, &normals[i][0], searchParams);
        if (complete)
        {
            bucket_indexes[i] = ret_index;
            buckets[ret_index].count += 1;
        }
    }
    return bucket_indexes;
}






} // namespace FastGA