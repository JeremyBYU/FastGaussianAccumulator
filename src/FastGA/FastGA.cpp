#include "FastGA.hpp"
// #include "FastGAS.hpp"
#include <algorithm>

namespace FastGA {

std::ostream& operator<<(std::ostream& out, const std::array<double, 3>& v)
{
    if (!v.empty())
    {
        out << '[';
        std::copy(v.begin(), v.end(), std::ostream_iterator<double>(out, ", "));
        out << "\b\b]";
    }
    return out;
}

template <class T>
GaussianAccumulator<T>::GaussianAccumulator(const int level, const double max_phi) : mesh(), buckets(), mask(), projected_bbox(), num_buckets(0)
{
    // Create refined mesh of the icosahedron
    mesh = FastGA::Ico::RefineIcosahedron(level);
    auto max_phi_rad = degreesToRadians(max_phi);
    auto min_z = std::cos(max_phi_rad);

    // Indicates which triangle normals are part of buckets
    mask.resize(mesh.triangle_normals.size(), 0);
    // Create the angle buckets which are no greater than phi
    buckets.reserve(mesh.triangle_normals.size());
    for (size_t i = 0; i < mesh.triangle_normals.size(); i++)
    {
        if (mesh.triangle_normals[i][2] >= min_z)
        {
            buckets.push_back({mesh.triangle_normals[i], 0, 0, {0, 0}});
            mask[i] = 1;
        }
    }
    // Put all valid triangles (mask == 1) in the front of the list
    mesh.triangle_normals = Helper::BubbleDownMask(mesh.triangle_normals, mask);
    mesh.triangles = Helper::BubbleDownMask(mesh.triangles, mask);

    num_buckets = buckets.size();
}

template <class T>
MatX3d GaussianAccumulator<T>::GetBucketNormals()
{
    MatX3d bucket_normals;
    bucket_normals.reserve(buckets.size());
    std::transform(buckets.begin(), buckets.end(), std::back_inserter(bucket_normals),
                   [](const Bucket<T>& bucket) -> std::array<double, 3> { return bucket.normal; });
    return bucket_normals;
}

template <class T>
std::vector<double> GaussianAccumulator<T>::GetNormalizedBucketCounts()
{
    std::vector<double> normalized_counts(buckets.size());
    auto max_elem = std::max_element(buckets.begin(), buckets.end(), [](const Bucket<T>& lhs, const Bucket<T>& rhs) { return lhs.count < rhs.count; });
    auto max_count = max_elem->count;
    // std::cout << "Max Count: " << max_count << std::endl;
    for (size_t i = 0; i < buckets.size(); i++)
    {
        normalized_counts[i] = static_cast<double>(buckets[i].count / static_cast<double>(max_count));
    }
    return normalized_counts;
}

template <class T>
std::vector<T> GaussianAccumulator<T>::GetBucketIndices()
{
    std::vector<T> bucket_indices;
    bucket_indices.reserve(buckets.size());
    std::transform(buckets.begin(), buckets.end(), std::back_inserter(bucket_indices),
                   [](const Bucket<T>& bucket) -> T { return bucket.hilbert_value; });
    return bucket_indices;
}

template <class T>
MatX2d GaussianAccumulator<T>::GetBucketProjection()
{
    std::vector<std::array<double, 2>> bucket_projection;
    bucket_projection.reserve(buckets.size());
    std::transform(buckets.begin(), buckets.end(), std::back_inserter(bucket_projection),
                   [](const Bucket<T>& bucket) -> std::array<double, 2> { return bucket.projection; });
    return bucket_projection;
}

template <class T>
void GaussianAccumulator<T>::ClearCount()
{
    for (Bucket<T>& bucket : buckets)
    {
        bucket.count = 0;
    }
}

GaussianAccumulatorKD::GaussianAccumulatorKD(const int level, const double max_phi, const size_t max_leaf_size) : GaussianAccumulator<uint32_t>(level, max_phi), bucket2kd(buckets), index_params(max_leaf_size), kd_tree_ptr()
{
    // Get projected coordinates of these buckets
    projected_bbox = Helper::InitializeProjection(buckets);
    // Compute Hilbert Values for these buckets
    auto x_range = projected_bbox.max_x - projected_bbox.min_x;
    auto y_range = projected_bbox.max_y - projected_bbox.min_y;
    // std::cout << x_range << ", " << y_range << ", " << projected_bbox.min_x << std::endl;
    std::array<uint32_t, 2> xy_int;
    for (auto& bucket : buckets)
    {
        auto& projection = bucket.projection;
        Helper::ScaleXYToUInt32(&(projection[0]), xy_int.data(), projected_bbox.min_x, projected_bbox.min_y, x_range, y_range);
        // std::cout << "Int Proj: " << xy_int[0] << ", " << xy_int[1] <<std::endl;;
        bucket.hilbert_value = static_cast<uint32_t>(Hilbert::hilbertXYToIndex(16u, xy_int[0], xy_int[1]));
    }
    // Sort buckets and triangles by their unique index (hilbert curve value)
    auto sort_idx = Helper::sort_permutation(buckets, [](Bucket<uint32_t> const& a, Bucket<uint32_t> const& b) { return a.hilbert_value < b.hilbert_value; });
    Helper::ApplyPermutationInPlace(buckets, sort_idx);
    Helper::ApplyPermutationInPlace(mesh.triangle_normals, sort_idx);
    Helper::ApplyPermutationInPlace(mesh.triangles, sort_idx);
    // Index KD Tree
    kd_tree_ptr = std::make_unique<NFA::nano_kd_tree_t>(3, bucket2kd, index_params);
    kd_tree_ptr->buildIndex();
}

std::vector<size_t> GaussianAccumulatorKD::Integrate(const MatX3d& normals, const float eps)
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
        resultSet.init(&ret_index, &out_dist_sqr); // must reset result set
        bool complete = kd_tree_ptr->findNeighbors(resultSet, &normals[i][0], searchParams);
        if (complete)
        {
            bucket_indexes[i] = ret_index;
            buckets[ret_index].count += 1;
        }
        else
        {
            std::cerr << "Couldn't find normal in kdtree: " << normals[i] << std::endl;
        }
    }
    return bucket_indexes;
}

// Optimal Search
GaussianAccumulatorOpt::GaussianAccumulatorOpt(const int level, const double max_phi) : GaussianAccumulator<uint32_t>(level, max_phi), bucket_neighbors()
{
    // Get projected coordinates of these buckets
    projected_bbox = Helper::InitializeProjection(buckets);
    // Compute Hilbert Values for these buckets
    auto x_range = projected_bbox.max_x - projected_bbox.min_x;
    auto y_range = projected_bbox.max_y - projected_bbox.min_y;
    // std::cout << x_range << ", " << y_range << ", " << projected_bbox.min_x << std::endl;
    std::array<uint32_t, 2> xy_int;
    for (auto& bucket : buckets)
    {
        auto& projection = bucket.projection;
        Helper::ScaleXYToUInt32(&(projection[0]), xy_int.data(), projected_bbox.min_x, projected_bbox.min_y, x_range, y_range);
        // std::cout << "Int Proj: " << xy_int[0] << ", " << xy_int[1] <<std::endl;;
        bucket.hilbert_value = static_cast<uint32_t>(Hilbert::hilbertXYToIndex(16u, xy_int[0], xy_int[1]));
    }
    // Sort buckets and triangles by their unique index (hilbert curve value)
    auto sort_idx = Helper::sort_permutation(buckets, [](Bucket<uint32_t> const& a, Bucket<uint32_t> const& b) { return a.hilbert_value < b.hilbert_value; });
    Helper::ApplyPermutationInPlace(buckets, sort_idx);
    Helper::ApplyPermutationInPlace(mesh.triangle_normals, sort_idx);
    Helper::ApplyPermutationInPlace(mesh.triangles, sort_idx);

    bucket_neighbors = Ico::ComputeTriangleNeighbors(mesh.triangles, mesh.triangle_normals, num_buckets);
}

std::vector<size_t> GaussianAccumulatorOpt::Integrate(const MatX3d& normals, const int num_nbr)
{
    std::vector<size_t> bucket_indexes(normals.size());
    MatX2d projection;
    std::vector<uint32_t> hilbert_values;
    std::tie(projection, hilbert_values) = Helper::ConvertNormalsToHilbert(normals, projected_bbox);

    Bucket<uint32_t> to_find = {{0, 0, 0}, 0, 0, {0, 0}};
    auto centered_tri_iter = buckets.begin();
    size_t centered_tri_idx = 0;
    size_t best_bucket_idx = 0;
    double best_bucket_dist = 10.0;
    size_t max_limit = std::numeric_limits<size_t>::max();

    for (size_t i = 0; i < normals.size(); i++)
    {
        auto& hv = hilbert_values[i];
        auto& normal = normals[i];
        to_find.hilbert_value = hv;
        auto upper_idx = std::upper_bound(buckets.begin(), buckets.end(), to_find);
        auto lower_idx = upper_idx - 1;
        if (upper_idx == buckets.end())
            upper_idx = buckets.begin();

        auto lower_dist = Helper::SquaredDistance(normal, lower_idx->normal);
        auto upper_dist = Helper::SquaredDistance(normal, upper_idx->normal);
        // Best idx chosen
        centered_tri_iter = lower_idx;
        best_bucket_dist = lower_dist;
        // min_distances[0] = lower_dist;
        if (lower_dist > upper_dist)
        {
            centered_tri_iter = upper_idx;
            best_bucket_dist = upper_dist;
            // min_distances[0] = upper_dist;
        }
        centered_tri_idx = std::distance(buckets.begin(), centered_tri_iter);
        best_bucket_idx = centered_tri_idx;

        for (int nbr_counter = 0; nbr_counter < num_nbr; nbr_counter++)
        {
            auto& bucket_nbr_idx = bucket_neighbors[centered_tri_idx][nbr_counter];
            if (bucket_nbr_idx == max_limit)
                break;
            auto dist = Helper::SquaredDistance(normal, buckets[bucket_nbr_idx].normal);
            if (dist < best_bucket_dist)
            {
                best_bucket_dist = dist;
                best_bucket_idx = bucket_nbr_idx;
                // std::cout << "Found a better" << std::endl;
            }
        }

        buckets[best_bucket_idx].count += 1;
        bucket_indexes[i] = best_bucket_idx;
    }
    return bucket_indexes;
}

template class FastGA::GaussianAccumulator<uint32_t>;
template class FastGA::GaussianAccumulator<uint64_t>;

} // namespace FastGA