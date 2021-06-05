#include "FastGA.hpp"
// #include "FastGAS.hpp"
#include <algorithm>

#include "fastcluster.h"

namespace FastGA {

template <class T>
GaussianAccumulator<T>::GaussianAccumulator(const int level, const double max_phi) : mesh(), buckets(), mask(), projected_bbox(), num_buckets(0), sort_idx()
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
MatX3d GaussianAccumulator<T>::GetBucketNormals(const bool mesh_order)
{
    MatX3d bucket_normals;
    bucket_normals.reserve(buckets.size());
    std::transform(buckets.begin(), buckets.end(), std::back_inserter(bucket_normals),
                   [](const Bucket<T>& bucket) -> std::array<double, 3> { return bucket.normal; });

    if (!mesh_order)
        return bucket_normals;
    
    auto reversed_sort_idx  = Helper::sort_permutation(sort_idx, std::less<size_t>());
    auto new_bucket_normals = Helper::ApplyPermutation(bucket_normals, reversed_sort_idx);
    return new_bucket_normals;
}

template <class T>
std::vector<double> GaussianAccumulator<T>::GetNormalizedBucketCounts(const bool mesh_order)
{
    std::vector<double> normalized_counts(buckets.size());
    auto max_elem = std::max_element(buckets.begin(), buckets.end(), [](const Bucket<T>& lhs, const Bucket<T>& rhs) { return lhs.count < rhs.count; });
    auto max_count = max_elem->count;
    // std::cout << "Max Count: " << max_count << std::endl;
    for (size_t i = 0; i < buckets.size(); i++)
    {
        normalized_counts[i] = static_cast<double>(buckets[i].count / static_cast<double>(max_count));
    }

    if (!mesh_order)
        return normalized_counts;

    auto reversed_sort_idx  = Helper::sort_permutation(sort_idx, std::less<size_t>());
    auto new_normalized_counts = Helper::ApplyPermutation(normalized_counts, reversed_sort_idx);
    return new_normalized_counts;
}

template <class T>
std::vector<double> GaussianAccumulator<T>::GetNormalizedBucketCountsByVertex(const bool mesh_order)
{
    auto normalized_bucket_counts = GetNormalizedBucketCounts(mesh_order);
    auto normalized_bucket_counts_by_vertex = Helper::Mean(mesh.adjacency_matrix, normalized_bucket_counts);
    double max_elem = *std::max_element(normalized_bucket_counts_by_vertex.begin(), normalized_bucket_counts_by_vertex.end(), std::less<double>());
    // std::cout << "Max Count: " << max_count << std::endl;
    for (size_t i = 0; i < normalized_bucket_counts_by_vertex.size(); i++)
    {
        normalized_bucket_counts_by_vertex[i] = normalized_bucket_counts_by_vertex[i] / max_elem;
    }
    return normalized_bucket_counts_by_vertex;
}

template <class T>
std::vector<T> GaussianAccumulator<T>::GetBucketSFCValues()
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

template <class T>
Ico::IcoMesh GaussianAccumulator<T>::CopyIcoMesh(const bool mesh_order)
{
    if (!mesh_order)
        return mesh;
    auto reversed_sort_idx  = Helper::sort_permutation(sort_idx, std::less<size_t>());
    auto new_triangle_normals = Helper::ApplyPermutation(mesh.triangle_normals, reversed_sort_idx);
    auto new_triangles = Helper::ApplyPermutation(mesh.triangles, reversed_sort_idx);
    Ico::IcoMesh new_mesh;
    new_mesh.triangle_normals = new_triangle_normals;
    new_mesh.triangles = new_triangles;
    new_mesh.vertices = mesh.vertices;

    return new_mesh;
}


GaussianAccumulatorKD::GaussianAccumulatorKD(const int level, const double max_phi, const size_t max_leaf_size) : GaussianAccumulator<uint32_t>(level, max_phi), bucket2kd(buckets), index_params(max_leaf_size), kd_tree_ptr()
{
    // Note that we will be sorting the buckets by their hilbert curve values
    // This is **not** necessary for k-d tree search, the order doesn't matter at all when the k-d tree index is built (last line of this function)
    // However we do this just to keep it consistent with the other classes that. I'm not sure if I rely upon this for visualization code.

    // Get projected coordinates of these buckets
    projected_bbox = Helper::InitializeProjection(buckets);
    // Compute Hilbert Values for these buckets
    auto x_range = projected_bbox.max_x - projected_bbox.min_x;
    auto y_range = projected_bbox.max_y - projected_bbox.min_y;
    std::array<uint32_t, 2> xy_int;
    for (auto& bucket : buckets)
    {
        auto& projection = bucket.projection;
        Helper::ScaleXYToUInt32(&(projection[0]), xy_int.data(), projected_bbox.min_x, projected_bbox.min_y, x_range, y_range);
        // std::cout << "Int Proj: " << xy_int[0] << ", " << xy_int[1] <<std::endl;;
        bucket.hilbert_value = static_cast<uint32_t>(Hilbert::hilbertXYToIndex(16u, xy_int[0], xy_int[1]));
    }
    // Sort buckets and triangles by their unique index (hilbert curve value)
    sort_idx = Helper::sort_permutation(buckets, [](Bucket<uint32_t> const& a, Bucket<uint32_t> const& b) { return a.hilbert_value < b.hilbert_value; });
    Helper::ApplyPermutationInPlace(buckets, sort_idx);
    Helper::ApplyPermutationInPlace(mesh.triangle_normals, sort_idx);
    Helper::ApplyPermutationInPlace(mesh.triangles, sort_idx);

    // Index k-d tree, this is what actually matters...
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

GaussianAccumulatorOpt::GaussianAccumulatorOpt(const int level, const double max_phi) : GaussianAccumulator<uint32_t>(level, max_phi), bucket_hv(), bucket_neighbors(), regression()
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
    sort_idx = Helper::sort_permutation(buckets, [](Bucket<uint32_t> const& a, Bucket<uint32_t> const& b) { return a.hilbert_value < b.hilbert_value; });
    Helper::ApplyPermutationInPlace(buckets, sort_idx);
    Helper::ApplyPermutationInPlace(mesh.triangle_normals, sort_idx);
    Helper::ApplyPermutationInPlace(mesh.triangles, sort_idx);

    bucket_hv = this->GetBucketSFCValues();
    bucket_neighbors = Ico::ComputeTriangleNeighbors(mesh.triangles, mesh.triangle_normals, num_buckets);

    std::vector<size_t> seq(bucket_hv.size());
    std::for_each(seq.begin(), seq.end(), [i = 0](size_t& x) mutable { x = i++; });
    Helper::LinearRegression(bucket_hv, seq, regression);
}

template <class T>
inline size_t LocalSearch(size_t lower_idx, size_t upper_idx, const std::array<double, 3>& normal,
                          std::vector<Bucket<T>>& buckets, MatX12I& bucket_neighbors, const int& num_nbr)
{

    auto lower_dist = Helper::SquaredDistance(normal, buckets[lower_idx].normal);
    auto upper_dist = Helper::SquaredDistance(normal, buckets[upper_idx].normal);
    // Best idx chosen
    auto centered_tri_idx = upper_dist > lower_dist ? lower_idx : upper_idx;
    auto best_bucket_dist = upper_dist > lower_dist ? lower_dist : upper_dist;

    auto best_bucket_idx = centered_tri_idx;
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
        }
    }
    return best_bucket_idx;
}

template <class T>
inline void CalculateSearchBounds(T hv, int& start_index, int& end_index, int& size, Regression& regression)
{
    auto bucket_index = static_cast<int>(regression.intercept + regression.slope * static_cast<double>(hv));
    start_index = bucket_index - regression.subtract_index;
    start_index = start_index < 0 ? 0 : start_index;
    end_index = start_index + regression.power_of_2 >= size ? size : start_index + regression.power_of_2;
    start_index = end_index == size ? end_index - regression.power_of_2 : start_index;
}

std::vector<size_t> GaussianAccumulatorOpt::Integrate(const MatX3d& normals, const int num_nbr)
{
    std::vector<size_t> bucket_indexes(normals.size());
    MatX2d projection;
    std::vector<uint32_t> hilbert_values;
    std::tie(projection, hilbert_values) = Helper::ConvertNormalsToHilbert(normals, projected_bbox);

    int size = static_cast<int>(bucket_hv.size());
    auto start_index_int = 0;
    auto end_index_int = 0;

    for (size_t i = 0; i < normals.size(); i++)
    {
        auto& normal = normals[i];
        auto& hv = hilbert_values[i];
        // Reduce search bounds by linearly interpolating where the bucket should be given a hilbert value
        CalculateSearchBounds<uint32_t>(hv, start_index_int, end_index_int, size, regression);
        // this is a faster lower_bound
        auto upper_idx_int = FBS::binary_search_branchless(&bucket_hv[start_index_int], regression.power_of_2, hv) + start_index_int;
        auto lower_idx_int = upper_idx_int - 1;
        lower_idx_int = lower_idx_int < 0 ? 0 : lower_idx_int;
        upper_idx_int = upper_idx_int > size ? size - 1 : upper_idx_int;

        auto best_bucket_idx = LocalSearch<uint32_t>(lower_idx_int, upper_idx_int, normal, buckets, bucket_neighbors, num_nbr);

        buckets[best_bucket_idx].count += 1;
        bucket_indexes[i] = best_bucket_idx;
    }

    return bucket_indexes;
}

GaussianAccumulatorS2::GaussianAccumulatorS2(const int level, const double max_phi) : GaussianAccumulator<uint64_t>(level, max_phi), bucket_hv(), bucket_neighbors(), regression()
{
    // assign values
    for (size_t i = 0; i < buckets.size(); i++)
    {
        auto& normal = buckets[i].normal;
        buckets[i].hilbert_value = NanoS2ID::S2CellId(normal);
    }
    // Sort buckets and triangles by their unique index (hilbert curve value)
    sort_idx = Helper::sort_permutation(buckets, [](Bucket<uint64_t> const& a, Bucket<uint64_t> const& b) { return a.hilbert_value < b.hilbert_value; });
    Helper::ApplyPermutationInPlace(buckets, sort_idx);
    Helper::ApplyPermutationInPlace(mesh.triangle_normals, sort_idx);
    Helper::ApplyPermutationInPlace(mesh.triangles, sort_idx);

    bucket_hv = this->GetBucketSFCValues();
    bucket_neighbors = Ico::ComputeTriangleNeighbors(mesh.triangles, mesh.triangle_normals, num_buckets);
    // Just a sequence of integers
    std::vector<size_t> seq(bucket_hv.size());
    std::for_each(seq.begin(), seq.end(), [i = 0](size_t& x) mutable { x = i++; });
    Helper::LinearRegression(bucket_hv, seq, regression);
    // std::cout << "Negative Error:" << regression.neg_error << "; Positive Error: " << regression.pos_error << "; Subtract: " << regression.subtract_index << "; Power of 2: " << regression.power_of_2 <<  std::endl;
}

std::vector<size_t> GaussianAccumulatorS2::Integrate(const MatX3d& normals, const int num_nbr)
{
    size_t num_normals = normals.size();
    std::vector<size_t> bucket_indexes(num_normals);
    std::vector<uint64_t> hilbert_values(num_normals);

    int size = static_cast<int>(bucket_hv.size());
    for (size_t i = 0; i < num_normals; i++)
    {
        auto& normal = normals[i];
        hilbert_values[i] = NanoS2ID::S2CellId(normal);
    }

    auto start_index_int = 0;
    auto end_index_int = 0;
    // Integrate the normal into the bucket
    for (size_t i = 0; i < normals.size(); i++)
    {
        auto& normal = normals[i];
        auto& hv = hilbert_values[i];
        // Reduce search bounds by linearly interpolating where the bucket should be given a hilbert value
        CalculateSearchBounds<uint64_t>(hv, start_index_int, end_index_int, size, regression);
        // this is a faster lower_bound
        auto upper_idx_int = FBS::binary_search_branchless(&bucket_hv[start_index_int], regression.power_of_2, hv) + start_index_int;
        auto lower_idx_int = upper_idx_int - 1;
        lower_idx_int = lower_idx_int < 0 ? 0 : lower_idx_int;
        upper_idx_int = upper_idx_int > size ? size - 1 : upper_idx_int;

        auto best_bucket_idx = LocalSearch<uint64_t>(lower_idx_int, upper_idx_int, normal, buckets, bucket_neighbors, num_nbr);

        buckets[best_bucket_idx].count += 1;
        bucket_indexes[i] = best_bucket_idx;
    }
    return bucket_indexes;
}

template class FastGA::GaussianAccumulator<uint32_t>;
template class FastGA::GaussianAccumulator<uint64_t>;

///////////////////////
///////////////////////
// Start new S2Beta

GaussianAccumulatorS2Beta::GaussianAccumulatorS2Beta(const int level) : mesh(), buckets(), num_buckets(0),  bucket_hv(), bucket_neighbors(), ico_chart(level), sort_idx(), regression()
{

    // Create refined mesh of the icosahedron
    mesh = FastGA::Ico::RefineIcosahedron(level);
    // Create the angle buckets which are no greater than phi
    buckets.reserve(mesh.triangle_normals.size());
    for (size_t i = 0; i < mesh.triangle_normals.size(); i++)
    {
        buckets.push_back({mesh.triangle_normals[i], mesh.triangle_normals[i], 0, 0});

    }
    num_buckets = buckets.size();

    // assign values
    for (size_t i = 0; i < buckets.size(); i++)
    {
        auto& normal = buckets[i].normal;
        buckets[i].hilbert_value = NanoS2ID::S2CellId(normal);
    }
    // Sort buckets and triangles by their unique index (hilbert curve value)
    sort_idx = Helper::sort_permutation(buckets, [](BucketS2 const& a, BucketS2 const& b) { return a.hilbert_value < b.hilbert_value; });
    Helper::ApplyPermutationInPlace(buckets, sort_idx);
    Helper::ApplyPermutationInPlace(mesh.triangle_normals, sort_idx);
    Helper::ApplyPermutationInPlace(mesh.triangles, sort_idx);

    bucket_hv = this->GetBucketSFCValues();
    bucket_neighbors = Ico::ComputeTriangleNeighbors(mesh.triangles, mesh.triangle_normals, num_buckets);
    // Just a sequence of integers
    std::vector<size_t> seq(bucket_hv.size());
    std::for_each(seq.begin(), seq.end(), [i = 0](size_t& x) mutable { x = i++; });
    Helper::LinearRegression(bucket_hv, seq, regression);
    // std::cout << "Negative Error:" << regression.neg_error << "; Positive Error: " << regression.pos_error << "; Subtract: " << regression.subtract_index << "; Power of 2: " << regression.power_of_2 <<  std::endl;
}


inline size_t LocalSearchS2(size_t lower_idx, size_t upper_idx, const std::array<double, 3>& normal,
                          std::vector<BucketS2>& buckets, MatX12I& bucket_neighbors, const int& num_nbr)
{

    auto lower_dist = Helper::SquaredDistance(normal, buckets[lower_idx].normal);
    auto upper_dist = Helper::SquaredDistance(normal, buckets[upper_idx].normal);
    // Best idx chosen
    auto centered_tri_idx = upper_dist > lower_dist ? lower_idx : upper_idx;
    auto best_bucket_dist = upper_dist > lower_dist ? lower_dist : upper_dist;

    auto best_bucket_idx = centered_tri_idx;
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
        }
    }
    return best_bucket_idx;
}


void GaussianAccumulatorS2Beta::AverageBucketNormals()
{
        // assign values
    for (size_t i = 0; i < buckets.size(); ++i)
    {
        auto& bucket = buckets[i];
        double scale = 1.0 / (1.0 + static_cast<double>(bucket.count));
        Helper::ScaleItemInPlace<double>(bucket.average_normal, scale);
        Helper::normalize3(&bucket.average_normal[0]);
    }
}

std::vector<size_t> GaussianAccumulatorS2Beta::Integrate(const MatX3d& normals, const int num_nbr)
{
    size_t num_normals = normals.size();
    std::vector<size_t> bucket_indexes(num_normals);
    std::vector<uint64_t> hilbert_values(num_normals);

    int size = static_cast<int>(bucket_hv.size());
    for (size_t i = 0; i < num_normals; i++)
    {
        auto& normal = normals[i];
        hilbert_values[i] = NanoS2ID::S2CellId(normal);
    }

    auto start_index_int = 0;
    auto end_index_int = 0;
    // Integrate the normal into the bucket
    for (size_t i = 0; i < normals.size(); i++)
    {
        auto& normal = normals[i];
        auto& hv = hilbert_values[i];
        // Reduce search bounds by linearly interpolating where the bucket should be given a hilbert value
        CalculateSearchBounds<uint64_t>(hv, start_index_int, end_index_int, size, regression);
        // this is a faster lower_bound
        auto upper_idx_int = FBS::binary_search_branchless(&bucket_hv[start_index_int], regression.power_of_2, hv) + start_index_int;
        auto lower_idx_int = upper_idx_int - 1;
        lower_idx_int = lower_idx_int < 0 ? 0 : lower_idx_int;
        upper_idx_int = upper_idx_int > size ? size - 1 : upper_idx_int;

        auto best_bucket_idx = LocalSearchS2(lower_idx_int, upper_idx_int, normal, buckets, bucket_neighbors, num_nbr);

        buckets[best_bucket_idx].count += 1;
        bucket_indexes[i] = best_bucket_idx;
        Helper::InPlaceAdd<double, 3>(normal, buckets[best_bucket_idx].average_normal);
    }
    AverageBucketNormals();
    return bucket_indexes;
}


std::vector<uint64_t> GaussianAccumulatorS2Beta::GetBucketSFCValues()
{
    std::vector<uint64_t> bucket_indices;
    bucket_indices.reserve(buckets.size());
    std::transform(buckets.begin(), buckets.end(), std::back_inserter(bucket_indices),
                   [](const BucketS2& bucket) -> uint64_t { return bucket.hilbert_value; });
    return bucket_indices;
}


MatX3d GaussianAccumulatorS2Beta::GetBucketNormals(const bool mesh_order)
{
    MatX3d bucket_normals;
    bucket_normals.reserve(buckets.size());
    std::transform(buckets.begin(), buckets.end(), std::back_inserter(bucket_normals),
                   [](const BucketS2& bucket) -> std::array<double, 3> { return bucket.normal; });

    if (!mesh_order)
        return bucket_normals;
    
    auto reversed_sort_idx  = Helper::sort_permutation(sort_idx, std::less<size_t>());
    auto new_bucket_normals = Helper::ApplyPermutation(bucket_normals, reversed_sort_idx);
    return new_bucket_normals;
}

MatX3d GaussianAccumulatorS2Beta::GetBucketAverageNormals(const bool mesh_order)
{
    MatX3d bucket_normals;
    bucket_normals.reserve(buckets.size());
    std::transform(buckets.begin(), buckets.end(), std::back_inserter(bucket_normals),
                   [](const BucketS2& bucket) -> std::array<double, 3> { return bucket.average_normal; });

    if (!mesh_order)
        return bucket_normals;
    
    auto reversed_sort_idx  = Helper::sort_permutation(sort_idx, std::less<size_t>());
    auto new_bucket_normals = Helper::ApplyPermutation(bucket_normals, reversed_sort_idx);
    return new_bucket_normals;
}

std::vector<int> GaussianAccumulatorS2Beta::GetBucketCounts(const bool mesh_order)
{
    std::vector<int> bucket_counts(buckets.size());
    // std::cout << "Max Count: " << max_count << std::endl;
    for (size_t i = 0; i < buckets.size(); i++)
    {
        bucket_counts[i] = buckets[i].count;
    }

    if (!mesh_order)
        return bucket_counts;

    auto reversed_sort_idx  = Helper::sort_permutation(sort_idx, std::less<size_t>());
    auto new_normalized_counts = Helper::ApplyPermutation(bucket_counts, reversed_sort_idx);
    return new_normalized_counts;
}

std::vector<double> GaussianAccumulatorS2Beta::GetNormalizedBucketCounts(const bool mesh_order)
{
    std::vector<double> normalized_counts(buckets.size());
    auto max_elem = std::max_element(buckets.begin(), buckets.end(), [](const BucketS2& lhs, const BucketS2& rhs) { return lhs.count < rhs.count; });
    auto max_count = max_elem->count;
    // std::cout << "Max Count: " << max_count << std::endl;
    for (size_t i = 0; i < buckets.size(); i++)
    {
        normalized_counts[i] = static_cast<double>(buckets[i].count / static_cast<double>(max_count));
    }

    if (!mesh_order)
        return normalized_counts;

    auto reversed_sort_idx  = Helper::sort_permutation(sort_idx, std::less<size_t>());
    auto new_normalized_counts = Helper::ApplyPermutation(normalized_counts, reversed_sort_idx);
    return new_normalized_counts;
}

std::vector<double> GaussianAccumulatorS2Beta::GetNormalizedBucketCountsByVertex(const bool mesh_order)
{
    auto normalized_bucket_counts = GetNormalizedBucketCounts(mesh_order);
    auto normalized_bucket_counts_by_vertex = Helper::Mean(mesh.adjacency_matrix, normalized_bucket_counts);
    double max_elem = *std::max_element(normalized_bucket_counts_by_vertex.begin(), normalized_bucket_counts_by_vertex.end(), std::less<double>());
    // std::cout << "Max Count: " << max_count << std::endl;
    for (size_t i = 0; i < normalized_bucket_counts_by_vertex.size(); i++)
    {
        normalized_bucket_counts_by_vertex[i] = normalized_bucket_counts_by_vertex[i] / max_elem;
    }
    return normalized_bucket_counts_by_vertex;
}

MatX3d GaussianAccumulatorS2Beta::GetAverageNormalsByVertex(const bool mesh_order)
{
    // bucket == triangle
    // auto normalized_bucket_counts = GetBucketCounts(mesh_order); // 0-1 in normalized value for this bucket
    auto normalized_bucket_counts = GetNormalizedBucketCounts(mesh_order); // 0-1 in normalized value for this bucket
    auto bucket_average_normals = GetBucketAverageNormals(mesh_order);  // the average normal in this bucket
    auto vertex_average_normals = Helper::MeanAverageNormals(mesh.adjacency_matrix, bucket_average_normals, normalized_bucket_counts);
    return vertex_average_normals;
}

void GaussianAccumulatorS2Beta::ClearCount()
{
    for (BucketS2& bucket : buckets)
    {
        bucket.count = 0;
        bucket.average_normal = bucket.normal;
    }
}

Ico::IcoMesh GaussianAccumulatorS2Beta::CopyIcoMesh(const bool mesh_order)
{
    if (!mesh_order)
        return mesh;
    auto reversed_sort_idx  = Helper::sort_permutation(sort_idx, std::less<size_t>());
    auto new_triangle_normals = Helper::ApplyPermutation(mesh.triangle_normals, reversed_sort_idx);
    auto new_triangles = Helper::ApplyPermutation(mesh.triangles, reversed_sort_idx);
    Ico::IcoMesh new_mesh;
    new_mesh.triangle_normals = new_triangle_normals;
    new_mesh.triangles = new_triangles;
    new_mesh.vertices = mesh.vertices;

    return new_mesh;
}

std::vector<int> ClusterData(MatX3d &peaks, double t)
{
    auto npoints = static_cast<int>(peaks.size());
    if (npoints == 1)
    {
        return {0};
    }
    // computation of condensed distance matrix
    std::vector<double> distmat;
    // intermediate values
    std::vector<int> merge(2 * (npoints -1), 0);
    std::vector<double> height(npoints-1, 0.0);
    // final clusters (output)
    std::vector<int> labels(npoints, 0);
    // calculate distance matrix
    for (int i=0; i<npoints; ++i) {
        for (int j=i+1; j< npoints; j++) {
            auto dist = std::sqrt(Helper::SquaredDistance(peaks[i], peaks[j]));
            distmat.push_back(dist);
        }
    }

    // clustering call
    HClust::hclust_fast(npoints, distmat.data(), HClust::hclust_fast_methods::HCLUST_METHOD_SINGLE, merge.data(), height.data());
    HClust::cutree_cdist(npoints, merge.data(), height.data(), t, labels.data());

    return labels;
}
MatX3d AverageClusters(MatX3d &peaks, std::vector<double> &peak_weights, std::vector<int> &clusters, double min_total_weight)
{
    auto total_peaks = *std::max_element(clusters.begin(),clusters.end()) + 1;
    MatX3d average_peaks(total_peaks, {0.0, 0.0, 0.0});
    std::vector<double> peak_total_weight(total_peaks, 0.0);

    for (size_t i=0; i < peaks.size(); ++i)
    {
        auto &peak = peaks[i];
        auto &weight = peak_weights[i];
        auto &cluster = clusters[i];
        auto &average_peak_cluster = average_peaks[cluster];
        Helper::InPlaceAddScale<double, 3>(peak, average_peak_cluster, weight);
        peak_total_weight[cluster] += weight;
    }

    MatX3d average_peaks_filt;
    std::vector<double> peak_total_weight_filt;
    for (size_t i =0; i < average_peaks.size(); ++i)
    {
        auto &average_peak_cluster = average_peaks[i];
        auto &total_weight = peak_total_weight[i];
        Helper::InPlaceDivide<double, 3>(average_peak_cluster,total_weight );
        Helper::normalize3(average_peak_cluster.data());
        if (total_weight >= min_total_weight)
        {
            average_peaks_filt.push_back(average_peak_cluster);
            peak_total_weight_filt.push_back(total_weight);
        }
    }

    auto sort_idx  = Helper::sort_permutation(peak_total_weight_filt, std::greater<double>());
    auto sorted_average_peaks = Helper::ApplyPermutation(average_peaks_filt, sort_idx);

    return sorted_average_peaks;
}


MatX3d GaussianAccumulatorS2Beta::FindPeaks(uint8_t threshold_abs, bool exclude_border, double cluster_distance, double min_cluster_weight)
{
    std::unordered_set<int> hash;
    auto average_vertex_normals = GetAverageNormalsByVertex(true);
    auto normalized_bucket_counts_per_vertex = GetNormalizedBucketCountsByVertex(true);
    ico_chart.FillImage(normalized_bucket_counts_per_vertex);
    auto img_idx_peaks = ico_chart.FindPeaks(threshold_abs, exclude_border);
    MatX3d normal_peaks;
    std::vector<double> peak_values;
    for (auto &idx_peak: img_idx_peaks)
    {
        auto vertex_idx = *ico_chart.image_to_vertex_idx.PointerAt<int>(idx_peak[0], idx_peak[1]);
        if (hash.find(vertex_idx) == hash.end()) {
            hash.insert(vertex_idx);
            auto normal = average_vertex_normals[vertex_idx];
            auto value = normalized_bucket_counts_per_vertex[vertex_idx];
            normal_peaks.push_back(normal);
            peak_values.push_back(value);
        }
    }
    if (normal_peaks.size() == 0)
    {
        return {};
    }
    auto clusters = ClusterData(normal_peaks, cluster_distance);
    auto sorted_averaged_peaks = AverageClusters(normal_peaks, peak_values, clusters, min_cluster_weight);
    return sorted_averaged_peaks;

}





} // namespace FastGA