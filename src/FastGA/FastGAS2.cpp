#include "FastGA/FastGAS2.hpp"
#include <algorithm>

namespace FastGA {

GaussianAccumulatorS2::GaussianAccumulatorS2(const int level, const double max_phi) : GaussianAccumulator<uint64_t>(level, max_phi), bucket_neighbors()
{
    // assign values
    for(size_t i =0; i < buckets.size(); i++)
    {
        auto& normal = buckets[i].normal;
        S2Point s2_point(normal[0], normal[1], normal[2]);
        S2CellId s2_id(s2_point);
        auto id = s2_id.id();
        buckets[i].hilbert_value = id;
    }
    // Sort buckets and triangles by their unique index (hilbert curve value)
    auto sort_idx = Helper::sort_permutation(buckets, [](Bucket<uint64_t> const& a, Bucket<uint64_t> const& b) { return a.hilbert_value < b.hilbert_value; });
    Helper::ApplyPermutationInPlace(buckets, sort_idx);
    Helper::ApplyPermutationInPlace(mesh.triangle_normals, sort_idx);
    Helper::ApplyPermutationInPlace(mesh.triangles, sort_idx);

    bucket_neighbors = Ico::ComputeTriangleNeighbors(mesh.triangles, mesh.triangle_normals, num_buckets);

}

std::vector<size_t> GaussianAccumulatorS2::Integrate(const MatX3d& normals, const int num_nbr)
{
    std::vector<size_t> bucket_indexes(normals.size());


    Bucket<uint64_t> to_find = {{0, 0, 0}, 0, 0, {0, 0}};
    auto centered_tri_iter = buckets.begin();
    size_t centered_tri_idx = 0;
    size_t best_bucket_idx = 0;
    double best_bucket_dist = 10.0;
    size_t max_limit = std::numeric_limits<size_t>::max();

    for (size_t i = 0; i < normals.size(); i++)
    {
        auto& normal = normals[i];
        S2Point s2_point(normal[0], normal[1], normal[2]);
        S2CellId s2_id(s2_point);
        auto hv = s2_id.id();
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

template class FastGA::GaussianAccumulator<uint64_t>;

}