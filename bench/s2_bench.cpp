
#include <random>
#include <cmath>
#include <chrono>
#include <vector>
#include <iostream>
#include "FastGA.hpp"
#include "Hilbert/Hilbert.hpp"
#include <benchmark/benchmark.h>

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include "s2/s2earth.h"
#include "s2/s1chord_angle.h"
#include "s2/s2closest_point_query.h"
#include "s2/s2point_index.h"
#include "s2/s2testing.h"

FastGA::MatX3d initialize_normals(int N, double max_phi_degrees = 95, unsigned seed = 1)
{
    FastGA::MatX3d normals = FastGA::MatX3d(N);
    std::mt19937 generator = std::mt19937(seed);
    std::uniform_real_distribution<double> uniform01 = std::uniform_real_distribution<double>(0.0, 1.0);
    auto max_phi_radians = degreesToRadians(max_phi_degrees);
    for (int i = 0; i < N;)
    {
        double theta = 2 * M_PI * uniform01(generator);
        double phi = acos(1 - 2 * uniform01(generator));
        double x = sin(phi) * cos(theta);
        double y = sin(phi) * sin(theta);
        double z = cos(phi);
        if (phi > max_phi_radians)
            continue;
        normals[i] = {x, y, z};
        i++;
    }
    return normals;
}

static void BM_S2PointQuery_1 (benchmark::State& st)
{
    FastGA::GaussianAccumulatorKD GA = FastGA::GaussianAccumulatorKD(4, 100.0);
    auto normals = initialize_normals(100000);
    S2PointIndex<int> s2_index;
    // std::cout<< "Bucket Size: " << GA.buckets.size() << std::endl;
    for (size_t i = 0; i < GA.buckets.size(); i++)
    {
        auto& bucket_normal = GA.buckets[i].normal;
        // std::cout << "Indexing for: " << bucket_normal << std::endl;
        S2Point s2_point(bucket_normal[0], bucket_normal[1], bucket_normal[2]);
        s2_index.Add(s2_point, i);
    }
    S2ClosestPointQuery<int> query(&s2_index);
    query.mutable_options()->set_max_results(1);
    for (auto _ : st)
    {
        for (size_t i = 0; i < normals.size(); i++)
        {
            auto& normal = normals[i];
            S2Point s2_point(normal[0], normal[1], normal[2]);
            S2ClosestPointQuery<int>::PointTarget target(s2_point);
            auto results = query.FindClosestPoints(&target);
            // std::cout << results.size() << std::endl;
            benchmark::DoNotOptimize(results.data());
            benchmark::ClobberMemory();
        }
    }
}

static void BM_S2CellID (benchmark::State& st)
{
    auto normals = initialize_normals(100000);
    // std::cout<< "Bucket Size: " << GA.buckets.size() << std::endl;
    for (auto _ : st)
    {
        for (size_t i = 0; i < normals.size(); i++)
        {
            auto& normal = normals[i];
            S2Point s2_point(normal[0], normal[1], normal[2]);
            S2CellId s2_id(s2_point); 
            auto id = s2_id.id();
            benchmark::DoNotOptimize(id);
            benchmark::ClobberMemory();
        }
    }
}

static void BM_S2NanoCellID (benchmark::State& st)
{
    auto normals = initialize_normals(100000);
    // std::cout<< "Bucket Size: " << GA.buckets.size() << std::endl;
    for (auto _ : st)
    {
        for (size_t i = 0; i < normals.size(); i++)
        {
            auto& normal = normals[i];
            auto id = NanoS2ID::S2CellId(normal);
            benchmark::DoNotOptimize(id);
            benchmark::ClobberMemory();
        }
    }
}

// static void BM_S2NanoCellID_UINT32 (benchmark::State& st)
// {
//     auto normals = initialize_normals(100000);
//     // std::cout<< "Bucket Size: " << GA.buckets.size() << std::endl;
//     for (auto _ : st)
//     {
//         for (size_t i = 0; i < normals.size(); i++)
//         {
//             auto& normal = normals[i];
//             auto id = NanoS2ID::S2CellId_UINT32(normal);
//             benchmark::DoNotOptimize(id);
//             benchmark::ClobberMemory();
//         }
//     }
// }

BENCHMARK(BM_S2PointQuery_1)->UseRealTime()->Unit(benchmark::kMillisecond);
BENCHMARK(BM_S2CellID)->UseRealTime()->Unit(benchmark::kMillisecond);
BENCHMARK(BM_S2NanoCellID)->UseRealTime()->Unit(benchmark::kMillisecond);
// BENCHMARK(BM_S2NanoCellID_UINT32)->UseRealTime()->Unit(benchmark::kMillisecond);
