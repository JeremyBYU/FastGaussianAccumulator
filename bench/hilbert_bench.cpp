#include <random>
#include <cmath>
#include <chrono>
#include <vector>
#include <iostream>
#include "FastGA/Helper.hpp"
#include "Hilbert/Hilbert.hpp"
#include <benchmark/benchmark.h>

class Normals : public benchmark::Fixture
{
  public:
    unsigned seed = 1;
    int N = 100000;
    std::mt19937 generator = std::mt19937(seed);
    std::uniform_real_distribution<double> uniform01 = std::uniform_real_distribution<double>(0.0, 1.0);
    FastGA::MatX3d normals = FastGA::MatX3d(N);
    FastGA::MatX2d projection = FastGA::MatX2d(N);
    FastGA::MatX2ui projection_uint32 = FastGA::MatX2ui(N);
    FastGA::Helper::BBOX projected_bounds;
    void SetUp(const ::benchmark::State& state)
    {
        // generate N random numbers
        initialize_normals();
        projected_bounds = FastGA::Helper::InitializeProjection(normals, projection);
    }
    void initialize_normals(double max_phi_degrees = 100)
    {
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
    }
};

BENCHMARK_DEFINE_F(Normals, BM_ProjectXYZ_TO_XY)
(benchmark::State& st)
{
    std::vector<uint32_t> hilbert_values(N);
    // std::cout << "BBOX" << projected_bounds.min_x << ", " << projected_bounds.min_y << std::endl;
    for (auto _ : st)
    {
        projected_bounds = FastGA::Helper::InitializeProjection(normals, projection);
    }
}

BENCHMARK_DEFINE_F(Normals, BM_MeanTwoPointsBaseline)
(benchmark::State& st)
{
    FastGA::MatX3d assign(normals);
    for (auto _ : st)
    {
        for(size_t i = 0; i < normals.size() -1; i++)
        {
            assign[i] = FastGA::Helper::Mean<double, 3>(normals[i], normals[i+1]);
        }
        benchmark::DoNotOptimize(assign.data());
        benchmark::ClobberMemory();
    }
}

BENCHMARK_DEFINE_F(Normals, BM_ScaleXYToUInt32)
(benchmark::State& st)
{
    // std::cout << "BBOX" << projected_bounds.min_x << ", " << projected_bounds.min_y << std::endl;
    for (auto _ : st)
    {
        double range_x = projected_bounds.max_x - projected_bounds.min_x;
        double range_y = projected_bounds.max_y - projected_bounds.min_y;
        for (int i = 0; i < N; i++)
        {
            FastGA::Helper::ScaleXYToUInt32(&projection[i][0], &projection_uint32[i][0], projected_bounds.min_x, projected_bounds.min_y, range_x, range_y);
        }
    }
}

BENCHMARK_DEFINE_F(Normals, BM_HilbertXY32)
(benchmark::State& st)
{
    std::vector<uint32_t> hilbert_values(N);
    // std::cout << "BBOX" << projected_bounds.min_x << ", " << projected_bounds.min_y << std::endl;
    for (auto _ : st)
    {
        for (int i = 0; i < N; i++)
        {
            hilbert_values[i] = Hilbert::hilbertXYToIndex(16, projection_uint32[i][0], projection_uint32[i][1]);
        }
    }
}

BENCHMARK_DEFINE_F(Normals, BM_NormalsToHilbert)
(benchmark::State& st)
{
    // std::cout << "BBOX" << projected_bounds.min_x << ", " << projected_bounds.min_y << std::endl;
    for (auto _ : st)
    {
        FastGA::MatX2d projection;
        std::vector<uint32_t> hilbert_values;
        std::tie(projection, hilbert_values) = FastGA::Helper::ConvertNormalsToHilbert(normals, projected_bounds);
    }
}


BENCHMARK_REGISTER_F(Normals, BM_MeanTwoPointsBaseline)->UseRealTime()->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(Normals, BM_ProjectXYZ_TO_XY)->UseRealTime()->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(Normals, BM_ScaleXYToUInt32)->UseRealTime()->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(Normals, BM_HilbertXY32)->UseRealTime()->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(Normals, BM_NormalsToHilbert)->UseRealTime()->Unit(benchmark::kMicrosecond);