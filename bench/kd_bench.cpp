#include <random>
#include <cmath>
#include <chrono>
#include <vector>
#include <iostream>
#include "FastGA/FastGA.hpp"
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
    void initialize_normals(double max_phi_degrees = 95)
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

BENCHMARK_DEFINE_F(Normals, BM_FastGAKD)
(benchmark::State& st)
{
    FastGA::GaussianAccumulatorKD GA = FastGA::GaussianAccumulatorKD(4, 100.0, st.range(0));
    for (auto _ : st)
    {
        auto test = GA.Integrate(normals);
    }
}

BENCHMARK_DEFINE_F(Normals, BM_FastGAOpt)
(benchmark::State& st)
{
    FastGA::GaussianAccumulatorOpt GA = FastGA::GaussianAccumulatorOpt(4, 100.0);
    for (auto _ : st)
    {
        auto test = GA.Integrate(normals);
    }
}


BENCHMARK_REGISTER_F(Normals, BM_FastGAKD)->RangeMultiplier(2)->Ranges({{1, 32}})->UseRealTime()->Unit(benchmark::kMillisecond);
BENCHMARK_REGISTER_F(Normals, BM_FastGAOpt)->UseRealTime()->Unit(benchmark::kMillisecond);