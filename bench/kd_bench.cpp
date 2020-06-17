#include <random>
#include <cmath>
#include <chrono>
#include <vector>
#include <iostream>
#include "FastGA/FastGA.hpp"
#include "Hilbert/Hilbert.hpp"
#include <benchmark/benchmark.h>
#include "npy/npy.h"

#define max_phi 180.0
class RandomSample : public benchmark::Fixture
{
  public:
    unsigned seed = 1;
    int N = 100000;
    std::mt19937 generator = std::mt19937(seed);
    std::uniform_real_distribution<double> uniform01 = std::uniform_real_distribution<double>(0.0, 1.0);
    FastGA::MatX3d normals = FastGA::MatX3d(N);

    void SetUp(const ::benchmark::State& state)
    {
        // generate N random numbers
        initialize_normals(max_phi - 5.0);
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

class Mesh : public benchmark::Fixture
{
  public:
    FastGA::MatX3d normals = FastGA::MatX3d();

    void SetUp(const ::benchmark::State& state)
    {
        std::vector<unsigned long> shape;
        bool fortran_order;
        std::vector<double> data;
        // std::cout << "help me" << std::endl;
        npy::LoadArrayFromNumpy("fixtures/normals/basement.npy", shape, fortran_order, data);
        // std::cout << shape[0] << " " << shape[1] << std::endl;
        // generate N random numbers
        normals = FastGA::MatX3d(shape[0]);
        // std::cout << "reallocated" << std::endl;
        for (int i = 0; i < shape[0]; ++i)
        {
            normals[i] = {data[i * 3], data[i * 3 + 1], data[i * 3 + 2]};
        }
    }
};

BENCHMARK_DEFINE_F(RandomSample, BM_FastGAKD)
(benchmark::State& st)
{
    FastGA::GaussianAccumulatorKD GA = FastGA::GaussianAccumulatorKD(4, max_phi, st.range(0));
    // float eps = 0.1 * st.range(1);
    for (auto _ : st)
    {
        auto test = GA.Integrate(normals);
    }
}

BENCHMARK_DEFINE_F(Mesh, BM_FastGAKD)
(benchmark::State& st)
{
    FastGA::GaussianAccumulatorKD GA = FastGA::GaussianAccumulatorKD(4, max_phi, st.range(0));
    // float eps = 0.1 * st.range(1);
    for (auto _ : st)
    {
        auto test = GA.Integrate(normals);
    }
}

BENCHMARK_DEFINE_F(RandomSample, BM_FastGAOpt)
(benchmark::State& st)
{
    FastGA::GaussianAccumulatorOpt GA = FastGA::GaussianAccumulatorOpt(4, max_phi);
    for (auto _ : st)
    {
        auto test = GA.Integrate(normals, st.range(0));
    }
}
BENCHMARK_DEFINE_F(Mesh, BM_FastGAOpt)
(benchmark::State& st)
{
    FastGA::GaussianAccumulatorOpt GA = FastGA::GaussianAccumulatorOpt(4, max_phi);
    for (auto _ : st)
    {
        auto test = GA.Integrate(normals, st.range(0));
    }
}

// BENCHMARK_DEFINE_F(Normals, BM_FastGAOpt2)
// (benchmark::State& st)
// {
//     FastGA::GaussianAccumulatorOpt GA = FastGA::GaussianAccumulatorOpt(4, max_phi);
//     for (auto _ : st)
//     {
//         auto test = GA.Integrate2(normals, st.range(0));
//     }
// }

BENCHMARK_DEFINE_F(RandomSample, BM_FastGAS2)
(benchmark::State& st)
{
    FastGA::GaussianAccumulatorS2 GA = FastGA::GaussianAccumulatorS2(4, max_phi);
    for (auto _ : st)
    {
        auto test = GA.Integrate(normals, st.range(0));
    }
}

BENCHMARK_DEFINE_F(Mesh, BM_FastGAS2)
(benchmark::State& st)
{
    FastGA::GaussianAccumulatorS2 GA = FastGA::GaussianAccumulatorS2(4, max_phi);
    for (auto _ : st)
    {
        auto test = GA.Integrate(normals, st.range(0));
    }
}

// BENCHMARK_DEFINE_F(Normals, BM_FastGAS2_Unrolled)
// (benchmark::State& st)
// {
//     FastGA::GaussianAccumulatorS2 GA = FastGA::GaussianAccumulatorS2(4, max_phi);
//     for (auto _ : st)
//     {
//         auto test = GA.IntegrateUnrolled(normals);
//     }
// }

BENCHMARK_REGISTER_F(RandomSample, BM_FastGAKD)->RangeMultiplier(2)->Ranges({{1, 32}})->UseRealTime()->Unit(benchmark::kMillisecond);
BENCHMARK_REGISTER_F(RandomSample, BM_FastGAOpt)->RangeMultiplier(2)->Ranges({{1, 12}})->UseRealTime()->Unit(benchmark::kMillisecond);
BENCHMARK_REGISTER_F(RandomSample, BM_FastGAS2)->RangeMultiplier(2)->Ranges({{1, 12}})->UseRealTime()->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(Mesh, BM_FastGAKD)->RangeMultiplier(2)->Ranges({{1, 32}})->UseRealTime()->Unit(benchmark::kMillisecond);
BENCHMARK_REGISTER_F(Mesh, BM_FastGAOpt)->RangeMultiplier(2)->Ranges({{1, 12}})->UseRealTime()->Unit(benchmark::kMillisecond);
BENCHMARK_REGISTER_F(Mesh, BM_FastGAS2)->RangeMultiplier(2)->Ranges({{1, 12}})->UseRealTime()->Unit(benchmark::kMillisecond);
// BENCHMARK_REGISTER_F(Normals, BM_FastGAKD)->RangeMultiplier(2)->Ranges({{8, 8}, {2, 20}})->UseRealTime()->Unit(benchmark::kMillisecond);
// BENCHMARK_REGISTER_F(Normals, BM_FastGAS2_Unrolled)->UseRealTime()->Unit(benchmark::kMillisecond);
// BENCHMARK_REGISTER_F(Normals, BM_FastGAOpt2)->RangeMultiplier(2)->Ranges({{1, 12}})->UseRealTime()->Unit(benchmark::kMillisecond);