#include <benchmark/benchmark.h>
#include "gaussian_integrator.hpp"


void BM_Test(benchmark::State& state)
{
    
    for (auto _ : state) {
        auto val = gaussian_integrator::test();
    }
}

BENCHMARK(BM_Test)->UseRealTime();

// Run the benchmark
BENCHMARK_MAIN();