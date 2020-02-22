#include <benchmark/benchmark.h>
#include "FastGA.hpp"


void BM_Test(benchmark::State& state)
{
    
    for (auto _ : state) {
        auto val = FastGA::test();
    }
}

BENCHMARK(BM_Test)->UseRealTime();

// Run the benchmark
BENCHMARK_MAIN();