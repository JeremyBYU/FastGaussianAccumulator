
// MIT License

// Copyright (c) 2017 stgatilov, 2019 Jeremy Castagno
// See https://dirtyhandscoding.wordpress.com/2017/08/25/performance-comparison-linear-search-vs-binary-search/
#ifndef FASTGA_FBS_HPP
#define FASTGA_FBS_HPP

#include <climits>
#include <type_traits>
#include <stdint.h>
#include <assert.h>

namespace FBS {

#ifdef _MSC_VER
#define FORCEINLINE __forceinline
#define NOINLINE __declspec(noinline)
#define ALIGN(n) __declspec(align(n))
FORCEINLINE uint32_t bsr(uint32_t x)
{
    unsigned long res;
    _BitScanReverse(&res, x);
    return res;
}
FORCEINLINE uint32_t bsf(uint32_t x)
{
    unsigned long res;
    _BitScanForward(&res, x);
    return res;
}
#else
#define FORCEINLINE __attribute__((always_inline)) inline
#define NOINLINE __attribute__((noinline))
#define ALIGN(n) __attribute__((aligned(n)))
FORCEINLINE uint32_t bsr(uint32_t x)
{
    return 31 - __builtin_clz(x);
}
FORCEINLINE uint32_t bsf(uint32_t x)
{
    return __builtin_ctz(x);
}
#endif

constexpr static intptr_t MINUS_ONE = -1;

template <class T>
inline int binary_search_branchless(const T* arr, int n, T key)
{
    assert((n & (n + 1)) == 0); //n = 2^k - 1
    intptr_t pos = MINUS_ONE;
    intptr_t logstep = bsr(n);
    intptr_t step = intptr_t(1) << logstep;
    while (step > 0)
    {
        pos = (arr[pos + step] < key ? pos + step : pos);
        step >>= 1;
    }
    return static_cast<int>(pos + 1);
}

} // namespace FBS

#endif