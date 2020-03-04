#ifndef NANOS2ID_HPP
#define NANOS2ID_HPP

#include <algorithm>
#include <type_traits>
#include <cmath>
#include <mutex>

namespace NanoS2ID {

///////////////////////////////////////////////
//           Start S2 Constants             ///
///////////////////////////////////////////////

using int8 = signed char;
using int16 = short;
using int32 = int;
using int64 = long long;

using uint8 = unsigned char;
using uint16 = unsigned short;
using uint32 = unsigned int;
using uint64 = unsigned long long;

const int kMaxCellLevel = 30;
const int kLimitIJ = 1 << kMaxCellLevel; // == S2CellId::kMaxSize
unsigned const int kMaxSiTi = 1U << (kMaxCellLevel + 1);
static const int kLookupBits = 4;
static uint16 lookup_pos[1 << (2 * kLookupBits + 2)];
static uint16 lookup_ij[1 << (2 * kLookupBits + 2)];

static const int kFaceBits = 3;
static const int kNumFaces = 6;
static const int kMaxLevel = kMaxCellLevel;
static const int kPosBits = 2 * kMaxLevel + 1;
static const int kMaxSize = 1 << kMaxLevel;

int constexpr kSwapMask = 0x01;
int constexpr kInvertMask = 0x02;
// kIJtoPos[orientation][ij] -> pos
const int kIJtoPos[4][4] = {
    // (0,0) (0,1) (1,0) (1,1)
    {0, 1, 3, 2}, // canonical order
    {0, 3, 1, 2}, // axes swapped
    {2, 3, 1, 0}, // bits inverted
    {2, 1, 3, 0}, // swapped & inverted
};

// kPosToIJ[orientation][pos] -> ij
const int kPosToIJ[4][4] = {
    // 0  1  2  3
    {0, 1, 3, 2}, // canonical order:    (0,0), (0,1), (1,1), (1,0)
    {0, 2, 3, 1}, // axes swapped:       (0,0), (1,0), (1,1), (0,1)
    {3, 2, 0, 1}, // bits inverted:      (1,1), (1,0), (0,0), (0,1)
    {3, 1, 0, 2}, // swapped & inverted: (1,1), (0,1), (0,0), (1,0)
};
// kPosToOrientation[pos] -> orientation_modifier
const int kPosToOrientation[4] = {
    kSwapMask,
    0,
    0,
    kInvertMask + kSwapMask,
};

///////////////////////////////////////////////
//            End S2 Constants             ///
///////////////////////////////////////////////

///////////////////////////////////////////////
//      Start Math Utility Functions         //
///////////////////////////////////////////////
static int32 FastIntRound(double x)
{

#if defined __GNUC__ && (defined __i386__ || defined __SSE2__)
#if defined __SSE2__
    // SSE2.
    int32 result;
    __asm__ __volatile__("cvtsd2si %1, %0"
                         : "=r"(result) // Output operand is a register
                         : "x"(x));     // Input operand is an xmm register
    return result;
#elif defined __i386__
    // FPU stack.  Adapted from /usr/include/bits/mathinline.h.
    int32 result;
    __asm__ __volatile__("fistpl %0"
                         : "=m"(result) // Output operand is a memory location
                         : "t"(x)       // Input operand is top of FP stack
                         : "st");       // Clobbers (pops) top of FP stack
    return result;
#endif // if defined __x86_64__ || ...
#else
    return Round<int32, double>(x);
#endif // if defined __GNUC__ && ...
}

static int64 FastInt64Round(double x)
{
#if defined __GNUC__ && (defined __i386__ || defined __x86_64__)
#if defined __x86_64__
    // SSE2.
    int64 result;
    __asm__ __volatile__("cvtsd2si %1, %0"
                         : "=r"(result) // Output operand is a register
                         : "x"(x));     // Input operand is an xmm register
    return result;
#elif defined __i386__
    // There is no CVTSD2SI in i386 to produce a 64 bit int, even with SSE2.
    // FPU stack.  Adapted from /usr/include/bits/mathinline.h.
    int64 result;
    __asm__ __volatile__("fistpll %0"
                         : "=m"(result) // Output operand is a memory location
                         : "t"(x)       // Input operand is top of FP stack
                         : "st");       // Clobbers (pops) top of FP stack
    return result;
#endif // if defined __i386__
#else
    return Round<int64, double>(x);
#endif // if defined __GNUC__ && ...
}

inline std::array<double, 3> Abs(const std::array<double, 3>& p)
{
    using std::abs;
    std::array<double, 3> a1{{abs(p[0]), abs(p[1]), abs(p[2])}};
    return a1;
}

// return the index of the largest component (fabs)
inline int LargestAbsComponent(const std::array<double, 3>& p)
{
    auto temp = Abs(p);
    return temp[0] > temp[1] ? temp[0] > temp[2] ? 0 : 2 : temp[1] > temp[2] ? 1 : 2;
}

///////////////////////////////////////////////
//       End Math Utility Functions         //
///////////////////////////////////////////////

///////////////////////////////////////////////
//      Start Hilbert Curve Functions       ///
///////////////////////////////////////////////

static void InitLookupCell(int level, int i, int j, int orig_orientation,
                           int pos, int orientation)
{
    if (level == kLookupBits)
    {
        int ij = (i << kLookupBits) + j;
        lookup_pos[(ij << 2) + orig_orientation] = (pos << 2) + orientation;
        lookup_ij[(pos << 2) + orig_orientation] = (ij << 2) + orientation;
    }
    else
    {
        level++;
        i <<= 1;
        j <<= 1;
        pos <<= 2;
        const int* r = kPosToIJ[orientation];
        InitLookupCell(level, i + (r[0] >> 1), j + (r[0] & 1), orig_orientation,
                       pos, orientation ^ kPosToOrientation[0]);
        InitLookupCell(level, i + (r[1] >> 1), j + (r[1] & 1), orig_orientation,
                       pos + 1, orientation ^ kPosToOrientation[1]);
        InitLookupCell(level, i + (r[2] >> 1), j + (r[2] & 1), orig_orientation,
                       pos + 2, orientation ^ kPosToOrientation[2]);
        InitLookupCell(level, i + (r[3] >> 1), j + (r[3] & 1), orig_orientation,
                       pos + 3, orientation ^ kPosToOrientation[3]);
    }
}

static std::once_flag flag;
inline static void MaybeInit()
{
    std::call_once(flag, [] {
        InitLookupCell(0, 0, 0, 0, 0, 0);
        InitLookupCell(0, 0, 0, kSwapMask, 0, kSwapMask);
        InitLookupCell(0, 0, 0, kInvertMask, 0, kInvertMask);
        InitLookupCell(0, 0, 0, kSwapMask | kInvertMask, 0, kSwapMask | kInvertMask);
    });
}

inline void GET_BITS_FUNCTION(int k, uint64& n, uint64& bits, int& i, int& j)
{
    const int mask = (1 << kLookupBits) - 1;
    bits += ((i >> (k * kLookupBits)) & mask) << (kLookupBits + 2);
    bits += ((j >> (k * kLookupBits)) & mask) << 2;
    bits = lookup_pos[bits];
    n |= (bits >> 2) << (k * 2 * kLookupBits);
    bits &= (kSwapMask | kInvertMask);
}

inline uint64 FromFaceIJ(int face, int i, int j)
{
    // Initialization if not done yet
    MaybeInit();

    uint64 n = static_cast<uint64>(face) << (kPosBits - 1);
    uint64 bits = (face & kSwapMask);
    GET_BITS_FUNCTION(7, n, bits, i, j);
    GET_BITS_FUNCTION(6, n, bits, i, j);
    GET_BITS_FUNCTION(5, n, bits, i, j);
    GET_BITS_FUNCTION(4, n, bits, i, j);
    GET_BITS_FUNCTION(3, n, bits, i, j);
    GET_BITS_FUNCTION(2, n, bits, i, j);
    GET_BITS_FUNCTION(1, n, bits, i, j);
    GET_BITS_FUNCTION(0, n, bits, i, j);

    auto final_value = n * 2 + 1;
    return final_value;
}
///////////////////////////////////////////////
//       End Hilbert Curve Functions       ///
///////////////////////////////////////////////

///////////////////////////////////////////////
//     Start Cubic Projections Fuctions     ///
///////////////////////////////////////////////

#define S2_LINEAR_PROJECTION 0
#define S2_TAN_PROJECTION 1
#define S2_QUADRATIC_PROJECTION 2

#define S2_PROJECTION S2_QUADRATIC_PROJECTION

#if S2_PROJECTION == S2_LINEAR_PROJECTION

inline double STtoUV(double s)
{
    return 2 * s - 1;
}

inline double UVtoST(double u)
{
    return 0.5 * (u + 1);
}

#elif S2_PROJECTION == S2_TAN_PROJECTION

inline double STtoUV(double s)
{
    s = std::tan(M_PI_2 * s - M_PI_4);
    return s + (1.0 / (int64{1} << 53)) * s;
}

inline double UVtoST(double u)
{
    volatile double a = std::atan(u);
    return (2 * M_1_PI) * (a + M_PI_4);
}

#elif S2_PROJECTION == S2_QUADRATIC_PROJECTION

inline double STtoUV(double s)
{
    if (s >= 0.5)
        return (1 / 3.) * (4 * s * s - 1);
    else
        return (1 / 3.) * (1 - 4 * (1 - s) * (1 - s));
}

inline double UVtoST(double u)
{
    if (u >= 0)
        return 0.5 * std::sqrt(1 + 3 * u);
    else
        return 1 - 0.5 * std::sqrt(1 - 3 * u);
}

#else
#error Unknown value for S2_PROJECTION
#endif
///////////////////////////////////////////////
//       End Cubic Projections Fuctions     ///
///////////////////////////////////////////////

///////////////////////////////////////////////
//       Start Public API Fuctions          ///
///////////////////////////////////////////////

inline int STtoIJ(double s)
{
    return std::max(0, std::min(kLimitIJ - 1,
                                FastIntRound(kLimitIJ * s - 0.5)));
}

inline int GetFace(const std::array<double, 3>& p)
{
    int face = LargestAbsComponent(p);
    if (p[face] < 0) face += 3;
    return face;
}

inline void ValidFaceXYZtoUV(int face, const std::array<double, 3>& p,
                             double* pu, double* pv)
{
    //   S2_DCHECK_GT(p.DotProd(GetNorm(face)), 0);
    switch (face)
    {
    case 0:
        *pu = p[1] / p[0];
        *pv = p[2] / p[0];
        break;
    case 1:
        *pu = -p[0] / p[1];
        *pv = p[2] / p[1];
        break;
    case 2:
        *pu = -p[0] / p[2];
        *pv = -p[1] / p[2];
        break;
    case 3:
        *pu = p[2] / p[0];
        *pv = p[1] / p[0];
        break;
    case 4:
        *pu = p[2] / p[1];
        *pv = -p[0] / p[1];
        break;
    default:
        *pu = -p[1] / p[2];
        *pv = -p[0] / p[2];
        break;
    }
}

inline int XYZtoFaceUV(const std::array<double, 3>& p, double* pu, double* pv)
{
    int face = GetFace(p);
    ValidFaceXYZtoUV(face, p, pu, pv);
    return face;
}

inline uint64 S2CellId(const std::array<double, 3>& p)
{
    double u, v;
    int face = XYZtoFaceUV(p, &u, &v);
    int i = STtoIJ(UVtoST(u));
    int j = STtoIJ(UVtoST(v));
    uint64 id = FromFaceIJ(face, i, j);
    return id;
}
///////////////////////////////////////////////
//       End Public API Fuctions          ///
///////////////////////////////////////////////

} // namespace NanoS2ID

#endif