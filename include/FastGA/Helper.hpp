#ifndef FASTGA_HELPER_HPP
#define FASTGA_HELPER_HPP
#define _USE_MATH_DEFINES
#include <cmath>

#include <cmath>
#include <vector>
#include <cstdint>
#include <limits>
#include <climits>
#include <tuple>
#include <iostream>
#include <type_traits>
#include <ostream>
#include <iterator>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <string>
#include "Hilbert/Hilbert.hpp"
#include "NanoS2ID/NanoS2ID.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define degreesToRadians(angleDegrees) ((angleDegrees)*M_PI / 180.0)
#define radiansToDegrees(angleRadians) ((angleRadians)*180.0 / M_PI)

// template <typename T>
// std::ostream& operator<<(std::ostream& out, const std::vector<T>& v)
// {
//     if (!v.empty())
//     {
//         out << '[';
//         std::copy(v.begin(), v.end(), std::ostream_iterator<T>(out, ", "));
//         out << "\b\b]";
//     }
//     return out;
// }

template <typename T>
std::ostream& operator<<(std::ostream& out, const std::array<T, 3>& v)
{
    if (!v.empty())
    {
        out << '[';
        std::copy(v.begin(), v.end(), std::ostream_iterator<T>(out, ", "));
        out << "\b\b]";
    }
    return out;
}

// template <typename T>
// std::ostream& operator<<(std::ostream& out, const std::vector<std::array<T, 2>>& v)
// {
//     if (!v.empty())
//     {
//         for(auto &ar: v)
//         {
//             out << ar << ", ";
//         }
//     }
//     out << std::endl;
//     return out;
// }

// template <typename T>
// std::ostream& operator<<(std::ostream& out, const std::array<T, 2>& v)
// {
//     if (!v.empty())
//     {
//         out << '[';
//         std::copy(v.begin(), v.end(), std::ostream_iterator<T>(out, ", "));
//         out << "\b\b]";
//     }
//     return out;
// }

namespace FastGA {

std::string GetFastGAVersion();

using MatX3d = std::vector<std::array<double, 3>>;
using MatX3I = std::vector<std::array<size_t, 3>>;
using MatX6I = std::vector<std::array<size_t, 6>>;
using MatX12I = std::vector<std::array<size_t, 12>>;
using MatX2d = std::vector<std::array<double, 2>>;
using MatX2I = std::vector<std::array<size_t, 2>>;
using MatX2ui = std::vector<std::array<uint32_t, 2>>;
using MatX2i = std::vector<std::array<int, 2>>;
using MatX4i = std::vector<std::array<int, 4>>;
struct Regression
{
    double slope;
    double intercept;
    int pos_error;
    int neg_error;
    int subtract_index;
    int power_of_2;
    int Predict(uint32_t value)
    {
        return static_cast<int>(intercept + slope * static_cast<double>(value));
    }
    int Predict(uint64_t value)
    {
        return static_cast<int>(intercept + slope * static_cast<double>(value));
    }
};
constexpr size_t max_limit = std::numeric_limits<size_t>::max();

/**
 * @brief The bucket class describes a cell on the S2 Histogram.
 *        It unfortunately has one member that should not really exist (projection).
 *       
 * @tparam T 
 */
template <class T>
struct Bucket
{
    /** @brief The surface **unit** normal of the triangle cell   */
    std::array<double, 3> normal;
    /** @brief Count variable for histogram   */
    uint32_t count;
    /** @brief Space Filling Curve value, may be Int32 or Uint64  */
    T hilbert_value;
    /** @brief Ignore this  */
    std::array<double, 2> projection;
    // Bucket(const std::array<double, 3> normal_, uint32_t count_, T hilbert_value_, const std::array<double, 2> projection_): normal(normal_), count(count_), hilbert_value(hilbert_value_), projection(projection_) {}
    bool operator<(const Bucket& other) const
    {
        return hilbert_value < other.hilbert_value;
    }
};

struct BucketS2
{
    /** @brief The surface **unit** normal of the triangle cell   */
    std::array<double, 3> normal;
    /** @brief The average **unit** normal after integration   */
    std::array<double, 3> average_normal;
    /** @brief Count variable for histogram   */
    uint32_t count;
    /** @brief Space Filling Curve value, may be Int32 or Uint64  */
    uint64_t hilbert_value;
    // Bucket(const std::array<double, 3> normal_, uint32_t count_, T hilbert_value_, const std::array<double, 2> projection_): normal(normal_), count(count_), hilbert_value(hilbert_value_), projection(projection_) {}
    bool operator<(const BucketS2& other) const
    {
        return hilbert_value < other.hilbert_value;
    }
};

namespace Helper {

const static uint32_t HILBERT_MAX_32 = std::numeric_limits<uint16_t>::max();
const static double EPSILON = 1e-5;

template <typename UnsignedType>
UnsignedType round_up_to_power_of_2(UnsignedType v) {
  static_assert(std::is_unsigned<UnsignedType>::value, "Only works for unsigned types");
  v--;
  for (size_t i = 1; i < sizeof(v) * CHAR_BIT; i *= 2) //Prefer size_t "Warning comparison between signed and unsigned integer"
  {
    v |= v >> i;
  }
  return ++v;
}

template <typename T, typename Compare>
inline std::vector<std::size_t> sort_permutation(
    const std::vector<T>& vec,
    Compare compare)
{
    std::vector<std::size_t> p(vec.size());
    std::iota(p.begin(), p.end(), 0);
    std::sort(p.begin(), p.end(),
              [&](std::size_t i, std::size_t j) { return compare(vec[i], vec[j]); });
    return p;
}

template <typename T>
inline T SquaredDistance(const std::array<T, 3>& a, const std::array<T, 3>& b)
{
    auto x = (a[0] - b[0]);
    auto y = (a[1] - b[1]);
    auto z = (a[2] - b[2]);
    return x * x + y * y + z * z;
}

template <typename T>
inline std::vector<T> BubbleDownMask(const std::vector<T>& vec, std::vector<uint8_t> mask)
{
    std::vector<T> start;
    std::vector<T> end;
    for (size_t i = 0; i < mask.size(); i++)
    {
        if (mask[i] > 0)
        {
            start.push_back(vec[i]);
        }
        else
        {
            end.push_back(vec[i]);
        }
    }
    start.insert(start.end(), end.begin(), end.end());
    return start;
}
template <class T>
double MovingAverage(const std::vector<T>& x)
{
    double moving_average = 0.0;
    for (size_t i = 0; i < x.size(); i++)
    {
        moving_average += (static_cast<double>(x[i]) - moving_average) / static_cast<double>(i + 1);
    }
    return moving_average;
}

template <class T, class G>
void LinearRegression(const std::vector<T>& x, const std::vector<G>& y, Regression& r)
{
    if (x.size() != y.size())
    {
        throw std::runtime_error("Arguments must be the same size");
    }
    size_t n = x.size();

    double x_bar = MovingAverage(x);
    double y_bar = MovingAverage(y);

    double numerator = 0.0;
    double denominator = 0.0;

    for (size_t i = 0; i < n; ++i)
    {
        numerator += (static_cast<double>(x[i]) - x_bar) * (static_cast<double>(y[i]) - y_bar);
        denominator += (static_cast<double>(x[i]) - x_bar) * (static_cast<double>(x[i]) - x_bar);
    }

    double slope = numerator / denominator;
    double intercept = y_bar - slope * x_bar;

    // Set slope and intercept
    r.intercept = intercept;
    r.slope = slope;
    // Determine worst bounds of predictions
    int neg_error = 10000000;
    int pos_error = -10000000;
    for (size_t i = 0; i < n; ++i)
    {
        auto predicted = r.Predict(x[i]);
        auto error = predicted - static_cast<int>(y[i]);
        if (error < neg_error)
            neg_error = error;
        if (error > pos_error)
            pos_error = error;
    }
    r.pos_error = pos_error;
    r.neg_error = neg_error;
    // Get power of 2 bounds
    auto max_val = static_cast<uint32_t>(std::abs(r.pos_error) + std::abs(r.neg_error));
    r.power_of_2 = round_up_to_power_of_2(max_val) - 1;
    r.subtract_index = r.power_of_2 / 2 + 1;
}

template <typename T>
inline void ApplyPermutationInPlace(
    std::vector<T>& vec,
    const std::vector<std::size_t>& p)
{
    std::vector<bool> done(p.size());
    for (std::size_t i = 0; i < p.size(); ++i)
    {
        if (done[i])
        {
            continue;
        }
        done[i] = true;
        std::size_t prev_j = i;
        std::size_t j = p[i];
        while (i != j)
        {
            std::swap(vec[prev_j], vec[j]);
            done[j] = true;
            prev_j = j;
            j = p[j];
        }
    }
}

template <typename T>
inline std::vector<T> ApplyPermutation(
    const std::vector<T>& vec,
    const std::vector<std::size_t>& p)
{
    std::vector<T> sorted_vec(p.size());
    std::transform(p.begin(), p.end(), sorted_vec.begin(),
                   [&](std::size_t i) { return vec[i]; });
    sorted_vec.insert(sorted_vec.end(), vec.begin() + p.size(), vec.end());
    return sorted_vec;
}

struct BBOX
{
    double min_x;
    double min_y;
    double max_x;
    double max_y;
};

template <class T>
void ScaleItemInPlace(std::array<T, 3>& item, T scalar)
{
    item[0] *= scalar;
    item[1] *= scalar;
    item[2] *= scalar;
}

template <class T, int dim>
T L2Norm(std::array<T, dim>& a)
{
    T norm = 0;
    for (size_t i = 0; i < a.size(); i++)
    {
        norm += a[i] * a[i];
    }
    norm = sqrt(norm);
    return norm;
}

template <class T, int dim>
std::string ArrayToString(const std::array<T, dim>& a)
{
    std::string a_s = "(";
    for (size_t i = 0; i < a.size(); i++)
    {
        a_s += std::to_string(a[i]) + ",";
    }
    return a_s + ")";
}

inline void crossProduct3(const std::array<double, 3>& u, const std::array<double, 3>& v, double* normal)
{
    // cross product
    normal[0] = u[1] * v[2] - u[2] * v[1];
    normal[1] = u[2] * v[0] - u[0] * v[2];
    normal[2] = u[0] * v[1] - u[1] * v[0];
}

inline void normalize3(double* normal)
{
    auto norm = std::sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
    normal[0] /= norm;
    normal[1] /= norm;
    normal[2] /= norm;
}

template <class T, int dim>
std::array<T, dim> Mean(std::array<T, dim>& a, std::array<T, dim>& b)
{
    std::array<T, dim> mean = {{0, 0, 0}};
    for (size_t i = 0; i < a.size(); i++)
    {
        mean[i] = (a[i] + b[i]) / 2.0;
    }
    return mean;
}

template <class T, int dim>
void InPlaceAdd(const std::array<T, dim>& a, std::array<T, dim>& b)
{
    for (size_t i = 0; i < dim; i++)
    {
        b[i] = (a[i] + b[i]);
    }
}

template <class T, int dim>
void InPlaceAddScale(const std::array<T, dim>& a, std::array<T, dim>& b, T scale)
{
    for (size_t i = 0; i < dim; i++)
    {
        b[i] = (a[i] * scale + b[i]);
    }
}

template <class T, int dim>
void InPlaceDivide(const std::array<T, dim>& a, double value)
{
    for (size_t i = 0; i < dim; i++)
    {
        a[i] = a[i] / value;
    }
}

template <size_t dim_indexes>
std::vector<double> Mean(std::vector<std::array<size_t, dim_indexes>>& indexes, std::vector<double> values)
{
    size_t num_values = indexes.size();
    std::vector<double> mean_values(num_values);
    for (size_t i = 0; i < indexes.size(); i++)
    {
        double total_mean = 0.0;
        size_t idx_i = 0;
        for (idx_i = 0; idx_i < dim_indexes; ++idx_i)
        {
            auto &value_idx = indexes[i][idx_i]; 
            if (value_idx == max_limit)
                break;
            total_mean += values[value_idx];
        }
        idx_i++;
        total_mean /= static_cast<double>(idx_i);
        mean_values[i] = total_mean;
    }
    
    return mean_values;
}

template <size_t dim_indexes>
FastGA::MatX3d MeanAverageNormals(std::vector<std::array<size_t, dim_indexes>>& indexes, FastGA::MatX3d normals, std::vector<double> &scales)
{
    size_t num_values = indexes.size();
    FastGA::MatX3d mean_vertices(num_values);
    for (size_t i = 0; i < indexes.size(); i++)
    {
        std::array<double, 3> total_mean{{0.0,0.0,0.0}};
        double total_scale = 0.0;
        size_t idx_i = 0;
        for (idx_i = 0; idx_i < dim_indexes; ++idx_i)
        {
            auto &value_idx = indexes[i][idx_i]; 
            if (value_idx == max_limit)
                break;
            auto &triangle_normal = normals[value_idx];
            auto scale = scales[value_idx];

            InPlaceAddScale<double, 3>(triangle_normal, total_mean, scale);
            total_scale += scale;
        }
        double invert_total_scale = 1.0 / total_scale;
        ScaleItemInPlace(total_mean, invert_total_scale);
        normalize3(&total_mean[0]);
        mean_vertices[i] = total_mean;
        
    }
    
    return mean_vertices;
}

template <class T, size_t dim_indexes, size_t dim_values>
std::vector<double> Mean(std::vector<std::array<size_t, dim_indexes>>& indexes, std::vector<std::array<T, dim_values>> values)
{
    size_t num_values = indexes.size();
    std::vector<std::array<T, dim_values>> mean_values(num_values);
    for (size_t i = 0; i < indexes.size(); i++)
    {
        std::array<T, dim_values> total_mean;
        size_t idx_i = 0;
        for (idx_i = 0; idx_i < dim_indexes; ++idx_i)
        {
            auto &value_idx = indexes[i][idx_i]; 
            if (value_idx == max_limit)
                break;
            InPlaceAdd(values[value_idx], total_mean);
        }
        idx_i++;
        InPlaceDivide(total_mean, static_cast<double>(idx_i));
        mean_values[i] = total_mean;
    }
    
    return mean_values;
}

template <class T, int dim>
void ScaleArrayInPlace(std::vector<std::array<T, dim>>& array, T scalar)
{
    for (auto&& item : array)
    {
        ScaleItemInPlace(item, scalar);
    }
}

template <class T>
inline void AzimuthEqualAreaProjectionXYZ(const double* xyz, T* xy)
{
    T top = std::sqrt(2.0 / (1 + xyz[2]));
    xy[0] = top * xyz[0];
    xy[1] = top * xyz[1];
}

template <class T>
inline void ScaleXYToUInt32(const T* xy, uint32_t* scale, T min_x, T min_y, T x_range, T y_range)
{
    // TODO UINT Overflow, need min and max
    // Technically dont do this if you make sure that xy is already min and maxed.
    // T scale_x = std::max(std::min((xy[0] - min_x) / x_range, 1.0), 0.0);
    // T scale_y = std::max(std::min((xy[1] - min_y) / y_range, 1.0), 0.0);

    T scale_x = ((xy[0] - min_x) / x_range);
    T scale_y = ((xy[1] - min_y) / y_range);

    scale[0] = static_cast<uint32_t>(scale_x * HILBERT_MAX_32);
    scale[1] = static_cast<uint32_t>(scale_y * HILBERT_MAX_32);
}

template <class T>
inline void AzimuthProjectionPhiTheta(T* pt, T* xy)
{
    xy[0] = pt[0] * sin(pt[1]);
    xy[1] = -pt[0] * cos(pt[1]);
}

inline BBOX InitializeProjection(const MatX3d& normals, MatX2d& projection)
{
    size_t N = normals.size();
    double min_x = std::numeric_limits<double>::max();
    double min_y = std::numeric_limits<double>::max();
    double max_x = std::numeric_limits<double>::lowest();
    double max_y = std::numeric_limits<double>::lowest();
    for (size_t i = 0; i < N; i++)
    {
        AzimuthEqualAreaProjectionXYZ(&(normals[i][0]), &(projection[i][0]));
        if (projection[i][0] < min_x)
        {
            min_x = projection[i][0];
        }
        if (projection[i][0] > max_x)
        {
            max_x = projection[i][0];
        }

        if (projection[i][1] < min_y)
        {
            min_y = projection[i][1];
        }
        if (projection[i][1] > max_y)
        {
            max_y = projection[i][1];
        }
    }
    return {min_x, min_y, max_x, max_y};
}

template <class T>
inline BBOX InitializeProjection(std::vector<Bucket<T>>& buckets)
{
    size_t N = buckets.size();
    double min_x = std::numeric_limits<double>::max();
    double min_y = std::numeric_limits<double>::max();
    double max_x = std::numeric_limits<double>::lowest();
    double max_y = std::numeric_limits<double>::lowest();
    for (size_t i = 0; i < N; i++)
    {
        auto& projection = buckets[i].projection;
        AzimuthEqualAreaProjectionXYZ(&(buckets[i].normal[0]), &projection[0]);
        if (projection[0] < min_x)
        {
            min_x = projection[0];
        }
        if (projection[0] > max_x)
        {
            max_x = projection[0];
        }

        if (projection[1] < min_y)
        {
            min_y = projection[1];
        }
        if (projection[1] > max_y)
        {
            max_y = projection[1];
        }
    }
    return {min_x, min_y, max_x, max_y};
}

inline std::tuple<MatX2d, std::vector<uint32_t>> ConvertNormalsToHilbert(const MatX3d& normals, BBOX& bbox)
{
    size_t N = normals.size();
    MatX2d projection(N);
    std::vector<uint32_t> hilbert_values(N);

    InitializeProjection(normals, projection);
    std::array<uint32_t, 2> xy_int;
    double x_range = bbox.max_x - bbox.min_x;
    double y_range = bbox.max_y - bbox.min_y;
    // std::cout << "Range: " << x_range << ", " << y_range << std::endl;
    for (size_t i = 0; i < N; i++)
    {
        ScaleXYToUInt32(&projection[i][0], xy_int.data(), bbox.min_x, bbox.min_y, x_range, y_range);
        hilbert_values[i] = Hilbert::hilbertXYToIndex(16u, xy_int[0], xy_int[1]);
    }
    return std::make_tuple(std::move(projection), std::move(hilbert_values));
}

inline std::vector<uint64_t> ConvertNormalsToS2ID(const MatX3d& normals)
{
    size_t N = normals.size();
    MatX2d projection(N);
    std::vector<uint64_t> s2_ids(N);
    for (size_t i = 0; i < N; i++)
    {
        auto& normal = normals[i];
        s2_ids[i] = NanoS2ID::S2CellId(normal);
    }
    return s2_ids;
}

void inline ComputeTriangleNormals(const MatX3d& vertices, const MatX3I& triangles, MatX3d& triangle_normals)
{
    size_t num_triangles = triangles.size();
    triangle_normals.resize(num_triangles);

    for (size_t i = 0; i < triangles.size(); i++)
    {
        auto& pi0 = triangles[i][0];
        auto& pi1 = triangles[i][1];
        auto& pi2 = triangles[i][2];

        std::array<double, 3> vv1 = {vertices[pi0][0], vertices[pi0][1], vertices[pi0][2]};
        std::array<double, 3> vv2 = {vertices[pi1][0], vertices[pi1][1], vertices[pi1][2]};
        std::array<double, 3> vv3 = {vertices[pi2][0], vertices[pi2][1], vertices[pi2][2]};

        // two lines of triangle
        // V1 is starting index
        std::array<double, 3> u{{vv2[0] - vv1[0], vv2[1] - vv1[1], vv2[2] - vv1[2]}};
        std::array<double, 3> v{{vv3[0] - vv1[0], vv3[1] - vv1[1], vv3[2] - vv1[2]}};

        // cross product
        crossProduct3(u, v, &triangle_normals[i][0]);
        // normalize
        normalize3(&triangle_normals[i][0]);
    }
}

} // namespace Helper
} // namespace FastGA
#endif