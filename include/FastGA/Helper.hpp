#ifndef FASTGA_HELPER_HPP
#define FASTGA_HELPER_HPP

#include <cmath>
#include <vector>
#include <cstdint>
#include <limits>
#include <tuple>
#include <iostream>
#include <map>
#include "Hilbert/Hilbert.hpp"

#define _USE_MATH_DEFINES
#define degreesToRadians(angleDegrees) ((angleDegrees) * M_PI / 180.0)
#define radiansToDegrees(angleRadians) ((angleRadians) * 180.0 / M_PI)





namespace FastGA {

const static uint32_t HILBERT_MAX_32 = std::numeric_limits<uint32_t>::max();

const static double EPSILON = 1e-5;


using MatX3d = std::vector<std::array<double, 3>>;
using MatX3I = std::vector<std::array<size_t, 3>>;
using MatX2d = std::vector<std::array<double, 2>>;
using MatX2ui = std::vector<std::array<uint32_t, 2>>;
struct BBOX
{
    double min_x;
    double min_y; 
    double max_x;
    double max_y;
};

template <class T>
void ScaleItemInPlace(std::array<T, 3> &item, T scalar)
{
    item[0] *= scalar;
    item[1] *= scalar;
    item[2] *= scalar;
}
// inline void ScaleItemInPlace(std::array<double, 3> &item, double scalar)
// {
//     item[0] *= scalar;
//     item[1] *= scalar;
//     item[2] *= scalar;
// }

template<class T, int dim>
T L2Norm(std::array<T, dim> &a)
{
    T norm = 0;
    for(size_t i = 0; i< a.size(); i++)
    {
        norm += a[i] * a[i];
    }
    norm = sqrt(norm);
    return norm;
}

template<class T, int dim>
std::array<T, dim> Mean(std::array<T, dim> &a, std::array<T, dim> &b)
{
    std::array<T, dim> mean;
    for(size_t i = 0; i< a.size(); i++)
    {
        mean[i] = (a[i] + b[i]) / 2.0;
    }
    return mean;
}

template <class T, int dim>
void ScaleArrayInPlace(std::vector<std::array<T, dim>> &array, T scalar)
{
    for (auto &&item: array)
    {
        ScaleItemInPlace(item, scalar);
    }
}

inline void AzimuthProjectionXYZ(double* xyz, double* xy)
{
    double phi = acos(xyz[2]);
    double scale = 0.0;
    if (phi < EPSILON)
    {
        scale = 0.0;
    }
    else
    {
        scale = 1.0 / (sqrt((xyz[0] * xyz[0]) + (xyz[1] * xyz[1])));
    }
    // scale = 1.0 / (sqrt((xyz[0] * xyz[0]) + (xyz[1] * xyz[1])));
    xy[0] = phi * xyz[1] * scale;
    xy[1] = -phi * xyz[0] * scale;
}
inline void ScaleXYToUInt32(const double* xy, uint32_t* scale, double min_x, double min_y, double x_range, double y_range)
{
    scale[0] = static_cast<uint32_t>(((xy[0] - min_x) / x_range) * HILBERT_MAX_32);
    scale[1] = static_cast<uint32_t>(((xy[1] - min_y) / y_range) * HILBERT_MAX_32);
}
inline void AzimuthProjectionPhiTheta(double* pt, double* xy)
{
    xy[0] = pt[0] * sin(pt[1]);
    xy[1] = -pt[0] * cos(pt[1]);
}

inline BBOX InitializeProjection(MatX3d &normals, MatX2d &projection)
{
    size_t N = normals.size();
    double min_x = std::numeric_limits<double>::max();
    double min_y = std::numeric_limits<double>::max();
    double max_x = std::numeric_limits<double>::lowest();
    double max_y = std::numeric_limits<double>::lowest();
    for (size_t i = 0; i < N; i++) 
    {
        AzimuthProjectionXYZ(&normals[i][0], &projection[i][0]);
        if (projection[i][0] < min_x)
        {
            min_x = projection[i][0];
        }
        if(projection[i][0] > max_x)
        {
            max_x = projection[i][0];
        }

        if (projection[i][1] < min_y)
        {
            min_y = projection[i][1];
        }
        if(projection[i][1] > max_y)
        {
            max_y = projection[i][1];
        }

    }
    return {min_x, min_y, max_x, max_y};

}

inline std::tuple<MatX2d, std::vector<uint32_t>> ConvertNormalsToHilbert(MatX3d &normals)
{
    size_t N = normals.size();
    MatX2d projection(N);
    std::vector<uint32_t> hilbert_values(N);

    auto bbox = InitializeProjection(normals, projection);
    std::array<uint32_t, 2> xy_int;
    double x_range = bbox.max_x - bbox.min_x;
    double y_range = bbox.max_y - bbox.min_y;
    // std::cout << "Range: " << x_range << ", " << y_range << std::endl;
    for (size_t i = 0; i < N; i++) 
    {
        ScaleXYToUInt32(&projection[i][0], xy_int.data(), bbox.min_x, bbox.min_y, x_range, y_range);
        hilbert_values[i] = Hilbert::hilbertXYToIndex(16u, xy_int[0], xy_int[1]);
    }
    return std::make_tuple(std::move(projection), hilbert_values);
    
}


}
#endif