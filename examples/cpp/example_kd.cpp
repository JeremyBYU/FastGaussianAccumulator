#include <random>
#include <cmath>
#include <chrono>
#include <vector>
#include <iostream>
#include "FastGA.hpp"

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

int main(int argc, char const *argv[])
{
    auto a = NanoS2ID::S2CellId({{0, 0, 1}});
    auto ga = FastGA::GaussianAccumulatorS2(4, 180.0);
    FastGA::MatX3d normals = initialize_normals(1000, 180, 1);
    std::cout << normals[0] << std::endl;
    auto values = ga.Integrate4(normals, 12);
    std::cout << values[0] << std::endl;
    return 0;
}


