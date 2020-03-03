
#include <random>
#include <cmath>
#include <chrono>
#include <vector>
#include <iostream>
#include "FastGA/FastGA.hpp"
#include "Hilbert/Hilbert.hpp"

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include "s2/s2earth.h"
#include "s2/s1chord_angle.h"
#include "s2/s2closest_point_query.h"
#include "s2/s2point_index.h"
#include "s2/s2testing.h"


int main(int argc, char const *argv[])
{
    auto GA = FastGA::GaussianAccumulatorOpt(4, 100.0);
    FastGA::MatX3d normals = {{0.99177847, -0.11935933, -0.04613903}, {-1, 0, 0}, {0, 0 , 1}, {1, 0, 0}, {0, 1, 0}, {0, -1, 0}};
    S2PointIndex<int> s2_index;
    for(size_t i = 0; i < GA.buckets.size(); i++)
    {
        auto &bucket_normal = GA.buckets[i].normal;
        S2Point s2_point(bucket_normal[0], bucket_normal[1], bucket_normal[2]);
        s2_index.Add(s2_point, i);
    }

    S2ClosestPointQuery<int> query(&s2_index);
    query.mutable_options()->set_max_results(1);

    // Query Points
    for (size_t i =0; i < normals.size(); i++)
    {
        auto &normal = normals[i];
        S2Point s2_point(normal[0], normal[1], normal[2]);
        S2CellId s2_id(s2_point);
        auto id = s2_id.id();
        S2ClosestPointQuery<int>::PointTarget target(s2_point);
        auto results = query.FindClosestPoints(&target);
        // std::cout << "Result Size: " << results.size() << std::endl;
        if (results.size() > 0)
        {
            auto bucket_point = results[0].point();
            std::cout << "Looking for: " << normal << "; S2ID is: " << id << "; Found " << bucket_point.x() << ", " << bucket_point.y() << ", " << bucket_point.z() << std::endl;
        }
    }


    return 0;
}