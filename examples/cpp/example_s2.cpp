
#include <random>
#include <cmath>
#include <chrono>
#include <vector>
#include <iostream>
#include "FastGA.hpp"
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
    FastGA::MatX3d normals = {{0.99177847, -0.11935933, -0.04613903}, {1, 1, 1}, {-1, 0, 0}, {0, 0 , 1}, {1, 0, 0}, {0, 1, 0}, {0, -1, 0}};
    // S2PointIndex<int> s2_index;
    // for(size_t i = 0; i < GA.buckets.size(); i++)
    // {
    //     auto &bucket_normal = GA.buckets[i].normal;
    //     S2Point s2_point(bucket_normal[0], bucket_normal[1], bucket_normal[2]);
    //     s2_index.Add(s2_point, i);
    // }

    // S2ClosestPointQuery<int> query(&s2_index);
    // query.mutable_options()->set_max_results(1);

    // Query Points
    for (size_t i =0; i < normals.size(); i++)
    {
        auto &normal = normals[i];
        S2Point s2_point(normal[0], normal[1], normal[2]);
        S2CellId s2_id(s2_point);
        auto id = s2_id.id();
        auto nano_id = NanoS2ID::S2CellId(normal);
        double pu = 0.0;
        double pv = 0.0;
        auto face_s2 = S2::XYZtoFaceUV(s2_point, &pu, &pv);
        auto st_u = S2::UVtoST(pu);
        auto st_v = S2::UVtoST(pv);
        auto ij_u = S2::STtoIJ(st_u);
        auto ij_v = S2::STtoIJ(st_v);
        std::cout << "S2      - ID " << id << ";Face: " << face_s2 << "; u:" << pu << " ;v: " << pv << "; st_u:" << st_u << " ; st_v: " << st_v << "; ij_u:" << ij_u << " ; ij_v: " << ij_v << std::endl;
        face_s2 = NanoS2ID::XYZtoFaceUV(normal, &pu, &pv);
        st_u = NanoS2ID::UVtoST(pu);
        st_v = NanoS2ID::UVtoST(pv);
        ij_u = NanoS2ID::STtoIJ(st_u);
        ij_v = NanoS2ID::STtoIJ(st_v);
        std::cout << "Nano S2 - ID " << nano_id << ";Face: " << face_s2 << "; u:" << pu << " ;v: " << pv << "; st_u:" << st_u << " ; st_v: " << st_v << "; ij_u:" << ij_u << " ; ij_v: " << ij_v << std::endl;
        // S2ClosestPointQuery<int>::PointTarget target(s2_point);
        // auto results = query.FindClosestPoints(&target);
        // // std::cout << "Result Size: " << results.size() << std::endl;
        // if (results.size() > 0)
        // {
        //     auto bucket_point = results[0].point();
        //     std::cout << "Looking for: " << normal << "; S2ID is: " << id << "; NanoS2ID is: " << nano_id  << "; Found " << bucket_point.x() << ", " << bucket_point.y() << ", " << bucket_point.z() << std::endl;
        // }
    }


    return 0;
}