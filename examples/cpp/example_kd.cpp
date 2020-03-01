#include "FastGA/FastGA.hpp"

int main(int argc, char const *argv[])
{
    auto ga = FastGA::GaussianAccumulatorKD(4, 100.0);
    FastGA::MatX3d normals = {{-1, 0, 0}, {0, 0 , 1}, {1, 0, 0}, {0, 1, 0}, {0, -1, 0}};
    auto values = ga.Integrate(normals);
    return 0;
}


