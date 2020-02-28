#include "FastGA.hpp"

namespace FastGA {
const std::string test()
{
    return "test";
}

GaussianAccumulator::GaussianAccumulator(int level) : mesh(), buckets()
{
    mesh = FastGA::Ico::RefineIcosahedron(level);
    // buckets = 
}
} // namespace FastGA