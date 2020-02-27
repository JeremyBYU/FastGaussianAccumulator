#include "FastGA.hpp"

namespace FastGA {
const std::string test()
{
    return "test";
}

GaussianAccumulator::GaussianAccumulator(int level) : mesh()
{
    mesh = FastGA::Ico::RefineIcosahedron(level);
}
} // namespace FastGA