#ifndef FASTGA_GA_HPP
#define FASTGA_GA_HPP

#include <string>
#include "FastGA/Ico.hpp"

namespace FastGA {
const std::string test();

class GaussianAccumulator
{

  public:
    Ico::IcoMesh mesh;
    GaussianAccumulator(int level = 1);

  private:
};

// GaussianAccumulator::GaussianAccumulator(int level) : mesh()
// {
//     mesh = FastGA::Ico::RefineIcosahedron(level);
// }

} // namespace FastGA
#endif
