#ifndef PYVINA_GRID_H_
#define PYVINA_GRID_H_

#include <vector>
#include "receptor.hpp"

namespace pyvina
{

    class Grid
    {
    public:
        std::vector<std::vector<double>> energy_;

        Grid(const receptor &rec);
    };

} // namespace pyvina

#endif // PYVINA_GRID_H_