
#include <iostream>

#include "common.h"

namespace pyvina
{

    namespace common
    {

        int Mr(int x, int y)
        {
            return CommonMr(x, y);
        }

        int Mp(int x, int y)
        {
            return CommonMp(x, y);
        }

    } // namespace common

    int CommonMr(int x, int y)
    {
        if (!(x <= y))
        {
            std::cout
                << " ### ERROR: x > y in CommonMr ### "
                << std::endl;
        }
        return (((y * (y + 1)) >> 1) + x);
    }

    int CommonMp(int x, int y)
    {
        if (x <= y)
        {
            return CommonMr(x, y);
        }
        return CommonMr(y, x);
    }

} // namespace pyvina
