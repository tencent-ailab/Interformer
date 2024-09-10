#ifndef PYVINA_MINIMIZER_BFGS_FORMULA_DERIVATIVE_H_
#define PYVINA_MINIMIZER_BFGS_FORMULA_DERIVATIVE_H_

#include <tuple>

#include "../receptor.hpp"
#include "../ligand.hpp"
#include "../scoring_function.hpp"

namespace pyvina
{
    namespace minimizer
    {

        /////////////////////////////////////////////////////
        /////   remember to use [ const T & ] !!!!!
        /////////////////////////////////////////////////////
        std::vector<double> BfgsFormulaDerivativeCore(
            const std::vector<double> &cnfr_0,
            const receptor &rec,
            const ligand &lig,
            const scoring_function &intra_sf);

    } // namespace minimizer
} // namespace pyvina

#endif // PYVINA_MINIMIZER_BFGS_FORMULA_DERIVATIVE_H_
