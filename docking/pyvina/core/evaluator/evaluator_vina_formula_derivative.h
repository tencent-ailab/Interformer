#ifndef PYVINA_EVALUATOR_EVALUATOR_VINA_FORMULA_DERIVATIVE_H_
#define PYVINA_EVALUATOR_EVALUATOR_VINA_FORMULA_DERIVATIVE_H_

#include <array>
#include <vector>

#include "../ligand.hpp"
#include "../receptor.hpp"
#include "../scoring_function.hpp"

namespace pyvina
{

    namespace evaluator
    {

        std::tuple<double,
                   bool,
                   std::vector<std::array<double, 3>>,
                   double,
                   double,
                   std::vector<double>>
        VinaFormulaDerivativeCore_ConformationToEnergy(
            const std::vector<double> &cnfr,
            const receptor &rec,
            const ligand &lig,
            const scoring_function &intra_sf);

    } // namespace evaluator

} // namespace pyvina

#endif // PYVINA_EVALUATOR_EVALUATOR_VINA_FORMULA_DERIVATIVE_H_
