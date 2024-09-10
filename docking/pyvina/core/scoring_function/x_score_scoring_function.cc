
#include "vina_scoring_function.h"
#include "x_score_scoring_function.h"

namespace pyvina
{
    namespace scoring_function
    {

        XScoreScoringFunction::XScoreScoringFunction(
            int _num_x_score_types,
            int _num_samples_per_angstrom,
            double _cutoff)
            : ScoringFunctionBase1(
                  _num_x_score_types,
                  _num_samples_per_angstrom,
                  _cutoff)
        {
            ;
        }

        double XScoreScoringFunction::Score(
            int x_score_type_1,
            int x_score_type_2,
            double r2) const
        {
            if (r2 > cutoff_square_)
            {
                return 0.0;
            }

            auto d0 = (VinaScoringFunction::vdw_radii_default_[x_score_type_1] +
                       VinaScoringFunction::vdw_radii_default_[x_score_type_2]);

            // Lennard-Jones 8-4 potential
            auto tmp1 = d0 * d0 / r2;
            tmp1 = tmp1 * tmp1;
            auto tmp2 = tmp1 * tmp1;
            return tmp2 - 2.00 * tmp1;
        }

        torch::Tensor XScoreScoringFunction::TorchCore_GridTtr2(
            int x_score_type_1,
            int x_score_type_2,
            const torch::Tensor &r2) const
        {
            ;
        }

        double XScoreScoringFunction::VanillaCore_GridTtr2(
            int x_score_type_1,
            int x_score_type_2,
            double r2) const
        {
            ;
        }

    } // namespace scoring_function
} // namespace pyvina
