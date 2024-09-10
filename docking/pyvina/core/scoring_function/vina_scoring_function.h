#ifndef PYVINA_SCORING_FUNCTION_VINA_SCORING_FUNCTION_H_
#define PYVINA_SCORING_FUNCTION_VINA_SCORING_FUNCTION_H_

// FIXME: why <array> doesn't work?
// #include <array>
#include "../array.hpp"
#include <vector>
#include <torch/torch.h>

#include "scoring_function_base_1.h"

namespace pyvina
{
    namespace scoring_function
    {
        class VinaScoringFunction : public ScoringFunctionBase1
        {
        public:
            ///// FIXME: Will comment vdw_radii_default_ in the future
            ///// (vdw_radii_default_ will be maintained in python code)
            static const array<double, 15> vdw_radii_default_;

            const std::vector<double> term_weights_;
            const std::vector<double> vdw_radii_;

            ////////////////////////////////////////////////////////////////
            /////////   move (num_x_score_types_, and etc) to ScoringFunctionBase1
            ////////////////////////////////////////////////////////////////
            // std::vector<std::vector<double>> energy_;
            // std::vector<std::vector<double>> derivative_;

            // const int num_x_score_types_;
            // const int num_samples_per_angstrom_;
            // const double cutoff_;

            // const int num_x_score_pairs_;
            // const int num_samples_whole_cutoff_;
            // const double cutoff_square_;
            ////////////////////////////////////////////////////////////////

            VinaScoringFunction(
                int _num_x_score_types,
                int _num_samples_per_angstrom,
                double _cutoff,
                const std::vector<double> &_term_weights,
                const std::vector<double> &_vdw_radii);

            double Score(
                int x_score_type_1,
                int x_score_type_2,
                double r2) const;

            torch::Tensor TorchCore_GridTtr2(
                int x_score_type_1,
                int x_score_type_2,
                const torch::Tensor &r2) const;

            double VanillaCore_GridTtr2(
                int x_score_type_1,
                int x_score_type_2,
                double r2) const;
        };
    } // namespace scoring_function
} // namespace pyvina

#endif // PYVINA_SCORING_FUNCTION_VINA_SCORING_FUNCTION_H_
