#ifndef PYVINA_SCORING_FUNCTION_SCORING_FUNCTION_BASE_1_H_
#define PYVINA_SCORING_FUNCTION_SCORING_FUNCTION_BASE_1_H_

#include <torch/torch.h>

namespace pyvina
{
    namespace scoring_function
    {
        class ScoringFunctionBase1
        {
        public:
            std::vector<std::vector<double>> energy_;
            std::vector<std::vector<double>> derivative_;

            const int num_x_score_types_;
            const int num_samples_per_angstrom_;
            const double cutoff_;

            const int num_x_score_pairs_;
            const int num_samples_whole_cutoff_;
            const double cutoff_square_;

            ScoringFunctionBase1(
                int _num_x_score_types,
                int _num_samples_per_angstrom,
                double _cutoff);

            virtual double Score(
                int x_score_type_1,
                int x_score_type_2,
                double r2) const = 0;

            virtual torch::Tensor TorchCore_GridTtr2(
                int x_score_type_1,
                int x_score_type_2,
                const torch::Tensor &r2) const = 0;

            virtual double VanillaCore_GridTtr2(
                int x_score_type_1,
                int x_score_type_2,
                double r2) const = 0;
        };
    } // namespace scoring_function
} // namespace pyvina

#endif // PYVINA_SCORING_FUNCTION_SCORING_FUNCTION_BASE_1_H_
