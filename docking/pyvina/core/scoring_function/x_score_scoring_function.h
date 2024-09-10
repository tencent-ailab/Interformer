#ifndef PYVINA_SCORING_FUNCTION_X_SCORE_SCORING_FUNCTION_H_
#define PYVINA_SCORING_FUNCTION_X_SCORE_SCORING_FUNCTION_H_

#include <vector>
#include <torch/torch.h>

#include "scoring_function_base_1.h"

namespace pyvina
{
    namespace scoring_function
    {
        class XScoreScoringFunction : public ScoringFunctionBase1
        {
        public:

            XScoreScoringFunction(
                int _num_x_score_types,
                int _num_samples_per_angstrom,
                double _cutoff);

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

#endif // PYVINA_SCORING_FUNCTION_X_SCORE_SCORING_FUNCTION_H_
