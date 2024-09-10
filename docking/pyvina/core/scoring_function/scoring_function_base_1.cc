
#include "scoring_function_base_1.h"

namespace pyvina
{
    namespace scoring_function
    {

        ScoringFunctionBase1::ScoringFunctionBase1(
            int _num_x_score_types,
            int _num_samples_per_angstrom,
            double _cutoff)
            : num_x_score_types_(_num_x_score_types),
              num_samples_per_angstrom_(_num_samples_per_angstrom),
              cutoff_(_cutoff),

              num_x_score_pairs_(
                  _num_x_score_types * (_num_x_score_types + 1) / 2),
              num_samples_whole_cutoff_(
                  int(_num_samples_per_angstrom *_cutoff *_cutoff + 11)),
              cutoff_square_(
                  _cutoff * _cutoff)
        {
            energy_.clear();
            derivative_.clear();
        }

    } // namespace scoring_function
} // namespace pyvina
