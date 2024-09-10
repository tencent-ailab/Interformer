
#include "vina_scoring_function.h"

namespace pyvina
{
    namespace scoring_function
    {
        ///// FIXME: Will comment vdw_radii_default_ in the future
        ///// (vdw_radii_default_ will be maintained in python code)
        const array<double, 15> VinaScoringFunction::vdw_radii_default_ = {
            1.9, //   C_H
            1.9, //   C_P
            1.8, //   N_P
            1.8, //   N_D
            1.8, //   N_A

            1.8, //   N_DA
            1.7, //   O_A
            1.7, //   O_DA
            2.0, //   S_P
            2.1, //   P_P

            1.5, //   F_H
            1.8, //  Cl_H
            2.0, //  Br_H
            2.2, //   I_H
            1.2, // Met_D
        };

        ////////////////////////////////////////////////////////////////////////
        /////////   move (num_x_score_types_, and etc) to ScoringFunctionBase1
        ////////////////////////////////////////////////////////////////////////
        // VinaScoringFunction::VinaScoringFunction(
        //     int _num_x_score_types,
        //     int _num_samples_per_angstrom,
        //     double _cutoff)
        //     : num_x_score_types_(_num_x_score_types),
        //       num_samples_per_angstrom_(_num_samples_per_angstrom),
        //       cutoff_(_cutoff),

        //       num_x_score_pairs_(
        //           _num_x_score_types * (_num_x_score_types + 1) / 2),
        //       num_samples_whole_cutoff_(
        //           int(_num_samples_per_angstrom *_cutoff *_cutoff + 11)),
        //       cutoff_square_(
        //           _cutoff * _cutoff)
        // {
        //     ;
        // }
        ////////////////////////////////////////////////////////////////////////

        VinaScoringFunction::VinaScoringFunction(
            int _num_x_score_types,
            int _num_samples_per_angstrom,
            double _cutoff,
            const std::vector<double> &_term_weights,
            const std::vector<double> &_vdw_radii)
            : ScoringFunctionBase1(_num_x_score_types,
                                   _num_samples_per_angstrom,
                                   _cutoff),
              term_weights_(_term_weights),
              vdw_radii_(_vdw_radii)
        {
            if (_num_x_score_types != _vdw_radii.size())
            {
                std::cout
                    << "####################################################" << std::endl
                    << "######  :: ALERT ::" << std::endl
                    << "######  Num of atom types != Size of vdw radii list" << std::endl
                    << "####################################################" << std::endl;
            }
        }

        double VinaScoringFunction::Score(
            int x_score_type_1,
            int x_score_type_2,
            double r2) const
        {
            ;
        }

        torch::Tensor VinaScoringFunction::TorchCore_GridTtr2(
            int x_score_type_1,
            int x_score_type_2,
            const torch::Tensor &r2) const
        {
            ;
        }

        double VinaScoringFunction::VanillaCore_GridTtr2(
            int x_score_type_1,
            int x_score_type_2,
            double r2) const
        {
            ;
        }

    } // namespace scoring_function
} // namespace pyvina
