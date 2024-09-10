
#include "evaluator_vina_formula_derivative.h"

#include "../differentiation/coord_differentiation.h"

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
            const scoring_function &intra_sf)
        {
            ///// derivatives of energy with repect to '6+K'
            /////////////////////////////////////////
            const int num_degrees_of_freedom = cnfr.size();
            auto derivatives_energy = std::vector<double>(
                num_degrees_of_freedom,
                0.0);

            ///// conformation (6+k) to coords
            /////////////////////////////////////////
            auto &&ha_hy_coords_and_ha_derivatives = lig.FormulaDerivativeCore_ConformationToCoords(
                cnfr);
            auto &&heavy_atom_coords = std::get<0>(
                ha_hy_coords_and_ha_derivatives);
            auto &&derivatives_heavy_atom_coord = std::get<2>(
                ha_hy_coords_and_ha_derivatives);

            ///// coords to energy
            /////////////////////////////////////////
            double inter_energy = 0.0;
            double intra_energy = 0.0;

            for (int i = 0; i < lig.num_heavy_atoms; i++)
            {
                std::array<double, 3> xyz = {
                    0.0, 0.0, 0.0};
                std::array<int, 3> xyz_0 = {
                    0, 0, 0};

                for (int j = 0; j < 3; j++)
                {
                    xyz[j] = ((heavy_atom_coords[i][j] - rec.corner0[j]) * rec.granularity_inverse);
                    xyz_0[j] = int(xyz[j]);

                    if (xyz_0[j] < 0 || xyz_0[j] + 1 >= rec.num_probes[j])
                    {
                        // return torch::zeros(1);
                        return std::make_tuple(
                            0.0,
                            false,
                            std::move(heavy_atom_coords),
                            0.0,
                            0.0,
                            std::vector<double>(
                                num_degrees_of_freedom,
                                0.0));
                    }
                }

                auto x_score_type = lig.heavy_atoms[i].xs;
                auto energy_xyz_0 = rec.GridTxyz(
                    x_score_type,
                    xyz_0[0],
                    xyz_0[1],
                    xyz_0[2]);
                inter_energy += energy_xyz_0;
                for (int j = 0; j < 3; j++)
                {
                    // int xyz_1[3]{xyz_0[0],
                    //              xyz_0[1],
                    //              xyz_0[2]};
                    auto xyz_1 = xyz_0;

                    xyz_1[j] += 1;

                    auto delta_energy = (rec.GridTxyz(
                                             x_score_type,
                                             xyz_1[0],
                                             xyz_1[1],
                                             xyz_1[2]) -
                                         energy_xyz_0);

                    inter_energy += (delta_energy * (xyz[j] - xyz_0[j]));

                    for (int degree_index = 0;
                         degree_index < num_degrees_of_freedom;
                         degree_index++)
                    {
                        derivatives_energy[degree_index] += (delta_energy *
                                                             rec.granularity_inverse *
                                                             derivatives_heavy_atom_coord[i][j][degree_index]);
                    }
                }
            }

            for (auto &&a_pair : lig.interacting_pairs)
            {
                auto i = a_pair.i0;
                auto j = a_pair.i1;

                auto delta_coord = heavy_atom_coords[i] - heavy_atom_coords[j];
                // auto r2 = torch::dot(delta_coord,
                //                      delta_coord);
                double r2 = (delta_coord[0] * delta_coord[0] +
                             delta_coord[1] * delta_coord[1] +
                             delta_coord[2] * delta_coord[2]);

                intra_energy += intra_sf.VanillaCore_GridTtr2(
                    lig.heavy_atoms[i].xs,
                    lig.heavy_atoms[j].xs,
                    r2);

                // auto derivative_to_r2 = (

                //     intra_sf.VanillaCore_GridTtr2(
                //         lig.heavy_atoms[i].xs,
                //         lig.heavy_atoms[j].xs,
                //         r2 + 0.005)

                //     -

                //     intra_sf.VanillaCore_GridTtr2(
                //         lig.heavy_atoms[i].xs,
                //         lig.heavy_atoms[j].xs,
                //         r2)

                // )/0.005;

                auto derivative_to_r2 = intra_sf.FormulaDerivativeCore_GridTtr2DerivativeToR2(
                    lig.heavy_atoms[i].xs,
                    lig.heavy_atoms[j].xs,
                    r2);

                for (int degree_index = 0;
                     degree_index < num_degrees_of_freedom;
                     degree_index++)
                {
                    for (int xyz_index = 0;
                         xyz_index < 3;
                         xyz_index++)
                    {
                        derivatives_energy[degree_index] += (

                            derivative_to_r2 *
                            (2 * delta_coord[xyz_index]) *
                            (derivatives_heavy_atom_coord[i][xyz_index][degree_index] -
                             derivatives_heavy_atom_coord[j][xyz_index][degree_index])

                        );
                    }
                }

            }

            return std::make_tuple((inter_energy + intra_energy),
                                   true,
                                   std::move(heavy_atom_coords),
                                   inter_energy,
                                   intra_energy,
                                   derivatives_energy);
        }

    } // namespace evaluator
} // namespace pyvina
