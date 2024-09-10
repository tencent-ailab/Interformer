
#include "evaluator_vina_vanilla.h"

namespace pyvina
{
    namespace evaluator
    {

        std::tuple<double,
                   bool,
                   std::vector<std::array<double, 3>>,
                   double,
                   double>
        VinaVanillaCore_ConformationToEnergy(
            const std::vector<double> &cnfr,
            const receptor &rec,
            const ligand &lig,
            const scoring_function &intra_sf)
        {
            ///// conformation (6+k) to coords
            /////////////////////////////////////////
            auto &&ha_and_hy_coords = lig.VanillaCore_ConformationToCoords(
                cnfr);
            auto &&heavy_atom_coords = ha_and_hy_coords.first;

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
                            0.0);
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
            }

            return std::make_tuple((inter_energy + intra_energy),
                                   true,
                                   std::move(heavy_atom_coords),
                                   inter_energy,
                                   intra_energy);
        }

    } // namespace evaluator
} // namespace pyvina
