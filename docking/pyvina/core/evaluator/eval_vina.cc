
#include "eval_vina.h"
// #include<torch/torch.h>

namespace pyvina
{

    namespace evaluator
    {

        std::tuple<torch::Tensor,
                   bool,
                   std::vector<torch::Tensor>,
                   torch::Tensor,
                   torch::Tensor>
        VinaTorchCore_ConformationToEnergy(
            const torch::Tensor &cnfr,
            const receptor &rec,
            const ligand &lig,
            const scoring_function &intra_sf)
        {
            auto inter_energy = torch::zeros(1);
            auto intra_energy = torch::zeros(1);

            bool status;
            std::vector<torch::Tensor> heavy_atom_coords;
            auto energy = EvalVina_GetEnergyFromConformation(cnfr,
                                                             rec,
                                                             lig,
                                                             intra_sf,
                                                             &status,
                                                             &heavy_atom_coords);

            return std::make_tuple(std::move(energy),
                                   std::move(status),
                                   std::move(heavy_atom_coords),
                                   std::move(inter_energy),
                                   std::move(intra_energy));
        }

    } // namespace evaluator

    std::tuple<torch::Tensor,
               bool,
               std::vector<torch::Tensor>>
    eval_vina_get_energy_from_conformation(
        const torch::Tensor &cnfr,
        const receptor &rec,
        const ligand &lig,
        const scoring_function &intra_sf)
    {

        bool status;
        std::vector<torch::Tensor> heavy_atom_coords;
        auto energy = EvalVina_GetEnergyFromConformation(cnfr,
                                                         rec,
                                                         lig,
                                                         intra_sf,
                                                         &status,
                                                         &heavy_atom_coords);

        return std::make_tuple(std::move(energy),
                               std::move(status),
                               std::move(heavy_atom_coords));
    }

    torch::Tensor EvalVina_GetEnergyFromConformation(
        const torch::Tensor &cnfr,
        const receptor &rec,
        const ligand &lig,
        const scoring_function &intra_sf,
        bool *status,
        std::vector<torch::Tensor> *out_heavy_atom_coords)
    {
        (*status) = false;
        out_heavy_atom_coords->clear();

        auto inter_energy = torch::zeros(1);
        auto intra_energy = torch::zeros(1);

        std::vector<torch::Tensor> heavy_atom_coords;
        std::vector<torch::Tensor> hydrogen_coords;

        lig.GetCoordsFromConformation(
            cnfr,
            &heavy_atom_coords,
            &hydrogen_coords);

        for (int i = 0; i < lig.num_heavy_atoms; i++)
        {
            auto xyz = torch::zeros(3);
            int xyz_0[3];

            for (int j = 0; j < 3; j++)
            {
                xyz[j] = ((heavy_atom_coords[i][j] - rec.corner0[j]) * rec.granularity_inverse);
                xyz_0[j] = int(
                    xyz[j].item<double>());

                if (xyz_0[j] < 0 || xyz_0[j] + 1 >= rec.num_probes[j])
                {
                    return torch::zeros(1);
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
                int xyz_1[3]{xyz_0[0],
                             xyz_0[1],
                             xyz_0[2]};

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
            auto r2 = torch::dot(
                delta_coord,
                delta_coord);

            intra_energy += intra_sf.GridTtr2(
                lig.heavy_atoms[i].xs,
                lig.heavy_atoms[j].xs,
                r2);
        }

        (*status) = true;
        (*out_heavy_atom_coords) = std::move(heavy_atom_coords);
        return (inter_energy + intra_energy);
    }

} // namespace pyvina
