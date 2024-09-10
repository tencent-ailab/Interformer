#ifndef PYVINA_EVALUATOR_EVAL_VINA_H_
#define PYVINA_EVALUATOR_EVAL_VINA_H_

#include <torch/extension.h>
#include <vector>

#include "../ligand.hpp"
#include "../receptor.hpp"
#include "../scoring_function.hpp"

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
            const scoring_function &intra_sf);

    } // namespace evaluator

    

    ////////////////////////////////////////////////////////////////////////////////////////
    ///// FIXME: It's not good to naming vars as cnfr, rec or lig
    ///// But currently conformation, receptor and ligand have been used as class names
    ////////////////////////////////////////////////////////////////////////////////////////
    // torch::Tensor eval_vina(
    //     const torch::Tensor & cnfr,//conformation
    //     const receptor & rec,
    //     const ligand & lig
    // );
    // torch::Tensor EvalVina(const torch::Tensor &cnfr,
    //                        const receptor &rec,
    //                        const ligand &lig);
    //
    ////////////////////////////////////////////////////////////////////////////////////////
    // torch::Tensor ComputeEnergyVina();
    ////////////////////////////////////////////////////////////////////////////////////////
    // Difficult to implement a eval_vina equal to python version
    // Only implement computing part
    ////////////////////////////////////////////////////////////////////////////////////////

    torch::Tensor EvalVina_GetEnergyFromConformation(
        const torch::Tensor &cnfr,
        const receptor &rec,
        const ligand &lig,
        const scoring_function &intra_sf,
        bool *status,
        std::vector<torch::Tensor> *out_heavy_atom_coords);

    std::tuple<torch::Tensor,
               bool,
               std::vector<torch::Tensor>>
    eval_vina_get_energy_from_conformation(
        const torch::Tensor &cnfr,
        const receptor &rec,
        const ligand &lig,
        const scoring_function &intra_sf);

} // namespace pyvina

#endif // PYVINA_EVALUATOR_EVAL_VINA_H_