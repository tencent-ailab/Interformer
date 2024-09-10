#ifndef PYVINA_MINIMIZER_BFGS_TORCH_H_
#define PYVINA_MINIMIZER_BFGS_TORCH_H_

#include <torch/extension.h>

#include "../receptor.hpp"
#include "../ligand.hpp"
#include "../scoring_function.hpp"

namespace pyvina
{
    namespace minimizer
    {

        /////////////////////////////////////////////////////
        /////   remember to use [ const T & ] !!!!!
        /////////////////////////////////////////////////////
        torch::Tensor BfgsTorchCore(
            const torch::Tensor &cnfr_0,
            const receptor &rec,
            const ligand &lig,
            const scoring_function &intra_sf);

    } // namespace minimizer
} // namespace pyvina

#endif // PYVINA_MINIMIZER_BFGS_TORCH_H_