
#include "bfgs_torch.h"

#include "../evaluator/eval_vina.h"

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
            const scoring_function &intra_sf)
        {

            pybind11::gil_scoped_release release;

            auto cnfr_1 = cnfr_0.clone().detach().requires_grad_(true);

            auto &&evaluated_1 = pyvina::evaluator::VinaTorchCore_ConformationToEnergy(
                cnfr_1,
                rec,
                lig,
                intra_sf);

            auto &&energy_1 = std::get<0>(evaluated_1);
            auto &&status_1 = std::get<1>(evaluated_1);

            if (!status_1)
            {
                std::cout
                    << "############################################################ \n"
                    << "##### Something strange happened !!!!! \n"
                    << "##### The input conformation for BfgsTorchCore() is invalid \n"
                    << "############################################################ \n"
                    << "##### energy_1: \n"
                    << energy_1 << std::endl
                    << "##### status_1: \n"
                    << status_1 << std::endl
                    << "##### cnfr_1: \n"
                    << cnfr_1 << std::endl
                    << "############################################################ \n"
                    << std::endl;

                return cnfr_0;
            }

            energy_1.backward();
            auto grad_1 = cnfr_1.grad();

            int n_hessian = cnfr_1.size(0);
            auto hessian = torch::eye(n_hessian);

            while (true)
            {
                auto p = -(torch::matmul(hessian,
                                         grad_1.t()));

                auto pg1 = torch::dot(p,
                                      grad_1);
                double alpha = 1.0;
                bool line_search_found = false;

                torch::Tensor cnfr_2;
                torch::Tensor energy_2;
                torch::Tensor grad_2;

                for (int i = 0; i < 5; i++)
                {

                    alpha *= 0.1;

                    // auto cnfr_2 = cnfr_1.clone().detach().requires_grad_(true);
                    cnfr_2 = std::move(
                        cnfr_1.clone().detach().requires_grad_(true));

                    {
                        torch::NoGradGuard no_grad;
                        cnfr_2 += alpha * p;
                    }

                    cnfr_2.requires_grad_(true);

                    auto evaluated_2 = pyvina::evaluator::VinaTorchCore_ConformationToEnergy(
                        cnfr_2,
                        rec,
                        lig,
                        intra_sf);

                    // auto &&energy_2 = std::get<0>(evaluated_2);
                    energy_2 = std::move(
                        std::get<0>(evaluated_2));

                    auto &&status_2 = std::get<1>(evaluated_2);
                    if (!status_2)
                    {
                        continue;
                    }
                    ////////////////////////////////////////
                    /////   FIXME: This might cause a BUG
                    ////////////////////////////////////////
                    if ((energy_2 > energy_1 + 0.0001 * alpha * pg1).item<bool>())
                    {
                        continue;
                    }

                    energy_2.backward();
                    grad_2 = cnfr_2.grad();

                    auto pg2 = torch::dot(p,
                                          grad_2);

                    ////////////////////////////////////////
                    /////   FIXME: This might cause a BUG
                    ////////////////////////////////////////
                    if ((pg2 >= 0.9 * pg1).item<bool>())
                    {
                        line_search_found = true;
                        break;
                    }
                }

                if (!line_search_found)
                {
                    break;
                }

                auto y = grad_2 - grad_1;
                auto minus_hy = -(torch::matmul(hessian,
                                                y));
                auto yhy = -torch::dot(y, minus_hy);

                auto yp = torch::dot(y, p);
                auto reverse_yp = 1.0 / yp;

                auto pco = reverse_yp * (reverse_yp * yhy + alpha);

                for (int i = 0; i < n_hessian; i++)
                {
                    for (int j = 0; j < n_hessian; j++)
                    {
                        hessian[i][j] += (reverse_yp * (minus_hy[i] * p[j] + minus_hy[j] * p[i]) + pco * p[i] * p[j]);
                    }
                }

                cnfr_1 = std::move(cnfr_2);
                energy_1 = std::move(energy_2);
                grad_1 = std::move(grad_2);
            }

            pybind11::gil_scoped_acquire acquire;

            return cnfr_1;
        }

    } // namespace minimizer
} // namespace pyvina
