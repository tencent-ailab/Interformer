// Copyright 2022 Tencent Inc. All rights reserved.
//
// Author: ruiyuanqian@tencent.com (ruiyuanqian)
//
// 文件注释
// ...

#include "pyvina/core/minimizers/core_minimizer_bfgs.h"

#include <sstream>
#include <stdexcept>
#include <string>

#include "pyvina/core/wrappers_for_exceptions/common.h"

namespace pyvina {
namespace core {
namespace minimizers {

CoreMinimizerBfgs::CoreMinimizerBfgs(

    const evaluators::CoreEvaluatorBase &core_evaluator_base

    )
    : CoreMinimizerBase(core_evaluator_base) {
  ;
}

std::string CoreMinimizerBfgs::GetStrAddressOfEvaluator(

    const evaluators::CoreEvaluatorBase &core_evaluator_base

) {
  std::ostringstream oss;
  oss << ((const void *)(&core_evaluator_base));
  return oss.str();
}

std::tuple<
  bool,
  double,
  std::vector<double>,
  double,
  std::vector<double>
> CoreMinimizerBfgs::Linesearch(

    const int &n_hessian,
    const std::vector<double> &cnfr_1,
    const double &energy_1,
    const std::vector<double> &p,
    const double &pg1,
    const evaluators::CoreEvaluatorBase &core_evaluator_base

){
  bool line_search_found = false;
  double alpha = 1.0;

  std::vector<double> cnfr_2;
  double energy_2 = 0.0;
  std::vector<double> grad_2;

  ////////////////////////////////////////
  ///// extracted to Linesearch()
  ////////////////////////////////////////
  for (int i = 0; i < 5; i++) {
    alpha *= 0.1;

    // auto cnfr_2 = cnfr_1.clone().detach().requires_grad_(true);
    // cnfr_2 = std::move(
    //     cnfr_1.clone().detach().requires_grad_(true));
    cnfr_2 = std::vector<double>(cnfr_1);

    // {
    //     torch::NoGradGuard no_grad;
    //     cnfr_2 += alpha * p;
    // }
    for (int j = 0; j < n_hessian; j++) {
      if (core_evaluator_base.core_pdbqt_ligand_.is_translation_fixed) {
        if (j < 3) {
          continue;
        }
      }

      cnfr_2[j] += alpha * p[j];
    }

    // cnfr_2.requires_grad_(true);

    auto evaluated_2 = core_evaluator_base.evaluate(cnfr_2);

    // auto &&energy_2 = std::get<0>(evaluated_2);
    energy_2 = std::move(std::get<0>(evaluated_2));

    auto &&status_2 = std::get<2>(evaluated_2);
    if (!status_2) {
      continue;
    }
    ////////////////////////////////////////
    /////   FIXME: This might cause a BUG
    ////////////////////////////////////////
    // if ((energy_2 > energy_1 + 0.0001 * alpha * pg1).item<bool>())
    // {
    //     continue;
    // }
    if (energy_2 > energy_1 + 0.0001 * alpha * pg1) {
      continue;
    }

    // energy_2.backward();
    // grad_2 = cnfr_2.grad();
    grad_2 = std::move(std::get<1>(evaluated_2));

    // auto pg2 = torch::dot(p,
    //                       grad_2);
    double pg2 = 0.0;
    for (int j = 0; j < n_hessian; j++) {
      pg2 += p[j] * grad_2[j];
    }

    ////////////////////////////////////////
    /////   FIXME: This might cause a BUG
    ////////////////////////////////////////
    // if ((pg2 >= 0.9 * pg1).item<bool>())
    // {
    //     line_search_found = true;
    //     break;
    // }
    if (pg2 >= 0.9 * pg1) {
      line_search_found = true;
      break;
    }
  }
  ////////////////////////////////////////
  ///// extracted to Linesearch()
  ////////////////////////////////////////

  return std::make_tuple(
    line_search_found,
    alpha,
    cnfr_2,
    energy_2,
    grad_2
  );

}

/////////////////////////////////////////////////////
/////   remember to use [ const T & ] !!!!!
/////////////////////////////////////////////////////
RecordPoseAndMore CoreMinimizerBfgs::MinimizeGivenEvaluator(

    const std::vector<double> &cnfr_0,

    const evaluators::CoreEvaluatorBase &core_evaluator_base

) {
  // pybind11::gil_scoped_release release;

  // auto cnfr_1 = cnfr_0.clone().detach().requires_grad_(true);
  auto cnfr_1 = std::vector<double>(cnfr_0);

  auto &&evaluated_1 = core_evaluator_base.evaluate(cnfr_1);

  auto &&energy_1 = std::get<0>(evaluated_1);
  auto &&status_1 = std::get<2>(evaluated_1);

  if (!status_1) {
    pyvina::core::wrappers_for_exceptions::Throw1ExceptionAfterPrinting(

        " :: pose_0 for CoreMinimizerBfgs() is invalid :: "

    );
  }

  // energy_1.backward();
  // auto grad_1 = cnfr_1.grad();
  auto grad_1 = std::get<1>(evaluated_1);

  // int n_hessian = cnfr_1.size(0);
  // auto hessian = torch::eye(n_hessian);
  int n_hessian = cnfr_1.size();

  auto hessian = std::vector<std::vector<double>>(n_hessian, std::vector<double>(n_hessian, 0.0));
  for (int i = 0; i < n_hessian; i++) {
    hessian[i][i] = 1.0;
  }

  while (true) {
    // auto p = -(torch::matmul(hessian,
    //                          grad_1.t()));
    auto p = std::vector<double>(n_hessian, 0.0);
    for (int i = 0; i < n_hessian; i++) {
      for (int j = 0; j < 1; j++) {
        for (int k = 0; k < n_hessian; k++) {
          // p[i][j] += -(hessian[i][k] * grad_1[k][j]);
          p[i] += -(hessian[i][k] * grad_1[k]);
        }
      }
    }

    // auto pg1 = torch::dot(p,
    //                       grad_1);
    double pg1 = 0.0;
    for (int i = 0; i < n_hessian; i++) {
      pg1 += p[i] * grad_1[i];
    }

    bool line_search_found = false;
    double alpha = 1.0;

    // torch::Tensor cnfr_2;
    // torch::Tensor energy_2;
    // torch::Tensor grad_2;
    std::vector<double> cnfr_2;
    double energy_2 = 0.0;
    std::vector<double> grad_2;

    std::tie(
      line_search_found,
      alpha,
      cnfr_2,
      energy_2,
      grad_2
    ) = CoreMinimizerBfgs::Linesearch(
      n_hessian,
      cnfr_1,
      energy_1,
      p,
      pg1,
      core_evaluator_base
    );

    // ////////////////////////////////////////
    // ///// extracted to Linesearch()
    // ////////////////////////////////////////
    // for (int i = 0; i < 5; i++) {
    //   alpha *= 0.1;

    //   // auto cnfr_2 = cnfr_1.clone().detach().requires_grad_(true);
    //   // cnfr_2 = std::move(
    //   //     cnfr_1.clone().detach().requires_grad_(true));
    //   cnfr_2 = std::vector<double>(cnfr_1);

    //   // {
    //   //     torch::NoGradGuard no_grad;
    //   //     cnfr_2 += alpha * p;
    //   // }
    //   for (int j = 0; j < n_hessian; j++) {
    //     if (core_evaluator_base.core_pdbqt_ligand_.is_translation_fixed) {
    //       if (j < 3) {
    //         continue;
    //       }
    //     }

    //     cnfr_2[j] += alpha * p[j];
    //   }

    //   // cnfr_2.requires_grad_(true);

    //   auto evaluated_2 = core_evaluator_base.evaluate(cnfr_2);

    //   // auto &&energy_2 = std::get<0>(evaluated_2);
    //   energy_2 = std::move(std::get<0>(evaluated_2));

    //   auto &&status_2 = std::get<2>(evaluated_2);
    //   if (!status_2) {
    //     continue;
    //   }
    //   ////////////////////////////////////////
    //   /////   FIXME: This might cause a BUG
    //   ////////////////////////////////////////
    //   // if ((energy_2 > energy_1 + 0.0001 * alpha * pg1).item<bool>())
    //   // {
    //   //     continue;
    //   // }
    //   if (energy_2 > energy_1 + 0.0001 * alpha * pg1) {
    //     continue;
    //   }

    //   // energy_2.backward();
    //   // grad_2 = cnfr_2.grad();
    //   grad_2 = std::move(std::get<1>(evaluated_2));

    //   // auto pg2 = torch::dot(p,
    //   //                       grad_2);
    //   double pg2 = 0.0;
    //   for (int j = 0; j < n_hessian; j++) {
    //     pg2 += p[j] * grad_2[j];
    //   }

    //   ////////////////////////////////////////
    //   /////   FIXME: This might cause a BUG
    //   ////////////////////////////////////////
    //   // if ((pg2 >= 0.9 * pg1).item<bool>())
    //   // {
    //   //     line_search_found = true;
    //   //     break;
    //   // }
    //   if (pg2 >= 0.9 * pg1) {
    //     line_search_found = true;
    //     break;
    //   }
    // }
    // ////////////////////////////////////////
    // ///// extracted to Linesearch()
    // ////////////////////////////////////////

    if (!line_search_found) {
      break;
    }

    // auto y = grad_2 - grad_1;
    auto y = std::vector<double>(n_hessian, 0.0);
    for (int i = 0; i < n_hessian; i++) {
      y[i] = grad_2[i] - grad_1[i];
    }

    // auto minus_hy = -(torch::matmul(hessian,
    //                                 y));
    auto minus_hy = std::vector<double>(n_hessian, 0.0);
    for (int i = 0; i < n_hessian; i++) {
      for (int j = 0; j < 1; j++) {
        for (int k = 0; k < n_hessian; k++) {
          // minus_hy[i][j] += -(hessian[i][k] * y[k][j]);
          minus_hy[i] += -(hessian[i][k] * y[k]);
        }
      }
    }

    // auto yhy = -torch::dot(y, minus_hy);
    double yhy = 0.0;
    for (int i = 0; i < n_hessian; i++) {
      yhy += -(y[i] * minus_hy[i]);
    }

    // auto yp = torch::dot(y, p);
    double yp = 0.0;
    for (int i = 0; i < n_hessian; i++) {
      yp += y[i] * p[i];
    }
    auto reverse_yp = 1.0 / yp;

    auto pco = reverse_yp * (reverse_yp * yhy + alpha);

    for (int i = 0; i < n_hessian; i++) {
      for (int j = 0; j < n_hessian; j++) {
        hessian[i][j] += (reverse_yp * (minus_hy[i] * p[j] + minus_hy[j] * p[i]) + pco * p[i] * p[j]);
      }
    }

    cnfr_1 = std::move(cnfr_2);
    energy_1 = std::move(energy_2);
    grad_1 = std::move(grad_2);
  }

  // pybind11::gil_scoped_acquire acquire;

  return CoreMinimizerBase::CreateRecordPoseAndMore(cnfr_1, energy_1);
  // return std::make_tuple(cnfr_1, energy_1);
}

RecordPoseAndMore CoreMinimizerBfgs::Minimize(

    const std::vector<double> &pose_0

) const {
  return this->MinimizeGivenEvaluator(pose_0, this->core_evaluator_base_);
}

}  // namespace minimizers
}  // namespace core
}  // namespace pyvina
