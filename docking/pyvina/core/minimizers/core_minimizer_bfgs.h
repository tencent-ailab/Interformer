// Copyright 2022 Tencent Inc. All rights reserved.
//
// Author: ruiyuanqian@tencent.com (ruiyuanqian)
//
// 文件注释
// ...

#ifndef PYVINA_CORE_MINIMIZERS_CORE_MINIMIZER_BFGS_H_
#define PYVINA_CORE_MINIMIZERS_CORE_MINIMIZER_BFGS_H_

#include <array>
#include <iostream>
#include <tuple>
#include <vector>

#include "pyvina/core/evaluators/core_evaluator_base.h"
#include "pyvina/core/minimizers/core_minimizer_base.h"

namespace pyvina {
namespace core {
namespace minimizers {

class CoreMinimizerBfgs : public CoreMinimizerBase {
 public:
  CoreMinimizerBfgs(

      const evaluators::CoreEvaluatorBase& core_evaluator_base

  );

  static std::string GetStrAddressOfEvaluator(

      const evaluators::CoreEvaluatorBase& core_evaluator_base

  );

  static std::tuple<
    bool,
    double,
    std::vector<double>,
    double,
    std::vector<double>
  > Linesearch(

      const int &n_hessian,
      const std::vector<double> &cnfr_1,
      const double &energy_1,
      const std::vector<double> &p,
      const double &pg1,
      const evaluators::CoreEvaluatorBase &core_evaluator_base

  );

  static RecordPoseAndMore MinimizeGivenEvaluator(

      const std::vector<double>& pose_0, const evaluators::CoreEvaluatorBase& core_evaluator_base

  );

  RecordPoseAndMore Minimize(

      const std::vector<double>& pose_0

  ) const;
};

}  // namespace minimizers
}  // namespace core
}  // namespace pyvina

#endif
