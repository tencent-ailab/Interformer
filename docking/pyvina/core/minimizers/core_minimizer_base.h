// Copyright 2022 Tencent Inc. All rights reserved.
//
// Author: ruiyuanqian@tencent.com (ruiyuanqian)
//
// 文件注释
// ...

#ifndef PYVINA_CORE_MINIMIZERS_CORE_MINIMIZER_BASE_H_
#define PYVINA_CORE_MINIMIZERS_CORE_MINIMIZER_BASE_H_

#include <array>
#include <iostream>
#include <tuple>
#include <vector>

#include "pyvina/core/evaluators/core_evaluator_base.h"

namespace pyvina {
namespace core {
namespace minimizers {

using RecordPoseAndMore = std::tuple<

    std::vector<double>,

    double

    >;

class CoreMinimizerBase {
 public:
  static RecordPoseAndMore CreateRecordPoseAndMore(

      const std::vector<double>&,

      double

  );
  static bool CompareRecordPoseAndMore(

      const RecordPoseAndMore& a_record, const RecordPoseAndMore& b_record

  );

  const evaluators::CoreEvaluatorBase& core_evaluator_base_;

  CoreMinimizerBase(

      const evaluators::CoreEvaluatorBase& core_evaluator_base

  );

  virtual RecordPoseAndMore Minimize(

      const std::vector<double>& pose_0

  ) const = 0;
};

}  // namespace minimizers
}  // namespace core
}  // namespace pyvina

#endif
