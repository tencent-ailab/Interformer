// Copyright 2022 Tencent Inc. All rights reserved.
//
// Author: ruiyuanqian@tencent.com (ruiyuanqian)
//
// 文件注释
// ...

#include "pyvina/core/minimizers/core_minimizer_base.h"

namespace pyvina {
namespace core {
namespace minimizers {

RecordPoseAndMore CoreMinimizerBase::CreateRecordPoseAndMore(

    const std::vector<double>& pose,

    double energy

) {
  return std::make_tuple(pose, energy);
}

bool CoreMinimizerBase::CompareRecordPoseAndMore(

    const RecordPoseAndMore& a_record, const RecordPoseAndMore& b_record

) {
  return (std::get<1>(a_record)) < (std::get<1>(b_record));
}

CoreMinimizerBase::CoreMinimizerBase(

    const evaluators::CoreEvaluatorBase& core_evaluator_base

    )
    : core_evaluator_base_{core_evaluator_base} {
  ;
}

}  // namespace minimizers
}  // namespace core
}  // namespace pyvina
