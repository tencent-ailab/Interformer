// Copyright 2022 Tencent Inc. All rights reserved.
//
// Author: ruiyuanqian@tencent.com (ruiyuanqian)
//
// 文件注释
// ...

#ifndef PYVINA_CORE_SAMPLERS_CORE_SAMPLER_MONTE_CARLO_H_
#define PYVINA_CORE_SAMPLERS_CORE_SAMPLER_MONTE_CARLO_H_

#include <array>
#include <iostream>
#include <tuple>
#include <vector>

#include "pyvina/core/minimizers/core_minimizer_base.h"
#include "pyvina/core/samplers/core_sampler_base.h"

namespace pyvina {
namespace core {
namespace samplers {

class CoreSamplerMonteCarlo : public CoreSamplerBase {
 public:
  CoreSamplerMonteCarlo(

      const minimizers::CoreMinimizerBase& core_minimizer_base

  );

  static std::vector<minimizers::RecordPoseAndMore> ThreadsampleGivenMinimizer(

      const minimizers::CoreMinimizerBase& core_minimizer_base,

      const std::array<double, 3>& corner_min, const std::array<double, 3>& corner_max,

      int num_steps_each_monte_carlo

  );

  static std::vector<minimizers::RecordPoseAndMore> SampleGivenMinimizer(

      const minimizers::CoreMinimizerBase& core_minimizer_base,

      const std::array<double, 3>& corner_min, const std::array<double, 3>& corner_max,

      int num_repeats_monte_carlo, int num_steps_each_monte_carlo

  );

  std::vector<minimizers::RecordPoseAndMore> Sample(

      const std::array<double, 3>& corner_min, const std::array<double, 3>& corner_max,

      int num_repeats_monte_carlo, int num_steps_each_monte_carlo

  ) const;
};

}  // namespace samplers
}  // namespace core
}  // namespace pyvina

#endif
