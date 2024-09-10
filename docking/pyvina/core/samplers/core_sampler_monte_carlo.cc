// Copyright 2022 Tencent Inc. All rights reserved.
//
// Author: ruiyuanqian@tencent.com (ruiyuanqian)
//
// 文件注释
// ...

#include "pyvina/core/samplers/core_sampler_monte_carlo.h"
#include "pyvina/core/constants/common.h"
#include "pyvina/core/wrappers_for_exceptions/common.h"
#include "pyvina/core/wrappers_for_random/common.h"

#include <omp.h>
#include <algorithm>
#include <cmath>

namespace pyvina {
namespace core {
namespace samplers {

CoreSamplerMonteCarlo::CoreSamplerMonteCarlo(

    const minimizers::CoreMinimizerBase& core_minimizer_base

    )
    : CoreSamplerBase(core_minimizer_base) {
  ;
}

std::vector<minimizers::RecordPoseAndMore> CoreSamplerMonteCarlo::ThreadsampleGivenMinimizer(

    const minimizers::CoreMinimizerBase& core_minimizer_base,

    const std::array<double, 3>& corner_min, const std::array<double, 3>& corner_max,

    int num_steps_each_monte_carlo

) {
  using pyvina::core::constants::kEnergyInvalid;
  using pyvina::core::constants::kNumMaxKeptEachMonteCarlo;
  using pyvina::core::constants::kNumMaxRerandomizePose;
  using pyvina::core::wrappers_for_random::get_uniform_real_between;

  const auto& core_pdbqt_ligand = core_minimizer_base.core_evaluator_base_.core_pdbqt_ligand_;
  const auto& core_evaluator_base = core_minimizer_base.core_evaluator_base_;

  ////////////////////////////////
  ///// pose 0
  ////////////////////////////////
  auto pose_0 = core_pdbqt_ligand.GetPoseInitial();
  double energy_0 = kEnergyInvalid;
  bool is_valid_0 = false;
  for (int i = 0; i < kNumMaxRerandomizePose; i++) {
    pose_0 = core_pdbqt_ligand.GetPoseRandomInBoundingbox(corner_min, corner_max);
    const auto&& tuple_the_evaluated_0 = core_evaluator_base.evaluate(pose_0);

    energy_0 = std::get<0>(tuple_the_evaluated_0);
    is_valid_0 = std::get<2>(tuple_the_evaluated_0);

    if (is_valid_0) {
      break;
    }
  }
  if (!is_valid_0) {
    pyvina::core::wrappers_for_exceptions::Throw1ExceptionAfterPrinting(
        " :: invalid pose_0 after :: [ " + std::to_string(kNumMaxRerandomizePose) + " ] :: iterations :: ");
  }

  ////////////////////////////////
  ///// monte carlo
  ////////////////////////////////
  std::vector<minimizers::RecordPoseAndMore> records_thread;
  for (int i = 0; i < num_steps_each_monte_carlo; i++) {
    auto pose_1 = pose_0;

    ////////////////////////////////
    ///// pose 2 :: random next
    ////////////////////////////////
    auto pose_2 = pose_1;
    bool is_valid_2 = false;
    for (int j = 0; j < kNumMaxRerandomizePose; j++) {
      pose_2 = core_pdbqt_ligand.GetPoseRandomNext(pose_1);
      is_valid_2 = std::get<2>(core_evaluator_base.evaluate(pose_2));

      if (is_valid_2) {
        break;
      }
    }
    if (!is_valid_2) {
      pyvina::core::wrappers_for_exceptions::Throw1ExceptionAfterPrinting(
          " :: invalid pose_2 after :: [ " + std::to_string(kNumMaxRerandomizePose) + " ] :: iterations :: ");
    }

    ////////////////////////////////
    ///// pose 3 :: minimized
    ////////////////////////////////
    auto tuple_the_minimized_2 = core_minimizer_base.Minimize(pose_2);
    records_thread.push_back(tuple_the_minimized_2);
    auto pose_3 = std::get<0>(tuple_the_minimized_2);
    auto energy_3 = std::get<1>(tuple_the_minimized_2);

    double energy_delta = (energy_0 - energy_3);
    if (

        (energy_delta > 0)

        || (get_uniform_real_between(0.0, 1.0) < std::exp(energy_delta))

    ) {
      energy_0 = energy_3;
      pose_0 = pose_3;
    }
  }

  ////////////////////////////////
  ///// sort
  ////////////////////////////////
  std::sort(records_thread.begin(), records_thread.end(),

            minimizers::CoreMinimizerBase::CompareRecordPoseAndMore);

  if (records_thread.size() > kNumMaxKeptEachMonteCarlo) {
    return std::vector<minimizers::RecordPoseAndMore>(

        records_thread.begin(),

        records_thread.begin() + kNumMaxKeptEachMonteCarlo

    );
  }

  return records_thread;
}

std::vector<minimizers::RecordPoseAndMore> CoreSamplerMonteCarlo::SampleGivenMinimizer(

    const minimizers::CoreMinimizerBase& core_minimizer_base,

    const std::array<double, 3>& corner_min, const std::array<double, 3>& corner_max,

    int num_repeats_monte_carlo, int num_steps_each_monte_carlo

) {
  int i;

  std::vector<std::vector<minimizers::RecordPoseAndMore> > records_from_repeats;
  records_from_repeats.resize(num_repeats_monte_carlo);

#pragma omp parallel for
  for (i = 0; i < num_repeats_monte_carlo; i++) {
//    printf(" ::  start monte carlo [ %d ] :: threadnum = [ %d / %d / %d ] :: \n",

//           i, omp_get_thread_num(), omp_get_max_threads(), omp_get_thread_limit());

    records_from_repeats[i] = CoreSamplerMonteCarlo::ThreadsampleGivenMinimizer(

        core_minimizer_base, corner_min, corner_max, num_steps_each_monte_carlo

    );

//    printf(" :: finish monte carlo [ %d ] :: threadnum = [ %d / %d / %d ] :: \n",

//           i, omp_get_thread_num(), omp_get_max_threads(), omp_get_thread_limit());
  }

  std::vector<minimizers::RecordPoseAndMore> records_merged;
  for (i = 0; i < num_repeats_monte_carlo; i++) {
    records_merged.insert(

        records_merged.end(),

        records_from_repeats[i].begin(), records_from_repeats[i].end()

    );
  }

  std::sort(records_merged.begin(), records_merged.end(),

            minimizers::CoreMinimizerBase::CompareRecordPoseAndMore);
  return records_merged;
}

std::vector<minimizers::RecordPoseAndMore> CoreSamplerMonteCarlo::Sample(

    const std::array<double, 3>& corner_min, const std::array<double, 3>& corner_max,

    int num_repeats_monte_carlo, int num_steps_each_monte_carlo

) const {
  return CoreSamplerMonteCarlo::SampleGivenMinimizer(

      this->core_minimizer_base_,

      corner_min, corner_max,

      num_repeats_monte_carlo, num_steps_each_monte_carlo

  );
}

}  // namespace samplers
}  // namespace core
}  // namespace pyvina
