// Copyright 2022 Tencent Inc. All rights reserved.
//
// Author: ruiyuanqian@tencent.com (ruiyuanqian)
//
// 文件注释
// ...

#include "pyvina/core/evaluators/core_evaluator_base.h"

#include <cmath>
#include <sstream>
#include <string>

namespace pyvina {
namespace core {
namespace evaluators {

CoreEvaluatorBase::CoreEvaluatorBase(

    const ligand& core_pdbqt_ligand

    )
    : core_pdbqt_ligand_{core_pdbqt_ligand} {
  ;
}

std::string CoreEvaluatorBase::get_str_address(

) const {
  std::ostringstream oss;
  oss << ((const void*)this);
  return oss.str();
}

void CoreEvaluatorBase::SetOption(const Option4Evaluator& option_new) {
  this->option_ = option_new;
  this->is_option_set_ = true;
}
void CoreEvaluatorBase::ResetOption() {
  this->option_ = Option4Evaluator();
  this->is_option_set_ = false;
}
Option4Evaluator CoreEvaluatorBase::GetOption() const { return this->option_; }

std::string CoreEvaluatorBase::FindInOption(

    const Option4Evaluator& a_map, const std::string& a_key

) const {
  auto it = a_map.find(a_key);
  if (it == a_map.end()) {
    throw std::out_of_range("KEY NOT FOUND IN OPTION");
  }
  return it->second;
}

TupleTheEvaluated CoreEvaluatorBase::CreateTupleTheEvaluatedInvalid(

) const {
  int num_degrees_of_freedom = this->core_pdbqt_ligand_.GetNumDegreesOfFreedom();

  return std::make_tuple(

      99999999.9, std::vector<double>(num_degrees_of_freedom, 0.0), false

  );
}

TupleTheEvaluated2Debug CoreEvaluatorBase::CreateTupleTheEvaluatedInvalid2Debug(

) const {
  auto&& r = this->CreateTupleTheEvaluatedInvalid();
  return std::make_tuple(

      std::get<0>(r), std::get<1>(r), std::get<2>(r),

      std::unordered_map<std::string,double>()

  );
}

bool CoreEvaluatorBase::IsEveryValueZero(const double& a_value) {
  if (a_value != 0.0) {
    return false;
  }
  return true;
}

bool CoreEvaluatorBase::IsEveryValueZero(const std::vector<double>& array_1d) {
  for (const auto& i_value : array_1d) {
    if (i_value != 0.0) {
      return false;
    }
  }
  return true;
}

double CoreEvaluatorBase::calculate_distance_between_2_positions(

    std::array<double, 3> position_a, std::array<double, 3> position_b

) const {
  // double distancesquare_ab = 0.0;

  // for (int i = 0; i < 3; i++) {
  //   double i_delta = (position_a[i] - position_b[i]);
  //   distancesquare_ab += (i_delta * i_delta);
  // }

  // return std::sqrt(distancesquare_ab);

  return std::sqrt(

      this->calculate_distancesquare_between_2_positions(position_a, position_b)

  );
}

double CoreEvaluatorBase::calculate_distancesquare_between_2_positions(

    std::array<double, 3> position_a, std::array<double, 3> position_b

) const {
  double distancesquare_ab = 0.0;

  for (int i = 0; i < 3; i++) {
    double i_delta = (position_a[i] - position_b[i]);
    distancesquare_ab += (i_delta * i_delta);
  }

  return distancesquare_ab;
}

}  // namespace evaluators
}  // namespace core
}  // namespace pyvina
