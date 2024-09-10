// Copyright 2022 Tencent Inc. All rights reserved.
//
// Author: ruiyuanqian@tencent.com (ruiyuanqian)
//
// 文件注释
// ...

#ifndef PYVINA_CORE_EVALUATORS_CORE_EVALUATOR_BASE_H_
#define PYVINA_CORE_EVALUATORS_CORE_EVALUATOR_BASE_H_

#include <array>
#include <iostream>
#include <unordered_map>
#include <tuple>
#include <vector>

#include "pyvina/core/ligand.hpp"

namespace pyvina {
namespace core {
namespace evaluators {

using TupleTheEvaluated = std::tuple<double, std::vector<double>, bool>;

using TupleTheEvaluated2Debug = std::tuple<

    double, std::vector<double>, bool,

    std::unordered_map<std::string, double>

    >;

using Option4Evaluator = std::unordered_map<std::string, std::string>;

class CoreEvaluatorBase {
 public:
  const ligand& core_pdbqt_ligand_;

  CoreEvaluatorBase(

      const ligand& core_pdbqt_ligand

  );

  std::string get_str_address() const;
  virtual std::string get_evaluatorname() const = 0;

  virtual TupleTheEvaluated evaluate(

      const std::vector<double>& pose

  ) const = 0;

  virtual TupleTheEvaluated2Debug evaluate_2debug(

      const std::vector<double>& pose

  ) const = 0;

  Option4Evaluator option_ = {};
  bool is_option_set_ = false;

  void SetOption(const Option4Evaluator& option_new);
  void ResetOption();
  Option4Evaluator GetOption() const;
  std::string FindInOption(

      const Option4Evaluator& a_map, const std::string& a_key

  ) const;

  virtual TupleTheEvaluated2Debug evaluate_given_option(

      const std::vector<double>& pose,

      const std::unordered_map<std::string, std::string>& option

  ) const = 0;

  TupleTheEvaluated CreateTupleTheEvaluatedInvalid() const;
  TupleTheEvaluated2Debug CreateTupleTheEvaluatedInvalid2Debug() const;

  static bool IsEveryValueZero(const double&);
  static bool IsEveryValueZero(const std::vector<double>&);

  double calculate_distance_between_2_positions(

      std::array<double, 3> position_a, std::array<double, 3> position_b

  ) const;

  double calculate_distancesquare_between_2_positions(

      std::array<double, 3> position_a, std::array<double, 3> position_b

  ) const;
};

}  // namespace evaluators
}  // namespace core
}  // namespace pyvina

#endif
