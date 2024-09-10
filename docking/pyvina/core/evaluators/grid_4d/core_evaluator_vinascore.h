// Copyright 2022 Tencent Inc. All rights reserved.
//
// Author: ruiyuanqian@tencent.com (ruiyuanqian)
//
// 文件注释
// ...

#ifndef PYVINA_CORE_EVALUATORS_GRID_4D_CORE_EVALUATOR_VINASCORE_H_
#define PYVINA_CORE_EVALUATORS_GRID_4D_CORE_EVALUATOR_VINASCORE_H_

#include <iostream>
#include <tuple>
#include <vector>

#include "pyvina/core/evaluators/grid_4d/core_evaluator_grid_4d.h"
#include "pyvina/core/receptor.hpp"

namespace pyvina {
namespace core {
namespace evaluators {
namespace grid_4d {

class CoreEvaluatorVinascore : public CoreEvaluatorGrid4D {
 public:
  const receptor& core_pdbqt_receptor_;

  CoreEvaluatorVinascore(

      // ligand
      const ligand& core_pdbqt_ligand,
      // receptor
      const receptor& core_pdbqt_receptor,
      // pocket
      std::array<double, 3> corner_min, std::array<double, 3> corner_max,
      // resolution
      int reciprocal_resolution,
      // vdwsum[u_index_ligand][v_index_receptor] => u_vdwradius + v_vdwradius
      std::vector<std::vector<double> > vdwradius_sum_inter,
      // e.g. receptor atom positions
      std::vector<std::array<double, 3> > positions_inter

  );

  int get_celltype_by_index_heavy_atom(

      int index_heavy_atom

  ) const;

  double calculate_no_grid_loss_inter_for_1_heavy_atom(

      int index_heavy_atom, const std::array<double, 3>& coordinates_heavy_atom

  ) const;

  std::string get_evaluatorname() const;
};

}  // namespace grid_4d
}  // namespace evaluators
}  // namespace core
}  // namespace pyvina

#endif
