// Copyright 2022 Tencent Inc. All rights reserved.
//
// Author: ruiyuanqian@tencent.com (ruiyuanqian)
//
// 文件注释
// ...

#ifndef PYVINA_CORE_EVALUATORS_GRID_4D_CORE_EVALUATOR_DISTANCEMAP_H_
#define PYVINA_CORE_EVALUATORS_GRID_4D_CORE_EVALUATOR_DISTANCEMAP_H_

#include <iostream>
#include <tuple>
#include <vector>

#include "pyvina/core/evaluators/grid_4d/core_evaluator_grid_4d.h"

namespace pyvina {
namespace core {
namespace evaluators {
namespace grid_4d {

class CoreEvaluatorDistancemap : public CoreEvaluatorGrid4D {
 public:
  std::vector<std::vector<double> > distancemap_inter_;

  CoreEvaluatorDistancemap(

      // ligand
      const ligand& core_pdbqt_ligand,
      // inter-molecular distance map
      std::vector<std::vector<double> > distancemap_inter,
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
