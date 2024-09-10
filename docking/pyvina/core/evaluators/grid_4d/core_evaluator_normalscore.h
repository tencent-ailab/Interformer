// Copyright 2022 Tencent Inc. All rights reserved.
//
// Author: ruiyuanqian@tencent.com (ruiyuanqian)
//
// 文件注释
// ...

#ifndef PYVINA_CORE_EVALUATORS_GRID_4D_CORE_EVALUATOR_NORMALSCORE_H_
#define PYVINA_CORE_EVALUATORS_GRID_4D_CORE_EVALUATOR_NORMALSCORE_H_

#include <iostream>
#include <tuple>
#include <vector>

#include "pyvina/core/evaluators/grid_4d/core_evaluator_grid_4d.h"

namespace pyvina {
namespace core {
namespace evaluators {
namespace grid_4d {

class CoreEvaluatorNormalscore : public CoreEvaluatorGrid4D {

 public:
  int num_normals;

  std::vector<std::vector<std::vector<double> > > pi_inter_;
  std::vector<std::vector<std::vector<double> > > mean_inter_;
  std::vector<std::vector<std::vector<double> > > sigma_inter_;

  CoreEvaluatorNormalscore(

      // ligand
      const ligand& core_pdbqt_ligand,
      // normal distribution related
      std::vector<std::vector<std::vector<double> > > pi_inter,
      std::vector<std::vector<std::vector<double> > > mean_inter,
      std::vector<std::vector<std::vector<double> > > sigma_inter,
      std::vector<std::vector<double> > vdwradius_sum_inter,
      // e.g. receptor atom positions
      std::vector<std::array<double, 3> > positions_inter,
      // pocket
      std::array<double, 3> corner_min, std::array<double, 3> corner_max,
      // resolution
      int reciprocal_resolution

  );

  int get_celltype_by_index_heavy_atom(

      int index_heavy_atom

  ) const;

  double calculate_no_grid_normalscore_inter_for_1_pair(

      int index_heavy_atom, int index_inter, double distance

  ) const;

  double calculate_no_grid_loss_inter_for_1_heavy_atom(

      int index_heavy_atom, const std::array<double, 3>& coordinates_heavy_atom

  ) const;

  void precalculate_losses_inter_in_grid_for_1_heavy_atom(

      int index_heavy_atom, int options_bitwise

  );

  std::string get_evaluatorname() const;
};

}  // namespace grid_4d
}  // namespace evaluators
}  // namespace core
}  // namespace pyvina

#endif
