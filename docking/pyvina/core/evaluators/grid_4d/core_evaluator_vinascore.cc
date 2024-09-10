// Copyright 2022 Tencent Inc. All rights reserved.
//
// Author: ruiyuanqian@tencent.com (ruiyuanqian)
//
// 文件注释
// ...

#include "pyvina/core/evaluators/grid_4d/core_evaluator_vinascore.h"

#include <cmath>

#include "pyvina/core/constants/common.h"
#include "pyvina/core/scoring_function.hpp"

namespace pyvina {
namespace core {
namespace evaluators {
namespace grid_4d {

CoreEvaluatorVinascore::CoreEvaluatorVinascore(

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

    )
    : CoreEvaluatorGrid4D(core_pdbqt_ligand,

                          corner_min, corner_max,

                          core_pdbqt_ligand.num_heavy_atoms, reciprocal_resolution,

                          vdwradius_sum_inter, positions_inter),
      core_pdbqt_receptor_(core_pdbqt_receptor) {
  ;
}

int CoreEvaluatorVinascore::get_celltype_by_index_heavy_atom(

    int index_heavy_atom

) const {
  return index_heavy_atom;
}

double CoreEvaluatorVinascore::calculate_no_grid_loss_inter_for_1_heavy_atom(

    int index_heavy_atom, const std::array<double, 3>& coordinates_heavy_atom

) const {
  const auto& a_xs = this->core_pdbqt_ligand_.heavy_atoms[index_heavy_atom].xs;

  double a_loss = 0.0;

  for (const auto& i_heavy_atom : this->core_pdbqt_receptor_.atoms) {
    const auto& b_xs = i_heavy_atom.xs;

    auto&& i_distance = this->calculate_distance_between_2_positions(

        coordinates_heavy_atom, i_heavy_atom.coord

    );

    if (i_distance >= scoring_function::cutoff) {
      continue;
    }

    a_loss += scoring_function::score_pybind11(a_xs, b_xs, i_distance);
  }

  return a_loss;
}

std::string CoreEvaluatorVinascore::get_evaluatorname(

) const {
  return "CORE_EVALUATOR_VINASCORE";
}

}  // namespace grid_4d
}  // namespace evaluators
}  // namespace core
}  // namespace pyvina
