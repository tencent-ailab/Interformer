// Copyright 2022 Tencent Inc. All rights reserved.
//
// Author: ruiyuanqian@tencent.com (ruiyuanqian)
//
// 文件注释
// ...

#include "pyvina/core/evaluators/grid_4d/core_evaluator_distancemap.h"

namespace pyvina {
namespace core {
namespace evaluators {
namespace grid_4d {

CoreEvaluatorDistancemap::CoreEvaluatorDistancemap(

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

    )
    : CoreEvaluatorGrid4D{core_pdbqt_ligand,

                          corner_min, corner_max,
                          // grid.shape[0]
                          core_pdbqt_ligand.num_heavy_atoms, reciprocal_resolution,

                          vdwradius_sum_inter, positions_inter},
      distancemap_inter_{distancemap_inter} {
  ;
}

int CoreEvaluatorDistancemap::get_celltype_by_index_heavy_atom(

    int index_heavy_atom

) const {
  /////
  return index_heavy_atom;
}

double CoreEvaluatorDistancemap::calculate_no_grid_loss_inter_for_1_heavy_atom(

    int index_heavy_atom, const std::array<double, 3>& coordinates_heavy_atom

) const {
  return 0.0;
}

std::string CoreEvaluatorDistancemap::get_evaluatorname(

) const {
  return "CORE_EVALUATOR_DISTANCEMAP";
}

}  // namespace grid_4d
}  // namespace evaluators
}  // namespace core
}  // namespace pyvina
