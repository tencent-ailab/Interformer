// Copyright 2022 Tencent Inc. All rights reserved.
//
// Author: ruiyuanqian@tencent.com (ruiyuanqian)
//
// 文件注释
// ...

#ifndef PYVINA_CORE_EVALUATOR_CORE_EVALUATOR_DISTANCEMAP_GRID_H_
#define PYVINA_CORE_EVALUATOR_CORE_EVALUATOR_DISTANCEMAP_GRID_H_

#include <array>
#include <tuple>
#include <vector>

#include "pyvina/core/ligand.hpp"
#include "pyvina/core/evaluators/grid_4d/core_grid_4d.h"

namespace pyvina {
namespace core_evaluators {

// class CoreGrid4D {
//  public:
//   std::vector<std::vector<std::vector<std::vector<double> > > > value_;

//   std::array<double, 3> corner_min_;
//   std::array<double, 3> corner_max_;

//   std::array<double, 3> double_shape3d_;
//   std::array<int, 3> int_shape3d_;
//   std::array<int, 3> int_shape3d_with_margin_;

//   int num_celltypes_;

//   int reciprocal_resolution_;
//   double resolution_;

//   CoreGrid4D(

//       // pocket
//       std::array<double, 3> corner_min, std::array<double, 3> corner_max,

//       // .shape[0]
//       int num_celltypes,

//       // resolution
//       int reciprocal_resolution

//   );

//   std::array<double, 3> get_cellcoordinates_by_cellindices(std::array<int, 3> cellindices);

//   std::array<int, 3> get_cellindices_by_cellcoordinates(std::array<double, 3> cellcoordinates);

//   bool contains_coordinates(std::array<double, 3> coordinates);
// };

class CoreEvaluatorDistancemapGrid {
 public:
  core::evaluators::grid_4d::CoreGrid4D core_grid_4d_;

  ligand core_ligand_;

  std::vector<std::vector<double> > inter_distancemap_;
  std::vector<std::array<double, 3> > list_inter_coordinates_;
  std::vector<std::vector<double> > inter_weightmap_;

  CoreEvaluatorDistancemapGrid(

      // ligand
      ligand core_ligand,

      // inter-molecular distance map
      std::vector<std::vector<double> > inter_distancemap, std::vector<std::array<double, 3> > list_inter_coordinates,
      std::vector<std::vector<double> > inter_weightmap,

      // pocket
      std::array<double, 3> corner_min, std::array<double, 3> corner_max,

      // resolution
      int reciprocal_resolution

  );

  double calc_distance_between_2_coordinates(

      const std::array<double, 3> &a_coordinates, const std::array<double, 3> &b_coordinates

  );

  double get_loss_inter_for_1_heavy_atom(

      int index_heavy_atom, const std::array<double, 3> &coordinates_heavy_atom

  );

  void precalculate_losses_inter_for_1_heavy_atom(int index_heavy_atom);

  std::tuple<double, std::vector<double>, bool> evaluate_grid_based(

      // avoid naming collision: cnfr => conformation
      const std::vector<double> &cnfr

  );

  std::tuple<double, std::vector<double>, bool> evaluate_without_grid(

      const std::vector<double> &cnfr

  );
};

}  // namespace core_evaluators
}  // namespace pyvina

#endif
