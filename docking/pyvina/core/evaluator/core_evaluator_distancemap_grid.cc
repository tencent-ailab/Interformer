// Copyright 2022 Tencent Inc. All rights reserved.
//
// Author: ruiyuanqian@tencent.com (ruiyuanqian)
//
// 文件注释
// ...

#include "pyvina/core/evaluator/core_evaluator_distancemap_grid.h"

#include <math.h>
#include <omp.h>
#include <stdexcept>

namespace pyvina {
namespace core_evaluators {

// CoreGrid4D::CoreGrid4D(

//     // pocket
//     std::array<double, 3> corner_min, std::array<double, 3> corner_max,

//     // .shape[0]
//     int num_celltypes,

//     // resolution
//     int reciprocal_resolution

//     )
//     : corner_min_{corner_min},
//       num_celltypes_{num_celltypes},
//       reciprocal_resolution_{reciprocal_resolution}

// {
//   if (!(this->num_celltypes_ > 0 && this->num_celltypes_ < 100)) {
//     throw std::invalid_argument("!(this->num_celltypes_ > 0 && this->num_celltypes_ < 100)");
//   }

//   this->resolution_ = (1.0 / this->reciprocal_resolution_);

//   for (auto i = 0; i < 3; i++) {
//     if (!(corner_min[i] < corner_max[i])) {
//       throw std::invalid_argument("!(corner_min[i] < corner_max[i])");
//     }

//     double i_length_raw = corner_max[i] - corner_min[i];

//     this->int_shape3d_[i] = (int(i_length_raw / this->resolution_) + 2);
//     this->int_shape3d_with_margin_[i] = (this->int_shape3d_[i] + 2);

//     this->double_shape3d_[i] = (this->int_shape3d_[i] * this->resolution_);

//     this->corner_max_[i] = this->corner_min_[i] + this->double_shape3d_[i];
//   }

//   this->value_ = std::vector<std::vector<std::vector<std::vector<double> > > >(
//       this->num_celltypes_,
//       std::vector<std::vector<std::vector<double> > >(
//           this->int_shape3d_with_margin_[0],
//           std::vector<std::vector<double> >(this->int_shape3d_with_margin_[1],
//                                             std::vector<double>(this->int_shape3d_with_margin_[2], 0.0))));
// }

// // double CoreGrid4D::get_coordinate_by_index_and_axis(int index, int axis) {
// //   if (!(axis >= 0 && axis < 3)) {
// //     throw std::invalid_argument("!(axis >= 0 && axis < 3)");
// //   }

// //   if (!(index >= 0 && index < this->int_shape3d_with_margin_[axis])) {
// //     throw std::invalid_argument("!(index > 0 && index < this->int_shape3d_with_margin_[axis])")
// //   }

// //   return (this->corner_min_[axis]
// //           //
// //           - (0.5 * this->resolution_)
// //           //
// //           + (index * this->resolution_));
// // }

// std::array<double, 3> CoreGrid4D::get_cellcoordinates_by_cellindices(std::array<int, 3> cellindices) {
//   std::array<double, 3> cellcoordinates;

//   int i;
//   for (i = 0; i < 3; i++) {
//     if (!(cellindices[i] >= 0 && cellindices[i] < this->int_shape3d_with_margin_[i])) {
//       throw std::invalid_argument("!(cellindices[i] >= 0 && cellindices[i] < this->int_shape3d_with_margin_[i])");
//     }

//     cellcoordinates[i] = (this->corner_min_[i]
//                           //
//                           - (0.5 * this->resolution_)
//                           //
//                           + (cellindices[i] * this->resolution_));
//   }

//   return cellcoordinates;
// }

// std::array<int, 3> CoreGrid4D::get_cellindices_by_cellcoordinates(std::array<double, 3> cellcoordinates) {
//   std::array<int, 3> cellindices;

//   int i;
//   for (i = 0; i < 3; i++) {
//     double i_delta = (cellcoordinates[i] - this->corner_min_[i]);

//     if (!(i_delta > 0 && i_delta < this->double_shape3d_[i])) {
//       throw std::invalid_argument("!(i_delta > 0 && i_delta < this->double_shape3d_[i])");
//     }

//     cellindices[i] = 1 + int(i_delta / this->resolution_);
//   }

//   return cellindices;
// }

// bool CoreGrid4D::contains_coordinates(std::array<double, 3> coordinates) {
//   for (int i = 0; i < 3; i++) {
//     if (!(coordinates[i] > this->corner_min_[i]
//           //
//           && coordinates[i] < this->corner_max_[i])) {
//       return false;
//     }
//   }

//   return true;
// }

// ///////////////////////////////////////////////////
// /////
// ///////////////////////////////////////////////////

double CoreEvaluatorDistancemapGrid::calc_distance_between_2_coordinates(

    const std::array<double, 3> &a_coordinates, const std::array<double, 3> &b_coordinates

) {
  double ab_distance_square = 0.0;

  for (int i = 0; i < 3; i++) {
    double i_delta = a_coordinates[i] - b_coordinates[i];
    ab_distance_square += (i_delta * i_delta);
  }

  double ab_distance = sqrt(ab_distance_square);

  if (ab_distance > 10.0) {
    ab_distance = 10.0;
  }

  return ab_distance;
}

double CoreEvaluatorDistancemapGrid::get_loss_inter_for_1_heavy_atom(

    int index_heavy_atom, const std::array<double, 3> &coordinates_heavy_atom

) {
  double loss_inter = 0.0;

  int i, j;
  for (i = 0; i < this->list_inter_coordinates_.size(); i++) {
    // distance
    // double i_distance_square = 0.0;
    // for (j = 0; j < 3; j++) {
    //   double j_delta = list_inter_coordinates_[i][j] - coordinates_heavy_atom[j];
    //   i_distance_square = i_distance_square + (j_delta * j_delta);
    // }
    // double i_distance = sqrt(i_distance_square);
    // if (i_distance > 10.0) {
    //   i_distance = 10.0;
    // }
    double i_distance =
        this->calc_distance_between_2_coordinates(coordinates_heavy_atom, this->list_inter_coordinates_[i]);

    // loss
    double i_weight = this->inter_weightmap_[index_heavy_atom][i];

    double i_delta = (this->inter_distancemap_[index_heavy_atom][i] - i_distance);
    double i_loss = i_delta * i_delta * i_weight;

    loss_inter += i_loss;
  }

  return loss_inter;
}

void CoreEvaluatorDistancemapGrid::precalculate_losses_inter_for_1_heavy_atom(int index_heavy_atom) {
  int i, j, k;

  const int i_max = this->core_grid_4d_.int_shape3d_with_margin_[0];
  const int j_max = this->core_grid_4d_.int_shape3d_with_margin_[1];
  const int k_max = this->core_grid_4d_.int_shape3d_with_margin_[2];

  for (i = 0; i < i_max; i++) {
    for (j = 0; j < j_max; j++) {
      for (k = 0; k < k_max; k++) {
        std::array<double, 3> coordinates_ijk =
            this->core_grid_4d_.get_cellcoordinates_by_cellindices(std::array<int, 3>{i, j, k});

        this->core_grid_4d_.value_[index_heavy_atom][i][j][k] =
            this->get_loss_inter_for_1_heavy_atom(index_heavy_atom, coordinates_ijk);
      }
    }
  }

  //
}

std::tuple<double, std::vector<double>, bool> CoreEvaluatorDistancemapGrid::evaluate_grid_based(

    // avoid naming collision: cnfr => conformation
    const std::vector<double> &cnfr

) {
  const int num_degrees_of_freedom = cnfr.size();
  auto dloss_over_dconformation = std::vector<double>(num_degrees_of_freedom, 0.0);

  auto &&regarding_ligand_coordinates = this->core_ligand_.FormulaDerivativeCore_ConformationToCoords(cnfr);
  auto &&list_heavy_atom_coordinates = std::get<0>(regarding_ligand_coordinates);
  auto &&dx_dy_dz_over_dconformation = std::get<2>(regarding_ligand_coordinates);

  /////////////////////////////
  ///// inter loss and related gradients
  /////////////////////////////
  double loss_inter = 0.0;
  for (int i = 0; i < this->core_ligand_.num_heavy_atoms; i++) {
    auto &&i_xyz = list_heavy_atom_coordinates[i];

    if (!(this->core_grid_4d_.contains_coordinates(i_xyz))) {
      return std::make_tuple(
          //
          99999999.9, std::vector<double>(num_degrees_of_freedom, 0.0), false);
    }

    auto &&i_indices = this->core_grid_4d_.get_cellindices_by_cellcoordinates(i_xyz);
    // printf(" [%d] heavy_atom ( %f , %f , %f )\n", i, i_xyz[0], i_xyz[1], i_xyz[2]);
    // printf(" grid_indices: %d %d %d\n", i_indices[0], i_indices[1], i_indices[2]);

    double i_loss = this->core_grid_4d_.value_[i][i_indices[0]][i_indices[1]][i_indices[2]];
    loss_inter += i_loss;

    double resolution = this->core_grid_4d_.resolution_;
    for (int j = 0; j < 3; j++) {
      auto j_indices = i_indices;
      j_indices[j] += 1;

      double j_loss = this->core_grid_4d_.value_[i][j_indices[0]][j_indices[1]][j_indices[2]];
      double dloss_over_dj = (j_loss - i_loss) / resolution;

      for (int k = 0; k < num_degrees_of_freedom; k++) {
        dloss_over_dconformation[k] += dloss_over_dj * dx_dy_dz_over_dconformation[i][j][k];
      }
    }
  }

  auto loss_total = loss_inter;
  return std::make_tuple(
      //
      loss_total, dloss_over_dconformation, true);
}

CoreEvaluatorDistancemapGrid::CoreEvaluatorDistancemapGrid(

    // ligand
    ligand core_ligand,

    // inter-molecular distance map
    std::vector<std::vector<double> > inter_distancemap, std::vector<std::array<double, 3> > list_inter_coordinates,
    std::vector<std::vector<double> > inter_weightmap,

    // pocket
    std::array<double, 3> corner_min, std::array<double, 3> corner_max,

    // resolution
    int reciprocal_resolution

    )
    : core_ligand_{core_ligand},
      inter_distancemap_{inter_distancemap},
      list_inter_coordinates_{list_inter_coordinates},
      inter_weightmap_{inter_weightmap},
      core_grid_4d_{corner_min, corner_max, core_ligand.num_heavy_atoms, reciprocal_resolution}

{
  if (!(this->core_ligand_.pdbqt_file_path_.size() > 0)) {
    throw std::invalid_argument("!(this->core_ligand_.pdbqt_file_path_.size() > 0)");
  }

#pragma omp parallel for
  for (int i = 0; i < this->core_ligand_.num_heavy_atoms; i++) {
    printf(" ::  start precalculating losses for heavy atom [ %d ] :: \n", i);
    this->precalculate_losses_inter_for_1_heavy_atom(i);
    printf(" :: finish precalculating losses for heavy atom [ %d ] :: \n", i);
  }

  //
}

}  // namespace core_evaluators
}  // namespace pyvina
