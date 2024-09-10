// Copyright 2022 Tencent Inc. All rights reserved.
//
// Author: ruiyuanqian@tencent.com (ruiyuanqian)
//
// 文件注释
// ...

#include "pyvina/core/evaluators/grid_4d/core_evaluator_normalscore.h"

#include <cmath>

#include "pyvina/core/constants/common.h"

namespace pyvina {
namespace core {
namespace evaluators {
namespace grid_4d {

CoreEvaluatorNormalscore::CoreEvaluatorNormalscore(

    // ligand
    const ligand& core_pdbqt_ligand,
    // normal distribution related
    std::vector<std::vector<std::vector<double>>> pi_inter, std::vector<std::vector<std::vector<double>>> mean_inter,
    std::vector<std::vector<std::vector<double>>> sigma_inter, std::vector<std::vector<double>> vdwradius_sum_inter,
    // e.g. receptor atom positions
    std::vector<std::array<double, 3>> positions_inter,
    // pocket
    std::array<double, 3> corner_min, std::array<double, 3> corner_max,
    // resolution
    int reciprocal_resolution

    )
    : CoreEvaluatorGrid4D{core_pdbqt_ligand, corner_min, corner_max,
                          // grid.shape[0]
                          core_pdbqt_ligand.num_heavy_atoms, reciprocal_resolution,
                          //
                          vdwradius_sum_inter, positions_inter},
      pi_inter_{pi_inter},
      mean_inter_{mean_inter},
      sigma_inter_{sigma_inter},
      num_normals{mean_inter[0][0].size()}

{
  ;
}

int CoreEvaluatorNormalscore::get_celltype_by_index_heavy_atom(

    int index_heavy_atom

) const {
  /////
  return index_heavy_atom;
}

double CoreEvaluatorNormalscore::calculate_no_grid_normalscore_inter_for_1_pair(

    int index_heavy_atom, int index_inter, double distance

) const {
  using pyvina::core::constants::kSqrt2Pi;

  double vdwdistance = (
      /////
      distance - this->vdwradius_sum_inter_[index_heavy_atom][index_inter]);

  if (vdwdistance >= this->cutoff_vdwdistance_) {
    return 0.0;
  }

  double normalscore_inter_no_grid = 0.0;

  for (int i = 0; i < this->num_normals; i++) {
    const auto& i_pi = this->pi_inter_[index_heavy_atom][index_inter][i];
    const auto& i_mean = this->mean_inter_[index_heavy_atom][index_inter][i];
    const auto& i_sigma = this->sigma_inter_[index_heavy_atom][index_inter][i];

    const auto& i_temp = (vdwdistance - i_mean) / i_sigma;

    const auto& i_normalscore = (

        (-1.0)

        *

        (i_pi)

        *

        (1.0 / (i_sigma * kSqrt2Pi))

        *

        (std::exp(-0.5 * i_temp * i_temp))

    );

    normalscore_inter_no_grid += i_normalscore;
  }

  if (vdwdistance < 0.0) {
    double score_collision_inter = (

        this->weight_collision_inter_

        *

        (vdwdistance * vdwdistance)

    );
    normalscore_inter_no_grid += score_collision_inter;
  }

  return normalscore_inter_no_grid;
}

double CoreEvaluatorNormalscore::calculate_no_grid_loss_inter_for_1_heavy_atom(

    int index_heavy_atom, const std::array<double, 3>& coordinates_heavy_atom

) const {
  using pyvina::core::constants::kPi;

  double loss_inter_no_grid = 0.0;

  for (int i = 0; i < this->positions_inter_.size(); i++) {
    double i_distance = this->calculate_distance_between_2_positions(

        coordinates_heavy_atom, this->positions_inter_[i]);

    loss_inter_no_grid += this->calculate_no_grid_normalscore_inter_for_1_pair(

        index_heavy_atom, i, i_distance);
  }

  // int i, j;

  // int num_normals = this->pi_inter_[0][0].size();

  // for (i = 0; i < this->positions_inter_.size(); i++) {
  //   double i_distance = this->calculate_distance_between_2_positions(coordinates_heavy_atom,
  //   this->positions_inter_[i]);

  //   double i_vdwdistance = i_distance - this->vdwradius_sum_inter_[index_heavy_atom][i];

  //   if (i_vdwdistance >= 4.0) {
  //     continue;
  //   }

  //   for (j = 0; j < num_normals; j++) {
  //     double ij_pi = this->pi_inter_[index_heavy_atom][i][j];
  //     double ij_mean = this->mean_inter_[index_heavy_atom][i][j];
  //     double ij_sigma = this->sigma_inter_[index_heavy_atom][i][j];

  //     double ij_temp = (i_vdwdistance - ij_mean) / ij_sigma;

  //     double ij_normal = (
  //         /////
  //         (1.0 / (ij_sigma * std::sqrt(2 * kPi)))
  //         /////
  //         * (std::exp(-0.5 * ij_temp * ij_temp)));

  //     loss_inter_no_grid += -(ij_pi * ij_normal);
  //   }
  // }

  return loss_inter_no_grid;
}

void CoreEvaluatorNormalscore::precalculate_losses_inter_in_grid_for_1_heavy_atom(

    int index_heavy_atom, int options_bitwise

) {
  if (!(options_bitwise & 1)) {
    CoreEvaluatorGrid4D::precalculate_losses_inter_in_grid_for_1_heavy_atom(

        index_heavy_atom, options_bitwise

    );
    return;
  }

//  printf(
//      " :: [ options_bitwise & 1 ]"
//      " :: TRUE"
//      " :: use custom precalculating in [ CoreEvaluatorNormalscore ] :: \n");

  const auto& reciprocal_resolution = (

      2 * this->core_grid_4d_.reciprocal_resolution_

  );
  const auto& resolution = 1.0 / reciprocal_resolution;

  const int margin_for_normalscores = 10;
  const int size_normalscores_by_distancesquare = int(

      (this->cutoff_distance_) * (this->cutoff_distance_) * reciprocal_resolution

      + margin_for_normalscores

  );

  std::vector<std::vector<double>> normalscores_by_distancesquare(

      this->positions_inter_.size(),

      std::vector<double>(size_normalscores_by_distancesquare, 0.0)

  );

  for (int i = 0; i < this->positions_inter_.size(); i++) {
    for (int j = 0; j < size_normalscores_by_distancesquare; j++) {
      double j_distance = std::sqrt(

          (j + 0.5) * resolution

      );

      normalscores_by_distancesquare[i][j] =
          this->calculate_no_grid_normalscore_inter_for_1_pair(index_heavy_atom, i, j_distance);
    }
  }

  for (int l = 0; l < this->positions_inter_.size(); l++) {
    const double l_cutoff_distance = (this->cutoff_vdwdistance_

                                      + this->vdwradius_sum_inter_[index_heavy_atom][l]);

    ///// calculate range(min/max) for i
    std::array<double, 3> position_so_far(this->positions_inter_[l]);
    const auto&& i_minmax =
        this->core_grid_4d_.get_minmax_cellindex(0, position_so_far, this->positions_inter_[l], l_cutoff_distance);

    for (int i = i_minmax[0]; i <= i_minmax[1]; i++) {
      ///// calculate range(min/max) for j
      position_so_far[0] = this->core_grid_4d_.get_cellcoordinate_by_cellindex(i, 0, true);
      const auto&& j_minmax =
          this->core_grid_4d_.get_minmax_cellindex(1, position_so_far, this->positions_inter_[l], l_cutoff_distance);

      for (int j = j_minmax[0]; j <= j_minmax[1]; j++) {
        ///// calculate range(min/max) for k
        position_so_far[1] = this->core_grid_4d_.get_cellcoordinate_by_cellindex(j, 1, true);
        const auto&& k_minmax =
            this->core_grid_4d_.get_minmax_cellindex(2, position_so_far, this->positions_inter_[l], l_cutoff_distance);

        for (int k = k_minmax[0]; k <= k_minmax[1]; k++) {
          const auto&& position_ijk = this->core_grid_4d_.get_cellcoordinates_by_cellindices(

              std::array<int, 3>{i, j, k});

          const auto&& distancesquare_ijkl = this->calculate_distancesquare_between_2_positions(

              position_ijk, this->positions_inter_[l]

          );

          const int index_distancesquare_ijkl = int(

              distancesquare_ijkl * reciprocal_resolution

          );

          if (index_distancesquare_ijkl >= size_normalscores_by_distancesquare) {
            continue;
          }

          this->core_grid_4d_.value_[index_heavy_atom][i][j][k] +=
              normalscores_by_distancesquare[l][index_distancesquare_ijkl];
        }
      }
    }
  }

  // for (int i = 0; i < this->core_grid_4d_.int_shape3d_with_margin_[0]; i++) {
  //   for (int j = 0; j < this->core_grid_4d_.int_shape3d_with_margin_[1]; j++) {
  //     for (int k = 0; k < this->core_grid_4d_.int_shape3d_with_margin_[2]; k++) {
  //       const auto&& position_ijk = this->core_grid_4d_.get_cellcoordinates_by_cellindices(

  //           std::array<int, 3>{i, j, k});

  //       double normalscore_ijk = 0.0;
  //       for (int l = 0; l < this->positions_inter_.size(); l++) {
  //         const auto&& distancesquare_ijkl = this->calculate_distancesquare_between_2_positions(

  //             position_ijk, this->positions_inter_[l]

  //         );

  //         const int index_distancesquare_ijkl = int(

  //             distancesquare_ijkl * reciprocal_resolution

  //         );

  //         if (index_distancesquare_ijkl >= size_normalscores_by_distancesquare) {
  //           continue;
  //         }

  //         normalscore_ijk += normalscores_by_distancesquare[l][index_distancesquare_ijkl];
  //       }

  //       this->core_grid_4d_.value_[index_heavy_atom][i][j][k] = normalscore_ijk;
  //     }
  //   }
  // }

  return;
}

std::string CoreEvaluatorNormalscore::get_evaluatorname(

) const {
  return "CORE_EVALUATOR_NORMALSCORE";
}

}  // namespace grid_4d
}  // namespace evaluators
}  // namespace core
}  // namespace pyvina
