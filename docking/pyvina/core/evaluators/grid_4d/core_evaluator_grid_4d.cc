// Copyright 2022 Tencent Inc. All rights reserved.
//
// Author: ruiyuanqian@tencent.com (ruiyuanqian)
//
// 文件注释
// ...

#include "pyvina/core/evaluators/grid_4d/core_evaluator_grid_4d.h"

#include <algorithm>
// #include <math.h>
#include <omp.h>
#include <stdexcept>

#include "pyvina/core/wrappers_for_exceptions/common.h"

namespace pyvina {
namespace core {
namespace evaluators {
namespace grid_4d {

CoreEvaluatorGrid4D::CoreEvaluatorGrid4D(

    // ligand
    const ligand& core_pdbqt_ligand,
    // pocket
    std::array<double, 3> corner_min, std::array<double, 3> corner_max,
    // grid.shape[0]
    int num_celltypes,
    // resolution
    int reciprocal_resolution,
    // vdwsum[u_index_ligand][v_index_receptor] => u_vdwradius + v_vdwradius
    std::vector<std::vector<double> > vdwradius_sum_inter,
    // e.g. receptor atom positions
    std::vector<std::array<double, 3> > positions_inter

    )
    :

      CoreEvaluatorBase(core_pdbqt_ligand),
      core_grid_4d_{corner_min, corner_max, num_celltypes, reciprocal_resolution},

      vdwradius_sum_inter_{vdwradius_sum_inter},
      positions_inter_{positions_inter},

      cutoff_vdwdistance_{4.0}

{
  // should NOT a empty ligand
  if (!(this->core_pdbqt_ligand_.pdbqt_file_path_.size() > 0)) {
    pyvina::core::wrappers_for_exceptions::Throw1ExceptionAfterPrinting(
        "!(this->core_pdbqt_ligand_.pdbqt_file_path_.size() > 0)");
  }

  for (int i = 0; i < this->vdwradius_sum_inter_.size(); i++) {
    for (int j = 0; j < this->vdwradius_sum_inter_[i].size(); j++) {
      double ij_vdwradius_sum = this->vdwradius_sum_inter_[i][j];

      if (ij_vdwradius_sum > this->max_vdwradius_sum_) {
        this->max_vdwradius_sum_ = ij_vdwradius_sum;
      }
    }
  }

  this->cutoff_distance_ = this->cutoff_vdwdistance_ + this->max_vdwradius_sum_;

  //////////////////////////////////////////////
  ///// NEVER call a virtual func in constructor
  //////////////////////////////////////////////
  ///// https://stackoverflow.com/questions/962132/calling-virtual-functions-inside-constructors
  //////////////////////////////////////////////
  this->core_grid_3d_intra_.create_grid_map();
}

void CoreEvaluatorGrid4D::precalculate_losses_inter_in_grid_for_1_heavy_atom(

    int index_heavy_atom, int options_bitwise

) {
  printf(" :: use default precalculating in [ CoreEvaluatorGrid4D ] :: \n");

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
            this->calculate_no_grid_loss_inter_for_1_heavy_atom(index_heavy_atom, coordinates_ijk);
      }
    }
  }

  /////
}

void CoreEvaluatorGrid4D::precalculate_core_grid_4d_(

    int options_bitwise

) {
//  printf(
//      "\n"
//      " :: bitwise options for precalculating :: [ %d ] :: \n"
//      " :: [ options_bitwise & 1 ] => [ %d ] :: \n"
//      "\n",
//      options_bitwise, (options_bitwise & 1));

  if (this->is_core_grid_4d_ready_) {
    pyvina::core::wrappers_for_exceptions::Throw1ExceptionAfterPrinting(
        ".precalculate_core_grid_4d_() should run only once");
  }

  int i;

#pragma omp parallel for
  for (i = 0; i < this->core_grid_4d_.num_celltypes_; i++) {
    // std::cout << " ::  start precalculating losses from heavy atom [ " << i << " ] :: " << std::endl;
    // printf(" ::  start precalculating losses from heavy atom [ %d ] :: threadnum = [ %d / %d / %d ] :: \n",
    //           i, omp_get_thread_num(), omp_get_max_threads(), omp_get_thread_limit());

    this->precalculate_losses_inter_in_grid_for_1_heavy_atom(i, options_bitwise);

    // printf(" :: finish precalculating losses from heavy atom [ %d ] :: threadnum = [ %d / %d / %d ] :: \n",
    //           i, omp_get_thread_num(), omp_get_max_threads(), omp_get_thread_limit());
    // std::cout << " :: finish precalculating losses from heavy atom [ " << i << " ] :: " << std::endl;
  }

  this->is_core_grid_4d_ready_ = true;
}

TupleTheEvaluated CoreEvaluatorGrid4D::evaluate(

    const std::vector<double>& pose

) const {
  if (this->is_option_set_) {
    auto&& evaluated_with_option = this->evaluate_given_option(pose, this->option_);
    return std::make_tuple(

        std::get<0>(evaluated_with_option),

        std::get<1>(evaluated_with_option),

        std::get<2>(evaluated_with_option)

    );
  }

  using pyvina::core::wrappers_for_exceptions::AssertGivenWhat;

  if (!(this->is_core_grid_4d_ready_)) {
    pyvina::core::wrappers_for_exceptions::Throw1ExceptionAfterPrinting(
        ".evaluate() but grid has not been precalculated");
  }

  const int num_degrees_of_freedom = this->core_pdbqt_ligand_.GetNumDegreesOfFreedom();
  AssertGivenWhat(num_degrees_of_freedom == pose.size(),

                  "num_degrees_of_freedom == pose.size()");

  auto dloss_over_dpose = std::vector<double>(num_degrees_of_freedom, 0.0);

  auto&& regarding_ligand_positions = this->core_pdbqt_ligand_.FormulaDerivativeCore_ConformationToCoords(pose);
  auto&& positions_heavy_atoms = std::get<0>(regarding_ligand_positions);
  auto&& dx_dy_dz_over_dpose = std::get<2>(regarding_ligand_positions);

  /////////////////////////////
  ///// inter loss
  /////////////////////////////
  double loss_inter = 0.0;
  for (int i = 0; i < this->core_pdbqt_ligand_.num_heavy_atoms; i++) {
    int i_celltype = this->get_celltype_by_index_heavy_atom(i);

    auto&& i_xyz = positions_heavy_atoms[i];

    if (!(this->core_grid_4d_.contains_coordinates(i_xyz))) {
      return this->CreateTupleTheEvaluatedInvalid();
    }

    auto&& i_cellindices = this->core_grid_4d_.get_cellindices_by_cellcoordinates(i_xyz);

    double i_loss = this->core_grid_4d_.value_[
        /////
        i_celltype][i_cellindices[0]][i_cellindices[1]][i_cellindices[2]];
    loss_inter += i_loss;

    double resolution = this->core_grid_4d_.resolution_;
    for (int j = 0; j < 3; j++) {
      auto j_cellindices = i_cellindices;
      j_cellindices[j] += 1;

      double j_loss = this->core_grid_4d_.value_[
          /////
          i_celltype][j_cellindices[0]][j_cellindices[1]][j_cellindices[2]];

      double dloss_over_dj = (j_loss - i_loss) / resolution;

      for (int k = 0; k < num_degrees_of_freedom; k++) {
        dloss_over_dpose[k] += dloss_over_dj * dx_dy_dz_over_dpose[i][j][k];
      }
    }
  }

  /////////////////////////////
  ///// intra loss
  /////////////////////////////
  double loss_intra = 0.0;
  const double weight_intra = this->weight_intra_;

  for (const auto& i_pair : this->core_pdbqt_ligand_.interacting_pairs) {
    const auto& a_index = i_pair.i0;
    const auto& b_index = i_pair.i1;
    const auto& a_position = positions_heavy_atoms[a_index];
    const auto& b_position = positions_heavy_atoms[b_index];
    const auto& a_xs = this->core_pdbqt_ligand_.heavy_atoms[a_index].xs;
    const auto& b_xs = this->core_pdbqt_ligand_.heavy_atoms[b_index].xs;

    double i_distancesquare = this->calculate_distancesquare_between_2_positions(

        a_position, b_position

    );

    loss_intra += (

        weight_intra

        * this->core_grid_3d_intra_.score_by_grid_collision_only(a_xs, b_xs, i_distancesquare)

    );

    double dloss_over_ddistancesquare = this->core_grid_3d_intra_.get_derivative_by_grid_collision_only(

        a_xs, b_xs, i_distancesquare

    );

    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < num_degrees_of_freedom; k++) {
        dloss_over_dpose[k] += (

            weight_intra

            * dloss_over_ddistancesquare

            * 2 * (a_position[j] - b_position[j])

            * (dx_dy_dz_over_dpose[a_index][j][k] - dx_dy_dz_over_dpose[b_index][j][k])

        );
      }
    }
  }

  auto loss_total = loss_inter + loss_intra;

  if (this->IsEveryValueZero(dloss_over_dpose)) {
    if (this->IsEveryValueZero(loss_total)) {
      return this->CreateTupleTheEvaluatedInvalid();
    }

    pyvina::core::wrappers_for_exceptions::Throw1ExceptionAfterPrinting(

        " :: all values in gradient are zero :: "

        + pyvina::core::wrappers_for_exceptions::GetStrFromArray1D(dloss_over_dpose)

    );
  }

  return std::make_tuple(
      /////
      loss_total, dloss_over_dpose, true);
}

TupleTheEvaluated2Debug CoreEvaluatorGrid4D::evaluate_2debug(

    const std::vector<double>& pose

) const {
  /////////////////////////////
  ///// COPY START
  /////////////////////////////

  using pyvina::core::wrappers_for_exceptions::AssertGivenWhat;

  if (!(this->is_core_grid_4d_ready_)) {
    pyvina::core::wrappers_for_exceptions::Throw1ExceptionAfterPrinting(
        ".evaluate() but grid has not been precalculated");
  }

  const int num_degrees_of_freedom = this->core_pdbqt_ligand_.GetNumDegreesOfFreedom();
  AssertGivenWhat(num_degrees_of_freedom == pose.size(),

                  "num_degrees_of_freedom == pose.size()");

  auto dloss_over_dpose = std::vector<double>(num_degrees_of_freedom, 0.0);

  auto&& regarding_ligand_positions = this->core_pdbqt_ligand_.FormulaDerivativeCore_ConformationToCoords(pose);
  auto&& positions_heavy_atoms = std::get<0>(regarding_ligand_positions);
  auto&& dx_dy_dz_over_dpose = std::get<2>(regarding_ligand_positions);

  /////////////////////////////
  ///// inter loss
  /////////////////////////////
  double loss_inter = 0.0;
  for (int i = 0; i < this->core_pdbqt_ligand_.num_heavy_atoms; i++) {
    int i_celltype = this->get_celltype_by_index_heavy_atom(i);

    auto&& i_xyz = positions_heavy_atoms[i];

    if (!(this->core_grid_4d_.contains_coordinates(i_xyz))) {
      return this->CreateTupleTheEvaluatedInvalid2Debug();
    }

    auto&& i_cellindices = this->core_grid_4d_.get_cellindices_by_cellcoordinates(i_xyz);

    double i_loss = this->core_grid_4d_.value_[
        /////
        i_celltype][i_cellindices[0]][i_cellindices[1]][i_cellindices[2]];
    loss_inter += i_loss;

    double resolution = this->core_grid_4d_.resolution_;
    for (int j = 0; j < 3; j++) {
      auto j_cellindices = i_cellindices;
      j_cellindices[j] += 1;

      double j_loss = this->core_grid_4d_.value_[
          /////
          i_celltype][j_cellindices[0]][j_cellindices[1]][j_cellindices[2]];

      double dloss_over_dj = (j_loss - i_loss) / resolution;

      for (int k = 0; k < num_degrees_of_freedom; k++) {
        dloss_over_dpose[k] += dloss_over_dj * dx_dy_dz_over_dpose[i][j][k];
      }
    }
  }

  /////////////////////////////
  ///// intra loss
  /////////////////////////////
  double loss_intra = 0.0;
  const double weight_intra = this->weight_intra_;

  for (const auto& i_pair : this->core_pdbqt_ligand_.interacting_pairs) {
    const auto& a_index = i_pair.i0;
    const auto& b_index = i_pair.i1;
    const auto& a_position = positions_heavy_atoms[a_index];
    const auto& b_position = positions_heavy_atoms[b_index];
    const auto& a_xs = this->core_pdbqt_ligand_.heavy_atoms[a_index].xs;
    const auto& b_xs = this->core_pdbqt_ligand_.heavy_atoms[b_index].xs;

    double i_distancesquare = this->calculate_distancesquare_between_2_positions(

        a_position, b_position

    );

    loss_intra += (

        weight_intra

        * this->core_grid_3d_intra_.score_by_grid_collision_only(a_xs, b_xs, i_distancesquare)

    );

    double dloss_over_ddistancesquare = this->core_grid_3d_intra_.get_derivative_by_grid_collision_only(

        a_xs, b_xs, i_distancesquare

    );

    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < num_degrees_of_freedom; k++) {
        dloss_over_dpose[k] += (

            weight_intra

            * dloss_over_ddistancesquare

            * 2 * (a_position[j] - b_position[j])

            * (dx_dy_dz_over_dpose[a_index][j][k] - dx_dy_dz_over_dpose[b_index][j][k])

        );
      }
    }
  }

  auto loss_total = loss_inter + loss_intra;

  if (this->IsEveryValueZero(dloss_over_dpose)) {
    if (this->IsEveryValueZero(loss_total)) {
      return this->CreateTupleTheEvaluatedInvalid2Debug();
    }

    pyvina::core::wrappers_for_exceptions::Throw1ExceptionAfterPrinting(

        " :: all values in gradient are zero :: "

        + pyvina::core::wrappers_for_exceptions::GetStrFromArray1D(dloss_over_dpose)

    );
  }

  /////////////////////////////
  ///// COPY END
  /////////////////////////////

  /////////////////////////////
  ///// dict 4details
  /////////////////////////////
  std::unordered_map<std::string, double> dict_4details;
  dict_4details["loss_total"] = loss_total;
  dict_4details["loss_inter"] = loss_inter;
  dict_4details["loss_intra"] = loss_intra;

  return std::make_tuple(
      /////
      loss_total, dloss_over_dpose, true,

      dict_4details

  );
}

TupleTheEvaluated2Debug CoreEvaluatorGrid4D::evaluate_no_grid(

    const std::vector<double>& pose,

    bool is_detailed

) const {
  using pyvina::core::wrappers_for_exceptions::AssertGivenWhat;
  const int num_degrees_of_freedom = this->core_pdbqt_ligand_.GetNumDegreesOfFreedom();
  AssertGivenWhat(num_degrees_of_freedom == pose.size(),

                  "num_degrees_of_freedom == pose.size()");

  auto dloss_over_dpose = std::vector<double>(num_degrees_of_freedom, 0.0);

  auto&& regarding_ligand_positions = this->core_pdbqt_ligand_.FormulaDerivativeCore_ConformationToCoords(pose);
  auto&& positions_heavy_atoms = std::get<0>(regarding_ligand_positions);

  /////////////////////////////
  ///// inter loss
  ///// TODO: no_grid gradients
  /////////////////////////////
  double loss_inter = 0.0;
  for (int i = 0; i < this->core_pdbqt_ligand_.num_heavy_atoms; i++) {
    auto&& i_xyz = positions_heavy_atoms[i];

    if (!(this->core_grid_4d_.contains_coordinates(i_xyz))) {
      this->CreateTupleTheEvaluatedInvalid2Debug();
    }

    double i_loss_inter = this->calculate_no_grid_loss_inter_for_1_heavy_atom(i, i_xyz);

    loss_inter += i_loss_inter;
  }

  /////////////////////////////
  ///// intra loss
  ///// TODO: no_grid gradients
  /////////////////////////////
  double loss_intra = 0.0;
  const double weight_intra = this->weight_intra_;

  for (const auto& i_pair : this->core_pdbqt_ligand_.interacting_pairs) {
    const auto& a_index = i_pair.i0;
    const auto& b_index = i_pair.i1;
    const auto& a_position = positions_heavy_atoms[a_index];
    const auto& b_position = positions_heavy_atoms[b_index];
    const auto& a_xs = this->core_pdbqt_ligand_.heavy_atoms[a_index].xs;
    const auto& b_xs = this->core_pdbqt_ligand_.heavy_atoms[b_index].xs;

    double i_distancesquare = this->calculate_distancesquare_between_2_positions(

        a_position, b_position

    );

    loss_intra += (

        weight_intra

        * this->core_grid_3d_intra_.score_by_grid_collision_only(a_xs, b_xs, i_distancesquare)

    );
  }

  auto loss_total = loss_inter + loss_intra;

  /////////////////////////////
  ///// dict 4details
  /////////////////////////////
  std::unordered_map<std::string, double> dict_4details;

  if (is_detailed) {
    dict_4details["loss_total"] = loss_total;
    dict_4details["loss_inter"] = loss_inter;
    dict_4details["loss_intra"] = loss_intra;
  }

  return std::make_tuple(
      /////
      loss_total, dloss_over_dpose, true,

      dict_4details

  );
}

double CoreEvaluatorGrid4D::get_weight_intra() { return this->weight_intra_; }

void CoreEvaluatorGrid4D::set_weight_intra(double a_weight_intra) {
  this->weight_intra_ = a_weight_intra;
  return;
}

double CoreEvaluatorGrid4D::get_weight_collision_inter() { return this->weight_collision_inter_; }

void CoreEvaluatorGrid4D::set_weight_collision_inter(double a_weight_collision_inter) {
  this->weight_collision_inter_ = a_weight_collision_inter;
  return;
}

std::pair<double, std::array<double, 3>> CoreEvaluatorGrid4D::calc_nogridloss_collision_inter_for_1_heavyatom(

    int index_heavyatom,

    const std::array<double, 3>& position_heavyatom

) const {
  double nogridloss = 0.0;
  std::array<double, 3> dloss_over_dposition = {0, 0, 0};

  for (int i = 0; i < this->positions_inter_.size(); i++) {
    double i_d = this->calculate_distance_between_2_positions(

        position_heavyatom, this->positions_inter_[i]

    );
    double i_c = this->vdwradius_sum_inter_[index_heavyatom][i];
    double i_vdwd = i_d - i_c;

    double i_loss = 0.0;
    double i_dloss_over_dd2 = 0.0;
    if (i_vdwd < 0.0) {
      i_loss = i_vdwd * i_vdwd;
      i_dloss_over_dd2 = 1 - i_c / i_d;
    }

    nogridloss += i_loss;
    for (int j = 0; j < 3; j++) {
      dloss_over_dposition[j] += (

          i_dloss_over_dd2

          * 2 * (position_heavyatom[j] - this->positions_inter_[i][j])

      );
    }
  }

  return std::make_pair(nogridloss, dloss_over_dposition);
}

TupleTheEvaluated2Debug CoreEvaluatorGrid4D::evaluate_nogrid_given_weights(

    const std::vector<double>& pose,

    double weight_intra, double weight_collision_inter

) const {
  using pyvina::core::wrappers_for_exceptions::AssertGivenWhat;

  if (!(this->is_core_grid_4d_ready_)) {
    pyvina::core::wrappers_for_exceptions::Throw1ExceptionAfterPrinting(
        ".evaluate() but grid has not been precalculated");
  }

  const int num_degrees_of_freedom = this->core_pdbqt_ligand_.GetNumDegreesOfFreedom();
  AssertGivenWhat(num_degrees_of_freedom == pose.size(),

                  "num_degrees_of_freedom == pose.size()");
  AssertGivenWhat(this->weight_collision_inter_ == 0.0,

                  "this->weight_collision_inter_ == 0.0");

  auto dloss_over_dpose = std::vector<double>(num_degrees_of_freedom, 0.0);

  auto&& regarding_ligand_positions = this->core_pdbqt_ligand_.FormulaDerivativeCore_ConformationToCoords(pose);
  auto&& positions_heavy_atoms = std::get<0>(regarding_ligand_positions);
  auto&& dx_dy_dz_over_dpose = std::get<2>(regarding_ligand_positions);

  /////////////////////////////
  ///// inter loss
  ///// copied from ::evaluate
  /////////////////////////////
  double loss_inter = 0.0;
  for (int i = 0; i < this->core_pdbqt_ligand_.num_heavy_atoms; i++) {
    int i_celltype = this->get_celltype_by_index_heavy_atom(i);

    auto&& i_xyz = positions_heavy_atoms[i];

    if (!(this->core_grid_4d_.contains_coordinates(i_xyz))) {
      return this->CreateTupleTheEvaluatedInvalid2Debug();
    }

    auto&& i_cellindices = this->core_grid_4d_.get_cellindices_by_cellcoordinates(i_xyz);

    double i_loss = this->core_grid_4d_.value_[
        /////
        i_celltype][i_cellindices[0]][i_cellindices[1]][i_cellindices[2]];
    loss_inter += i_loss;

    double resolution = this->core_grid_4d_.resolution_;
    for (int j = 0; j < 3; j++) {
      auto j_cellindices = i_cellindices;
      j_cellindices[j] += 1;

      double j_loss = this->core_grid_4d_.value_[
          /////
          i_celltype][j_cellindices[0]][j_cellindices[1]][j_cellindices[2]];

      double dloss_over_dj = (j_loss - i_loss) / resolution;

      for (int k = 0; k < num_degrees_of_freedom; k++) {
        dloss_over_dpose[k] += dloss_over_dj * dx_dy_dz_over_dpose[i][j][k];
      }
    }
  }

  double energy_normalscore_inter = loss_inter;

  /////////////////////////////
  ///// inter loss
  ///// TODO: collision from formula
  /////////////////////////////
  double energy_collision_inter = 0.0;
  for (int i = 0; i < this->core_pdbqt_ligand_.num_heavy_atoms; i++) {
    auto&& i_xyz = positions_heavy_atoms[i];
    auto&& i_result = this->calc_nogridloss_collision_inter_for_1_heavyatom(i, i_xyz);

    energy_collision_inter += (

        weight_collision_inter

        * i_result.first

    );

    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < num_degrees_of_freedom; k++) {
        dloss_over_dpose[k] += (

            weight_collision_inter

            * i_result.second[j]

            * dx_dy_dz_over_dpose[i][j][k]

        );
      }
    }
  }
  loss_inter += energy_collision_inter;

  /////////////////////////////
  ///// intra loss
  ///// copied from ::evaluate
  /////////////////////////////
  double loss_intra = 0.0;
  // const double weight_intra = this->weight_intra_;

  for (const auto& i_pair : this->core_pdbqt_ligand_.interacting_pairs) {
    const auto& a_index = i_pair.i0;
    const auto& b_index = i_pair.i1;
    const auto& a_position = positions_heavy_atoms[a_index];
    const auto& b_position = positions_heavy_atoms[b_index];
    const auto& a_xs = this->core_pdbqt_ligand_.heavy_atoms[a_index].xs;
    const auto& b_xs = this->core_pdbqt_ligand_.heavy_atoms[b_index].xs;

    double i_distancesquare = this->calculate_distancesquare_between_2_positions(

        a_position, b_position

    );

    loss_intra += (

        weight_intra

        * this->core_grid_3d_intra_.score_by_grid_collision_only(a_xs, b_xs, i_distancesquare)

    );

    double dloss_over_ddistancesquare = this->core_grid_3d_intra_.get_derivative_by_grid_collision_only(

        a_xs, b_xs, i_distancesquare

    );

    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < num_degrees_of_freedom; k++) {
        dloss_over_dpose[k] += (

            weight_intra

            * dloss_over_ddistancesquare

            * 2 * (a_position[j] - b_position[j])

            * (dx_dy_dz_over_dpose[a_index][j][k] - dx_dy_dz_over_dpose[b_index][j][k])

        );
      }
    }
  }

  auto loss_total = loss_inter + loss_intra;

  /////////////////////////////
  ///// dict 4details
  /////////////////////////////
  std::unordered_map<std::string, double> dict_4details;

  dict_4details["loss_total"] = loss_total;
  dict_4details["loss_inter"] = loss_inter;
  dict_4details["loss_intra"] = loss_intra;
  dict_4details["energy_normalscore_inter"] = energy_normalscore_inter;
  dict_4details["energy_collision_inter"] = energy_collision_inter;
  dict_4details["weight_intra"] = weight_intra;
  dict_4details["weight_collision_inter"] = weight_collision_inter;

  double distance_inter_min = 99.0;
  double vdwdistance_inter_min = 99.0;

  for (int i = 0; i < this->core_pdbqt_ligand_.num_heavy_atoms; i++) {
    for (int j = 0; j < this->positions_inter_.size(); j++) {
      double ij_distance = this->calculate_distance_between_2_positions(

          positions_heavy_atoms[i], this->positions_inter_[j]

      );
      double ij_vdwdistance = ij_distance - this->vdwradius_sum_inter_[i][j];

      if (distance_inter_min > ij_distance) {
        distance_inter_min = ij_distance;
      }

      if (vdwdistance_inter_min > ij_vdwdistance) {
        vdwdistance_inter_min = ij_vdwdistance;
      }
    }
  }

  dict_4details["distance_inter_min"] = distance_inter_min;
  dict_4details["vdwdistance_inter_min"] = vdwdistance_inter_min;

  return std::make_tuple(
      /////
      loss_total, dloss_over_dpose, true,

      dict_4details

  );
}

TupleTheEvaluated2Debug CoreEvaluatorGrid4D::evaluate_given_option(

    const std::vector<double>& pose,

    const std::unordered_map<std::string, std::string>& option

) const {
  return this->evaluate_nogrid_given_weights(

      pose,

      std::stod(this->FindInOption(option, "weight_intra")),

      std::stod(this->FindInOption(option, "weight_collision_inter"))

  );
}

}  // namespace grid_4d
}  // namespace evaluators
}  // namespace core
}  // namespace pyvina
