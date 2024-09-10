// Copyright 2022 Tencent Inc. All rights reserved.
//
// Author: ruiyuanqian@tencent.com (ruiyuanqian)
//
// 文件注释
// ...

#ifndef PYVINA_CORE_EVALUATORS_GRID_4D_CORE_EVALUATOR_GRID_4D_H_
#define PYVINA_CORE_EVALUATORS_GRID_4D_CORE_EVALUATOR_GRID_4D_H_

#include <iostream>
#include <tuple>
#include <vector>

#include "pyvina/core/evaluators/core_evaluator_base.h"
#include "pyvina/core/evaluators/grid_4d/core_grid_4d.h"
#include "pyvina/core/ligand.hpp"

#include "pyvina/core/scoring_function.hpp"

namespace pyvina {
namespace core {
namespace evaluators {
namespace grid_4d {

class CoreEvaluatorGrid4D : public CoreEvaluatorBase {
 protected:
  bool is_core_grid_4d_ready_ = false;

 public:
  CoreGrid4D core_grid_4d_;

  scoring_function core_grid_3d_intra_ = scoring_function();

  double weight_intra_ = 30.0;
  double weight_collision_inter_ = 40.0;

  std::vector<std::vector<double> > vdwradius_sum_inter_;
  double cutoff_vdwdistance_ = 0.0;
  double max_vdwradius_sum_ = 0.0;
  double cutoff_distance_ = 0.0;
  std::vector<std::array<double, 3> > positions_inter_;

  CoreEvaluatorGrid4D(

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

  );

  virtual int get_celltype_by_index_heavy_atom(

      int index_heavy_atom

  ) const = 0;

  virtual double calculate_no_grid_loss_inter_for_1_heavy_atom(

      int index_heavy_atom, const std::array<double, 3>& coordinates_heavy_atom

  ) const = 0;

  //   virtual void precalculate_losses_inter_in_grid_for_1_heavy_atom(int index_heavy_atom);

  virtual void precalculate_losses_inter_in_grid_for_1_heavy_atom(

      int index_heavy_atom, int options_bitwise

  );

  virtual void precalculate_core_grid_4d_(

      //   int options_bitwise = 0
      int options_bitwise

  );

  TupleTheEvaluated evaluate(

      const std::vector<double>& pose

  ) const;

  TupleTheEvaluated2Debug evaluate_2debug(

      const std::vector<double>& pose

  ) const;

  TupleTheEvaluated2Debug evaluate_given_option(

      const std::vector<double>& pose,

      const std::unordered_map<std::string, std::string>& option

  ) const;

  TupleTheEvaluated2Debug evaluate_no_grid(

      const std::vector<double>& pose,

      bool is_detailed = false

  ) const;

  double get_weight_intra();
  void set_weight_intra(double a_weight_intra);

  double get_weight_collision_inter();
  void set_weight_collision_inter(double a_weight_collision_inter);

  std::pair<double, std::array<double, 3>> calc_nogridloss_collision_inter_for_1_heavyatom(

      int index_heavyatom,

      const std::array<double, 3>& position_heavyatom

  ) const;

  TupleTheEvaluated2Debug evaluate_nogrid_given_weights(

      const std::vector<double>& pose,

      double weight_intra, double weight_collision_inter

  ) const;
};

}  // namespace grid_4d
}  // namespace evaluators
}  // namespace core
}  // namespace pyvina

#endif
