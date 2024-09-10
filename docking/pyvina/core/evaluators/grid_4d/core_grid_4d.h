// Copyright 2022 Tencent Inc. All rights reserved.
//
// Author: ruiyuanqian@tencent.com (ruiyuanqian)
//
// 文件注释
// ...

#ifndef PYVINA_CORE_EVALUATORS_GRID_4D_CORE_GRID_4D_H_
#define PYVINA_CORE_EVALUATORS_GRID_4D_CORE_GRID_4D_H_

#include <array>
#include <tuple>
#include <vector>

#include "pyvina/core/ligand.hpp"

namespace pyvina {
namespace core {
namespace evaluators {
namespace grid_4d {

class CoreGrid4D {
 public:
  std::vector<std::vector<std::vector<std::vector<double> > > > value_;

  std::array<double, 3> corner_min_;
  std::array<double, 3> corner_max_;

  std::array<double, 3> double_shape3d_;
  std::array<int, 3> int_shape3d_;
  std::array<int, 3> int_shape3d_with_margin_;

  int num_celltypes_;

  int reciprocal_resolution_;
  double resolution_;

  CoreGrid4D(

      // pocket
      std::array<double, 3> corner_min, std::array<double, 3> corner_max,

      // .shape[0]
      int num_celltypes,

      // resolution
      int reciprocal_resolution

  );

  ///////////////////////////////////////////////
  ///// should_valid:
  ///// means indices involved should be valid in .value_[][][][]
  /////
  ///// therefore, position_on_margin is valid
  ///// although .contains_coordinates(position_on_margin) is false
  ///////////////////////////////////////////////
  void checkonly_cellindex_valid(

      int cellindex, int index_axis

  ) const;

  double get_cellcoordinate_by_cellindex(

      int cellindex, int index_axis,

      bool should_cellindex_valid = true

  ) const;

  int get_cellindex_by_cellcoordinate(

      double cellcoordinate, int index_axis,

      bool should_cellindex_valid = true

  ) const;

  std::array<double, 3> get_cellcoordinates_by_cellindices(

      const std::array<int, 3>& cellindices,

      bool should_cellindices_valid = true

  ) const;

  std::array<int, 3> get_cellindices_by_cellcoordinates(

      const std::array<double, 3>& cellcoordinates,

      bool should_cellindices_valid = true

  ) const;

  std::array<int, 2> get_minmax_cellindex(

      int index_axis,

      const std::array<double, 3>& position_so_far,

      const std::array<double, 3>& position_inter,

      double cutoff_distance

  ) const;

  bool contains_coordinates(

      std::array<double, 3> coordinates

  ) const;
};

}  // namespace grid_4d
}  // namespace evaluators
}  // namespace core
}  // namespace pyvina

#endif
