// Copyright 2022 Tencent Inc. All rights reserved.
//
// Author: ruiyuanqian@tencent.com (ruiyuanqian)
//
// 文件注释
// ...

#include "pyvina/core/evaluators/grid_4d/core_grid_4d.h"

// #include <math.h>
// #include <omp.h>
#include <stdexcept>

#include "pyvina/core/wrappers_for_exceptions/common.h"

namespace pyvina {
namespace core {
namespace evaluators {
namespace grid_4d {

CoreGrid4D::CoreGrid4D(

    // pocket
    std::array<double, 3> corner_min, std::array<double, 3> corner_max,

    // .shape[0]
    int num_celltypes,

    // resolution
    int reciprocal_resolution

    )
    : corner_min_{corner_min},
      num_celltypes_{num_celltypes},
      reciprocal_resolution_{reciprocal_resolution}

{
  if (!(this->num_celltypes_ > 0 && this->num_celltypes_ < 100)) {
    pyvina::core::wrappers_for_exceptions::Throw1ExceptionAfterPrinting(
        "!(this->num_celltypes_ > 0 && this->num_celltypes_ < 100)");
  }

  this->resolution_ = (1.0 / this->reciprocal_resolution_);

  for (auto i = 0; i < 3; i++) {
    if (!(corner_min[i] < corner_max[i])) {
      pyvina::core::wrappers_for_exceptions::Throw1ExceptionAfterPrinting("!(corner_min[i] < corner_max[i])");
    }

    double i_length_raw = corner_max[i] - corner_min[i];

    this->int_shape3d_[i] = (

        int(i_length_raw * this->reciprocal_resolution_) + 2

    );
    this->int_shape3d_with_margin_[i] = (this->int_shape3d_[i] + 2);

    this->double_shape3d_[i] = (this->int_shape3d_[i] * this->resolution_);

    this->corner_max_[i] = this->corner_min_[i] + this->double_shape3d_[i];
  }

  this->value_ = std::vector<std::vector<std::vector<std::vector<double> > > >(
      this->num_celltypes_,
      std::vector<std::vector<std::vector<double> > >(
          this->int_shape3d_with_margin_[0],
          std::vector<std::vector<double> >(this->int_shape3d_with_margin_[1],
                                            std::vector<double>(this->int_shape3d_with_margin_[2], 0.0))));
}

// double CoreGrid4D::get_coordinate_by_index_and_axis(int index, int axis) {
//   if (!(axis >= 0 && axis < 3)) {
//     throw std::invalid_argument("!(axis >= 0 && axis < 3)");
//   }

//   if (!(index >= 0 && index < this->int_shape3d_with_margin_[axis])) {
//     throw std::invalid_argument("!(index > 0 && index < this->int_shape3d_with_margin_[axis])")
//   }

//   return (this->corner_min_[axis]
//           //
//           - (0.5 * this->resolution_)
//           //
//           + (index * this->resolution_));
// }

void CoreGrid4D::checkonly_cellindex_valid(

    int cellindex, int index_axis

) const {
  if (!(cellindex >= 0

        && cellindex < this->int_shape3d_with_margin_[index_axis])) {
    pyvina::core::wrappers_for_exceptions::Throw1ExceptionAfterPrinting(

        " :: invalid cellindex :: " + std::to_string(cellindex) +
        " :: \n"

        " :: index_axis :: " +
        std::to_string(index_axis) +
        " :: \n"

        " :: grid.shape[index_axis] :: " +
        std::to_string(this->int_shape3d_with_margin_[index_axis]) + " :: \n"

    );
  }
}

double CoreGrid4D::get_cellcoordinate_by_cellindex(

    int cellindex, int index_axis,

    bool should_cellindex_valid

) const {
  if (should_cellindex_valid) {
    this->checkonly_cellindex_valid(cellindex, index_axis);
  }

  return (

      this->corner_min_[index_axis]

      - (0.5 * this->resolution_)

      + (cellindex * this->resolution_)

  );
}

int CoreGrid4D::get_cellindex_by_cellcoordinate(

    double cellcoordinate, int index_axis,

    bool should_cellindex_valid

) const {
  double a_delta = (cellcoordinate - this->corner_min_[index_axis]);
  int cellindex = (

      1 + int(a_delta * this->reciprocal_resolution_)

  );

  if (should_cellindex_valid) {
    this->checkonly_cellindex_valid(cellindex, index_axis);
  }

  return cellindex;
}

std::array<double, 3> CoreGrid4D::get_cellcoordinates_by_cellindices(

    const std::array<int, 3>& cellindices,

    bool should_cellindices_valid

) const {
  std::array<double, 3> cellcoordinates;

  // int i;
  // for (i = 0; i < 3; i++) {
  //   if (!(cellindices[i] >= 0 && cellindices[i] < this->int_shape3d_with_margin_[i])) {
  //     throw std::invalid_argument("!(cellindices[i] >= 0 && cellindices[i] < this->int_shape3d_with_margin_[i])");
  //   }

  //   cellcoordinates[i] = (this->corner_min_[i]
  //                         //
  //                         - (0.5 * this->resolution_)
  //                         //
  //                         + (cellindices[i] * this->resolution_));
  // }

  for (int i = 0; i < 3; i++) {
    cellcoordinates[i] = this->get_cellcoordinate_by_cellindex(

        cellindices[i], i, should_cellindices_valid

    );
  }

  return cellcoordinates;
}

std::array<int, 3> CoreGrid4D::get_cellindices_by_cellcoordinates(

    const std::array<double, 3>& cellcoordinates,

    bool should_cellindices_valid

) const {
  std::array<int, 3> cellindices;

  // int i;
  // for (i = 0; i < 3; i++) {
  //   double i_delta = (cellcoordinates[i] - this->corner_min_[i]);

  //   if (!(i_delta > 0 && i_delta < this->double_shape3d_[i])) {
  //     throw std::invalid_argument("!(i_delta > 0 && i_delta < this->double_shape3d_[i])");
  //   }

  //   cellindices[i] = 1 + int(i_delta / this->resolution_);
  // }

  for (int i = 0; i < 3; i++) {
    cellindices[i] = this->get_cellindex_by_cellcoordinate(

        cellcoordinates[i], i, should_cellindices_valid

    );
  }

  return cellindices;
}

std::array<int, 2> CoreGrid4D::get_minmax_cellindex(

    int index_axis,

    const std::array<double, 3>& position_so_far,

    const std::array<double, 3>& position_inter,

    double cutoff_distance

) const {
  double distancesquare_rest = cutoff_distance * cutoff_distance;
  for (int i = 0; i < 3; i++) {
    if (i == index_axis) {
      continue;
    }

    const double i_delta = position_so_far[i] - position_inter[i];
    distancesquare_rest -= (i_delta * i_delta);
  }

  if (distancesquare_rest < 0) {
    return std::array<int, 2>{-1, -2};
  }

  const double distance_rest = std::sqrt(distancesquare_rest);

  const double a_min_cellcoordinate = position_inter[index_axis] - distance_rest;
  const double a_max_cellcoordinate = position_inter[index_axis] + distance_rest;

  int a_min_cellindex = (

      this->get_cellindex_by_cellcoordinate(a_min_cellcoordinate, index_axis, false)

      - 2);
  int a_max_cellindex = (

      this->get_cellindex_by_cellcoordinate(a_max_cellcoordinate, index_axis, false)

      + 2);

  return std::array<int, 2>{

      std::max(0, a_min_cellindex),

      std::min(this->int_shape3d_with_margin_[index_axis] - 1, a_max_cellindex)

  };
}

bool CoreGrid4D::contains_coordinates(

    std::array<double, 3> coordinates

) const {
  for (int i = 0; i < 3; i++) {
    if (!(coordinates[i] > this->corner_min_[i]
          //
          && coordinates[i] < this->corner_max_[i])) {
      return false;
    }
  }

  return true;
}

}  // namespace grid_4d
}  // namespace evaluators
}  // namespace core
}  // namespace pyvina
