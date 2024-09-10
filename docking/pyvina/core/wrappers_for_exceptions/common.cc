// Copyright 2022 Tencent Inc. All rights reserved.
//
// Author: ruiyuanqian@tencent.com (ruiyuanqian)
//
// 文件注释
// ...

#include "pyvina/core/wrappers_for_exceptions/common.h"

#include <array>
#include <cmath>
#include <iostream>
#include <vector>

namespace pyvina {
namespace core {
namespace wrappers_for_exceptions {

void Throw1ExceptionAfterPrinting(const std::string& str_what) {
  const static int kNumRepeatsPrinting = 1;

  for (int i = 0; i < kNumRepeatsPrinting; i++) {
    printf(str_what.c_str());
    std::cout << "\n\n :: FLUSH :: \n\n" << std::flush;
  }

  throw std::logic_error(str_what);
}

void AssertGivenWhat(bool condition, const std::string& str_what) {
  if (!(condition)) {
    Throw1ExceptionAfterPrinting(" :: Assertion Error :: " + str_what + " :: \n");
  }
}

template <typename T>
bool IsNanInArray1D(const T& array_1d) {
  for (const auto& i_value : array_1d) {
    if (std::isnan(i_value)) {
      return true;
    }
  }
  return false;
}

template <typename T>
bool IsNanInArray2D(const T& array_2d) {
  for (const auto& i_1d : array_2d) {
    if (IsNanInArray1D(i_1d)) {
      return true;
    }
  }
  return false;
}

template <typename T>
bool IsNanInArray3D(const T& array_3d) {
  for (const auto& i_2d : array_3d) {
    if (IsNanInArray2D(i_2d)) {
      return true;
    }
  }
  return false;
}

template <typename T>
std::string GetStrFromArray1D(const T& array_1d) {
  std::string str_array_1d = "[ ";

  for (const auto& i_value : array_1d) {
    str_array_1d += (std::to_string(i_value) + ", ");
  }

  return (str_array_1d + "]");
}

template <typename T>
std::string GetStrFromArray2D(const T& array_2d) {
  std::string str_array_2d = "[\n";
  for (const auto& i_id : array_2d) {
    str_array_2d += (GetStrFromArray1D(i_id) + "\n");
  }
  return (str_array_2d + "]\n");
}

template bool IsNanInArray1D(const std::vector<double>&);
template bool IsNanInArray2D(const std::vector<std::vector<double>>&);
template bool IsNanInArray3D(const std::vector<std::vector<std::vector<double>>>&);

template bool IsNanInArray1D(const std::array<double, 3>&);
template bool IsNanInArray1D(const std::array<double, 4>&);
template bool IsNanInArray2D(const std::vector<std::array<double, 3>>&);
template bool IsNanInArray2D(const std::vector<std::array<double, 4>>&);

template std::string GetStrFromArray1D(const std::vector<double>&);
template std::string GetStrFromArray2D(const std::vector<std::vector<double>>&);

template std::string GetStrFromArray1D(const std::array<double, 3>&);
template std::string GetStrFromArray1D(const std::array<double, 4>&);
template std::string GetStrFromArray2D(const std::vector<std::array<double, 3>>&);
template std::string GetStrFromArray2D(const std::vector<std::array<double, 4>>&);

}  // namespace wrappers_for_exceptions
}  // namespace core
}  // namespace pyvina
