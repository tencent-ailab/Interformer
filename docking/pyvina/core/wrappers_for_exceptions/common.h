// Copyright 2022 Tencent Inc. All rights reserved.
//
// Author: ruiyuanqian@tencent.com (ruiyuanqian)
//
// 文件注释
// ...

#ifndef PYVINA_CORE_WRAPPERS_FOR_EXCEPTIONS_COMMON_H_
#define PYVINA_CORE_WRAPPERS_FOR_EXCEPTIONS_COMMON_H_

#include <stdexcept>
#include <string>

namespace pyvina {
namespace core {
namespace wrappers_for_exceptions {

void Throw1ExceptionAfterPrinting(const std::string& str_what);

void AssertGivenWhat(

    bool condition, const std::string& str_what = " :: Anonymous Assertion :: "

);

template <typename T>
bool IsNanInArray1D(const T& array_1d);
template <typename T>
bool IsNanInArray2D(const T& array_2d);
template <typename T>
bool IsNanInArray3D(const T& array_3d);

template <typename T>
std::string GetStrFromArray1D(const T& array_1d);
template <typename T>
std::string GetStrFromArray2D(const T& array_2d);

}  // namespace wrappers_for_exceptions
}  // namespace core
}  // namespace pyvina

#endif
