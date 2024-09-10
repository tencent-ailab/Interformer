// Copyright 2022 Tencent Inc. All rights reserved.
//
// Author: ruiyuanqian@tencent.com (ruiyuanqian)
//
// 文件注释
// ...

#ifndef PYVINA_CORE_CONSTANTS_COMMON_H_
#define PYVINA_CORE_CONSTANTS_COMMON_H_

#include <cmath>

namespace pyvina {
namespace core {
namespace constants {

const double kPi = 3.141592653589793238463;
const double kSqrt2Pi = std::sqrt(2 * kPi);

const double kSmallValueForZeroRotation = 1e-6;

const double kRangeRandomTranslationXYZ = 1.0;
const double kRangeRandomRotationvectorXYZ = 0.01;

const int kNumMaxRerandomizePose = 1000;
const double kEnergyInvalid = 9999999.99;

const int kNumMaxKeptEachMonteCarlo = 500;

}  // namespace constants
}  // namespace core
}  // namespace pyvina

#endif
