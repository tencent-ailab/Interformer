// Copyright 2022 Tencent Inc. All rights reserved.
//
// Author: ruiyuanqian@tencent.com (ruiyuanqian)
//
// 文件注释
// ...

#ifndef PYVINA_CORE_WRAPPERS_FOR_RANDOM_COMMON_H_
#define PYVINA_CORE_WRAPPERS_FOR_RANDOM_COMMON_H_

#include <random>
#include <vector>

namespace pyvina {
namespace core {
namespace wrappers_for_random {

void SetGeneratorsForThreadsGivenSeeds(

    const std::vector<std::random_device::result_type>& list_seeds

);

int get_uniform_int_between(int a, int b);
double get_uniform_real_between(double a, double b);
double get_normal_real_given(double mean, double stddev);
double GetStandardnormalReal();

}  // namespace wrappers_for_random
}  // namespace core
}  // namespace pyvina

#endif
