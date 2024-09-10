// Copyright 2022 Tencent Inc. All rights reserved.
//
// Author: ruiyuanqian@tencent.com (ruiyuanqian)
//
// 文件注释
// ...

#include "pyvina/core/wrappers_for_random/common.h"

#include <omp.h>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "pyvina/core/wrappers_for_exceptions/common.h"

namespace pyvina {
namespace core {
namespace wrappers_for_random {

std::vector<std::mt19937_64> generators_for_threads;

bool _temp_for_initialization = []() {
  printf(" :: before initialization :: generators_for_threads.size() => [ %d ] :: \n", generators_for_threads.size());

  std::vector<std::random_device::result_type> list_seeds;

  int threadmax = omp_get_max_threads();
  std::random_device rd;
  for (int i = 0; i < threadmax; i++) {
    list_seeds.push_back(rd());
  }

  SetGeneratorsForThreadsGivenSeeds(list_seeds);

  printf(" ::  after initialization :: generators_for_threads.size() => [ %d ] :: \n", generators_for_threads.size());
  return true;
}();

void SetGeneratorsForThreadsGivenSeeds(

    const std::vector<std::random_device::result_type>& list_seeds

) {
  std::vector<std::mt19937_64> generators_new;

  int threadlimit = omp_get_thread_limit();
  int threadmax = omp_get_max_threads();
  printf(" :: threadlimit = [ %d ] :: \n", threadlimit);
  printf(" :: threadmax   = [ %d ] :: \n", threadmax);

  if (threadmax != list_seeds.size()) {
    pyvina::core::wrappers_for_exceptions::Throw1ExceptionAfterPrinting(

        " :: threadmax != list_seeds.size() :: \n"

        " :: threadmax => [ " +
        std::to_string(threadmax) + " ] :: \n"

        + " :: list_seeds.size() => [ " + std::to_string(list_seeds.size()) + " ] :: \n"

    );
  }

  for (int i = 0; i < threadmax; i++) {
    const auto& i_seed = list_seeds[i];

    std::cout << " :: i => [ " << i

              << " ] :: i_seed to i_generator_for_thread (pseudo-random generator) => [ " << i_seed

              << " ] :: " << std::endl;

    generators_new.push_back(

        // std::mt19937_64(rd())
        std::mt19937_64(i_seed)

    );
  }

  generators_for_threads = std::move(generators_new);
  return;
}

int checkget_threadnum() {
  int threadnum = omp_get_thread_num();

  if (threadnum >= generators_for_threads.size()) {
    pyvina::core::wrappers_for_exceptions::Throw1ExceptionAfterPrinting(

        "\nthreadnum   = " + std::to_string(omp_get_thread_num()) +

        "\nthreadlimit = " + std::to_string(omp_get_thread_limit()) +

        "\nthreadmax   = " + std::to_string(omp_get_max_threads()) + "\n"

    );
  }

  return threadnum;
}

int get_uniform_int_between(int a, int b) {
  int threadnum = checkget_threadnum();

  std::uniform_int_distribution<> distribution(a, b);

  return distribution(generators_for_threads[threadnum]);
}

double get_uniform_real_between(double a, double b) {
  int threadnum = checkget_threadnum();

  std::uniform_real_distribution<> distribution(a, b);

  return distribution(generators_for_threads[threadnum]);
}

double get_normal_real_given(double mean, double stddev) {
  int threadnum = checkget_threadnum();

  std::normal_distribution<> distribution(mean, stddev);

  return distribution(generators_for_threads[threadnum]);
}

double GetStandardnormalReal() { return get_normal_real_given(0.0, 1.0); }

}  // namespace wrappers_for_random
}  // namespace core
}  // namespace pyvina
