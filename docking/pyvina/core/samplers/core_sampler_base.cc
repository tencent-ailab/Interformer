// Copyright 2022 Tencent Inc. All rights reserved.
//
// Author: ruiyuanqian@tencent.com (ruiyuanqian)
//
// 文件注释
// ...

#include "pyvina/core/samplers/core_sampler_base.h"

namespace pyvina {
namespace core {
namespace samplers {

CoreSamplerBase::CoreSamplerBase(

    const minimizers::CoreMinimizerBase& core_minimizer_base

    )
    : core_minimizer_base_{core_minimizer_base} {
  ;
}

}  // namespace samplers
}  // namespace core
}  // namespace pyvina
