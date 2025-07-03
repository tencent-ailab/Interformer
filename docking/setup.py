"""
"""

import os
# import glob

from torch.utils.cpp_extension import CppExtension, BuildExtension
from setuptools import setup, find_packages



setup(
    name='PyVina',
    version='0.0.8',

    packages=find_packages(),

    ext_modules=[
        CppExtension(
            name='pyvina_core',
            sources=[

                'pyvina/core/bindings.cpp',

                'pyvina/core/array.cpp',
                'pyvina/core/atom.cpp',
                'pyvina/core/ligand.cpp',
                'pyvina/core/random_forest.cpp',
                'pyvina/core/random_forest_x.cpp',
                'pyvina/core/random_forest_y.cpp',
                'pyvina/core/receptor.cpp',
                'pyvina/core/result.cpp',
                'pyvina/core/scoring_function.cpp',

                'pyvina/core/tests/tensor_tests.cc',

                'pyvina/core/quaternion/quaternion_torch.cc',
                'pyvina/core/quaternion/quaternion_vanilla.cc',

                'pyvina/core/evaluator/eval_vina.cc',
                'pyvina/core/evaluator/evaluator_vina_vanilla.cc',
                'pyvina/core/evaluator/evaluator_vina_formula_derivative.cc',
                # 'pyvina/core/evaluator/core_evaluator_distancemap_grid.cc',

                'pyvina/core/wrappers_for_exceptions/common.cc',

                'pyvina/core/wrappers_for_random/common.cc',

                'pyvina/core/evaluators/core_evaluator_base.cc',
                'pyvina/core/evaluators/grid_4d/core_grid_4d.cc',
                'pyvina/core/evaluators/grid_4d/core_evaluator_grid_4d.cc',
                'pyvina/core/evaluators/grid_4d/core_evaluator_distancemap.cc',
                'pyvina/core/evaluators/grid_4d/core_evaluator_normalscore.cc',
                'pyvina/core/evaluators/grid_4d/core_evaluator_vinascore.cc',

                'pyvina/core/minimizers/core_minimizer_base.cc',
                'pyvina/core/minimizers/core_minimizer_bfgs.cc',

                'pyvina/core/samplers/core_sampler_base.cc',
                'pyvina/core/samplers/core_sampler_monte_carlo.cc',

                'pyvina/core/common/common.cc',
                'pyvina/core/common/format_convert.cc',

                'pyvina/core/minimizer/bfgs_torch.cc',
                'pyvina/core/minimizer/bfgs_formula_derivative.cc',

                'pyvina/core/scoring_function/scoring_function_base_1.cc',
                'pyvina/core/scoring_function/vina_scoring_function.cc',
                'pyvina/core/scoring_function/x_score_scoring_function.cc',

                'pyvina/core/differentiation/coord_differentiation.cc',

            ],
            include_dirs=[
                os.path.dirname(
                    os.path.abspath(__file__)
                ),
                # '/usr/include/openbabel-2.0'
            ],
            library_dirs=[
                # '/usr/lib/openbabel/2.3.2'
            ],
            extra_compile_args=[
                '-fopenmp'
            ],
            extra_link_args=[
                '-lgomp',
                '-lboost_system',
                # '-lopenbabel'
            ]
        )
    ],

    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=True)
    },

)
