"""
"""

import os
# import glob

from torch.utils.cpp_extension import CppExtension, BuildExtension
from setuptools import setup, find_packages

# def find_h_and_or_cc(relative_path, possible_suffix_names):
#     file_paths = []

#     root_abs_path = os.path.dirname(
#         __file__
#     )
#     target_abs_path = os.path.join(
#         root_abs_path,
#         relative_path
#     )

#     for _, _, file_names in os.walk(target_abs_path):
#         for file_name in file_names:
#             if (file_name.split('.')[-1].lower()) in possible_suffix_names:
#                 file_paths.append(
#                     os.path.join(
#                         target_abs_path,
#                         file_name
#                     )
#                 )

#     return file_paths

# core_cc = find_h_and_or_cc('pyvina/core',['c','cc','cpp'])
# core_h_cc = find_h_and_or_cc('pyvina/core',['h','hpp','c','cc','cpp'])

# print(" :: c/cc/cpp files :: ")
# print(core_cc)

# print(" :: h/hpp/c/cc/cpp files :: ")
# print(core_h_cc)

setup(
    name='PyVina',
    version='0.0.7',

    # packages=[
    #     'pyvina',
    #     'pyvina.ligand',
    #     'pyvina.third_party_tool',
    # ],
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

# import os
# import re
# import sys
# import platform
# import subprocess

# from setuptools import setup, Extension
# from setuptools.command.build_ext import build_ext
# from distutils.version import LooseVersion


# class CMakeExtension(Extension):
#     def __init__(self, name, sourcedir=''):
#         Extension.__init__(self, name, sources=[])
#         self.sourcedir = os.path.abspath(sourcedir)


# class CMakeBuild(build_ext):
#     def run(self):
#         try:
#             out = subprocess.check_output(['cmake', '--version'])
#         except OSError:
#             raise RuntimeError("CMake must be installed to build the following extensions: " +
#                                ", ".join(e.name for e in self.extensions))

#         if platform.system() == "Windows":
#             cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
#             if cmake_version < '3.1.0':
#                 raise RuntimeError("CMake >= 3.1.0 is required on Windows")

#         for ext in self.extensions:
#             self.build_extension(ext)

#     def build_extension(self, ext):
#         extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
#         # required for auto-detection of auxiliary "native" libs
#         if not extdir.endswith(os.path.sep):
#             extdir += os.path.sep

#         import torch
#         path_to_libtorch = os.path.dirname(
#             torch.__file__
#         )

#         cmake_args = [
#             '-DCMAKE_PREFIX_PATH=' + path_to_libtorch ,
#             '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
#             '-DPYTHON_EXECUTABLE=' + sys.executable
#             ]

#         cfg = 'Debug' if self.debug else 'Release'
#         build_args = ['--config', cfg]

#         if platform.system() == "Windows":
#             cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
#             if sys.maxsize > 2**32:
#                 cmake_args += ['-A', 'x64']
#             build_args += ['--', '/m']
#         else:
#             cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
#             build_args += ['--', '-j2']

#         env = os.environ.copy()
#         env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
#                                                               self.distribution.get_version())
#         if not os.path.exists(self.build_temp):
#             os.makedirs(self.build_temp)
#         subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
#         subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

# setup(
#     name='pyvina',
#     version='0.0.1',
#     author='Haidong Lan',
#     author_email='turbo0628g@gmail.com',
#     description='The vina docking tool in Python modules',
#     long_description='',
#     ext_modules=[CMakeExtension('pyvina')],
#     cmdclass=dict(build_ext=CMakeBuild),
#     zip_safe=False,
#     packages=['pyvina']
# )
