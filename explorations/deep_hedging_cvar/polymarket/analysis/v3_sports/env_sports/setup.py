"""Build the C env for PufferLib 3.0: python setup.py build_ext --inplace

Compiles binding.c (mm.h behind PufferLib's ocean env_binding.h) into the Python extension
`binding`. The data files come from data/build.py and data/features.py.
"""
import os

import numpy
import pufferlib
from setuptools import setup, Extension

OCEAN = os.path.join(os.path.dirname(pufferlib.__file__), 'ocean')

setup(name='mm_binding', ext_modules=[Extension(
    'binding', ['binding.c'], depends=['mm.h'], include_dirs=[numpy.get_include(), OCEAN],
    extra_compile_args=['-O3', '-march=native', '-ffp-contract=off', '-Wno-unused-function',
                        '-Wno-unused-variable', '-Wno-format-truncation'])])
