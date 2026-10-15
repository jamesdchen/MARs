"""Build the v3 C env for PufferLib 3.0: python setup.py build_ext --inplace

Compiles binding3.c (mm3.h behind PufferLib's ocean env_binding.h) into the Python extension
`binding3`. It does not touch ../binding*.so (v2), so both envs import side by side.
"""
import os

import numpy
import pufferlib
from setuptools import setup, Extension

OCEAN = os.path.join(os.path.dirname(pufferlib.__file__), 'ocean')

setup(name='mm3_binding', ext_modules=[Extension(
    'binding3', ['binding3.c'], depends=['mm3.h'], include_dirs=[numpy.get_include(), OCEAN],
    extra_compile_args=['-O3', '-march=native', '-ffp-contract=off', '-Wno-unused-function',
                        '-Wno-unused-variable', '-Wno-format-truncation'])])
