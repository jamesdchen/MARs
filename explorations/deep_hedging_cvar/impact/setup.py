"""Build the Cython env in place: python setup.py build_ext --inplace"""
from setuptools import setup, Extension
from Cython.Build import cythonize

setup(ext_modules=cythonize(
    [Extension('cy_hedging', ['cy_hedging.pyx'], extra_compile_args=['-O3', '-march=native'])],
    compiler_directives={'language_level': 3}))
