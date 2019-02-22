from distutils.core import setup
from Cython.Build import cythonize
from numpy import get_include
from distutils.extension import Extension

ext_modules = [
    Extension('cytextsim',
              sources=["preprocessing/cytextsim.pyx"],
              include_dirs=[get_include()]),
    Extension('tokenizer',
              sources=["preprocessing/tokenizer.pyx"],
              include_dirs=[get_include()])
]

setup(
    name='preprocessing',
    ext_modules=cythonize(ext_modules)
)
