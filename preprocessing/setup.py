from distutils.core import setup
from Cython.Build import cythonize
from numpy import get_include
from distutils.extension import Extension

ext_modules = [
    Extension('dedup.textsim',
              sources=["preprocessing/textsim.pyx"],
              include_dirs=[get_include()],
              language="c++")
]

setup(
    name='prepro',
    ext_modules=cythonize(ext_modules)
)
