from distutils.core import setup
from Cython.Build import cythonize
from numpy import get_include
from distutils.extension import Extension
import numpy

ext_modules = [
    Extension('dedup.lcs',
              sources=["preprocessing/lcs.pyx"],
              include_dirs=[get_include()],
              language="c++")
]

setup(
    name='dedup_prepro',
    ext_modules=cythonize(ext_modules),
    install_requires=[
        'Cython>=0.21.1',
        'numpy',
        'scipy'
    ]
)
