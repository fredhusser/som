from distutils.core import setup
try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension

from Cython.Distutils import build_ext
from numpy import get_include

# Build the Cython code
# Usage: python setup.py build_ext --inplace clean
ext_modules = [Extension('som._csom',
                         ['som/_csom.pyx'],
                         include_dirs=[get_include()],
                        )]

setup(name = "som._csom",
        cmdclass = {'build_ext': build_ext},
        ext_modules = ext_modules)

# Build the package
setup(
    name='som',
    version='1.0',
    packages=[''],
    url='',
    license='BSD',
    author='fredhusser',
    author_email='fredhusser@gmail.com',
    description='', requires=['sklearn', 'sklearn', 'matplotlib', 'scipy']
)
