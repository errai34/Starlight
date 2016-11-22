from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy


include_gsl_dir = "/usr/local/include/"
lib_gsl_dir = "/usr/local/lib/"

args = {
    "libraries": ["m", "gsl", "cblas"],
    "include_dirs": [numpy.get_include(), include_gsl_dir],
    "library_dirs": [lib_gsl_dir],
    "extra_link_args": ['-fopenmp'],
    "extra_compile_args": ["-ffast-math", "-fopenmp",
                          "-Wno-uninitialized",
                          "-Wno-maybe-uninitialized",
                          "-Wno-unused-function"]  # -march=native
    }

ext_modules = [
    Extension("starlight.gibbs_cy",  ["starlight/gibbs_cy.pyx"], **args)
    ]

setup(
  name="starlight",
  cmdclass={"build_ext": build_ext},
  ext_modules=ext_modules)
