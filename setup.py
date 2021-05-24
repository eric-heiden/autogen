import platform
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, ParallelCompile, build_ext, naive_recompile

__version__ = "0.0.1"

ext_modules = [
    Pybind11Extension(
        "_autogen",
        ['python/autogen_python.cpp', 'CppAD/cppad_lib/json_writer.cpp',
            'CppAD/cppad_lib/cpp_graph_op.cpp'],
        include_dirs=['python/pybind11/include/',
                      'CppAD/include/', 'CppADCodeGen/include/', 'include/'],
        extra_compile_args=[
            "-fopenmp", "-w"] if platform.system() in ("Linux", "Darwin") else [],
        extra_link_args=['-lgomp']if platform.system() in ("Linux",
                                                           "Darwin") else [],
        define_macros=[('VERSION_INFO', __version__)],
        cxx_std=17)
]

ParallelCompile("NPY_NUM_BUILD_JOBS",
                needs_recompile=naive_recompile).install()

setup(name='autogen',
      version=__version__,
      description='Code generation for automatic differentiation with GPU support',
      long_description="",
      url='https://github.com/eric-heiden/autogen',
      author='autogen authors',
      author_email='',
      license='MIT',
      cmdclass={"build_ext": build_ext},
      packages=['autogen'],
      package_dir={'': 'python'},
      ext_modules=ext_modules
      )
