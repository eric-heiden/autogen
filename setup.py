import platform
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, ParallelCompile, build_ext, naive_recompile

__version__ = "0.0.1"

debug_mode = True

if platform.system() in ("Linux", "Darwin"):
    extra_compile_args = ["-fopenmp", "-w"]
    extra_link_args = ['-lgomp']
    if debug_mode:
        extra_link_args.append('-g')
else:
    extra_compile_args = ["/openmp"]
    extra_link_args = []
    if debug_mode:
        extra_compile_args.append('/DEBUG:FULL')
        extra_compile_args.append('/Od')
        extra_link_args.append('/DEBUG:FULL')

ext_modules = [
    Pybind11Extension(
        "_autogen",
        ['python/autogen_python.cpp',
         'src/autogen.cpp'
         # 'CppAD/cppad_lib/json_writer.cpp',
         # 'CppAD/cppad_lib/cpp_graph_op.cpp'
         ],
        include_dirs=['python/pybind11/include/',
                      'CppAD/include/', 'CppADCodeGen/include/', 'include/'],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
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
      author='Eric Heiden',
      author_email='me@eric-heiden.com',
      license='MIT',
      cmdclass={"build_ext": build_ext},
      packages=['autogen'],
      package_dir={'': 'python'},
      ext_modules=ext_modules
      )
