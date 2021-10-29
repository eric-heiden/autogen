# autogen
[![autogen-docs](https://github.com/eric-heiden/autogen/actions/workflows/ci.yml/badge.svg)](https://github.com/eric-heiden/autogen/actions/workflows/ci.yml)

Code generation for automatic differentiation with GPU support.

This library leverages CppAD and CppADCodeGen to trace C++ and Python code, and turns it into efficient CUDA or C code.
At the same time, the Jacobian and Hessian code can be automatically generated through reverse-mode or forward-mode automatic differentiation.
The generated code is compiled to a dynamic library which typically runs orders of magnitude faster than the original user code that was traced,
while multiple calls to the forward or backward versions of the function can be parallelized through CUDA or OpenMP.

## Requirements

The library requires CMake and a C++ compiler with stable support for C++17, for example
* gcc-9 or later (gcc-8 seems to have some [issues with std::filesystem](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=90050))
* Clang 10 or later
* MSVC 2019 or later

First, check out the git submodules via
```
git submodule update --init --recursive
```

## Python

Note: Only Python 3.4 and newer is supported.

Install pybind11:
```sh
pip install pybind11
```

For development, install autogen via the following command:
```sh
pip install -e .
```

To specify explicitly a compatible C++17 compiler, you can do so via preprocessor definitions:
```sh
CC=gcc-9 CXX=g++-9 pip install -e .
```

## Features

The following features are available on the different operating systems:

| AutoDiff Mode        | UNIX            | Windows           |
| -------------------- | :-------------- |  :--------------  |
| CppAD tracing        | ✅              | ✅               |
| CPU code generation  | ✅ (GCC, Clang) | ✅ (MSVC, Clang) |
| CUDA code generation | ✅ (NVCC)       | ✅ (NVCC)        |

### Windows CPU Support
CPU-bound code compilation on Windows is available through Microsoft Visual C++ (MSVC) and the Clang compiler at the moment.
Depending on the selected compiler/linker, make sure `cl.exe` and `link.exe`, or `clang.exe` are available on the system path. It might be necessary to first load the build variables in the console session by running
```
"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
```
