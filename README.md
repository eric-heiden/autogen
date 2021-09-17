# autogen
Code generation for automatic differentiation with GPU support.

## Requirements

The library requires a C++ compiler with stable support for C++17, for example
* gcc-9 or later (gcc-8 seems to have some [issues with std::filesystem](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=90050))
* Clang 10 or later
* MSVC 2019 or later

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

| AutoDiff Mode      | UNIX | Windows |
| ------------------- |  :---:  |  :---:  |
| CppAD                | ✅ | ✅ |
| CPU code generation  | ✅ | ❌ |
| CUDA code generation | ✅ | ✅ |
