# autogen
Code generation for automatic differentiation with GPU support.

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
