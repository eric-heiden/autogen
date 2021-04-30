# autogen
Code generation for automatic differentiation with GPU support.

## Python

Note that you need python3 to run this.

Install pybind11:
```sh
pip install pybind11
```

Update the current PIP installation of autogen. You should also specify the 
compiler version to be compatible with c++17:
```sh
CC=gcc-9 CXX=g++-9 python setup.py install
```
