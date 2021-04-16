import numpy as np
import autogen_python
from autogen_python import CppADScalar

def test_function(in_x):
  out_y = autogen_python.ADVector([CppADScalar(1.0)])
  # convert to numpy
  # convert back to advector
  # out_y = [CppADScalar(1.0)]
  for i in range(len(in_x)):
    out_y[0] *= in_x[i]

  return out_y

ax = autogen_python.ADVector([CppADScalar(1.0), CppADScalar(2.0)])
# ax = [CppADScalar(1.0), CppADScalar(2.0)]
autogen_python.independent(ax)
ay = test_function(ax)
print(ay)
f = autogen_python.ADFun(ax, ay)

x = autogen_python.DoubleVector([2.0, 3.0])
# x = [2.0, 3.0]
y = f.forward(0, x)
print("y = ", y)
J = f.jacobian(x)
print("j = ", J)