import numpy as np
import autogen as ag

def test_function(in_x):
  out_y = np.ones(1, dtype=ag.scalar_type())
  for i in range(len(in_x)):
    out_y[0] *= in_x[i] ** 2.

  return out_y


f = ag.trace(test_function, [1., 2.])
gen = ag.GeneratedCppAD(f)

x = [2.0, 3.0]
y = f.forward(x)
print("y = ", y)
J = f.jacobian(x)
print("j = ", J)

x = [2.0, 3.0]
y = gen.forward(x)
print("y = ", y)
J = gen.jacobian(x)
print("j = ", J)