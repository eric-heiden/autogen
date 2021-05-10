import numpy as np
import autogen as ag

def test_function(in_x):
  out_y = np.ones(1, dtype=ag.scalar_type())
  for i in range(len(in_x)):
    out_y[0] *= in_x[i] ** 2.

  return out_y

# def test_function(in_x):
#   out_y = np.ones(1, dtype=ag.CppADScalar)
#   for i in range(len(in_x)):
#     out_y[0] *= in_x[i] ** 2.

#   return out_y

# def test_function_CG(in_x):
#   out_y = np.ones(1, dtype=ag.CGScalar)
#   for i in range(len(in_x)):
#     out_y[0] *= in_x[i] ** 2.

#   return out_y


f = ag.trace(test_function, [1., 2.], mode=ag.Mode.CPU)
print(f.to_json())

x = [2.0, 3.0]
y = f.forward(x)
print("y = ", y)
J = f.jacobian(x)
print("j = ", J)

# xs = [1., 2.]
# ad_x = ag.CGVector([ag.CGScalar(x) for x in xs])
# ag.independent(ad_x)
# for i in range(len(xs)):
#   xs[i] = ad_x[i]
# ys = test_function_CG(xs)
# ad_y = ag.CGVector(list(ys))
# f = ag.CGFunction(ad_x, ad_y)
# print(type(f))
# gen = ag.Autogen("test", f)

