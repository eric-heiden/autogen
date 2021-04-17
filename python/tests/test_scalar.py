import numpy as np
import math
from autogen_python import CppADScalar

# basic operator test
print (CppADScalar(1.0), -CppADScalar(1.0))
assert CppADScalar(1.0) == CppADScalar(1.0)
assert -CppADScalar(1.0) == CppADScalar(-1.0)
# assert CppADScalar(2.0)**2 == CppADScalar(4.0)
assert CppADScalar(1.0) * CppADScalar(5.0) == CppADScalar(10.0) / CppADScalar(2.0)
assert CppADScalar(1.0) + CppADScalar(5.0) == CppADScalar(10.0) - CppADScalar(4.0)

# sin test
arr = np.array([CppADScalar(0), CppADScalar(math.pi / 2)], dtype=CppADScalar)
sin_arr = np.sin(arr)
assert sin_arr[0] == CppADScalar(0)
assert sin_arr[1] == CppADScalar(1.0)

# Array operator with float test
arr = np.array([CppADScalar(1), CppADScalar(2)], dtype=CppADScalar)
arr = 3 * arr / 2.
assert arr[0] == CppADScalar(1.5)
assert arr[1] == CppADScalar(3.0)

print(arr)