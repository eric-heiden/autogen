from _autogen import *


def trace(fun, xs) -> CppADFunction:
    ad_x = CppADVector([CppADScalar(x) for x in xs])
    independent(ad_x)
    for i in range(len(xs)):
        xs[i] = ad_x[i]
    ys = fun(xs)
    ad_y = CppADVector(list(ys))
    f = CppADFunction(ad_x, ad_y)
    return f
