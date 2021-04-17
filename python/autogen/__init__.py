from _autogen import *


def trace(fun, xs) -> Function:
    ad_x = Vector([Scalar(x) for x in xs])
    independent(ad_x)
    for i in range(len(xs)):
        xs[i] = ad_x[i]
    ys = fun(xs)
    ad_y = Vector(list(ys))
    f = Function(ad_x, ad_y)
    return f
