from typing import Callable
from _autogen import *
from enum import Enum

# keeps track of the current autogen mode and other meta information
__AUTOGEN_SCOPE__ = None


class Mode(Enum):
    NUMERICAL = 1
    CPPAD = 2
    CODEGEN = 3


class Scope:
    def __init__(self, mode: Mode):
        self.mode = mode


def scalar_type():
    global __AUTOGEN_SCOPE__
    if __AUTOGEN_SCOPE__ is None or __AUTOGEN_SCOPE__.mode == Mode.NUMERICAL:
        return float
    if __AUTOGEN_SCOPE__.mode == Mode.CPPAD:
        return ADScalar
    return ADCGScalar


def scalar_name():
    global __AUTOGEN_SCOPE__
    if __AUTOGEN_SCOPE__ is None or __AUTOGEN_SCOPE__.mode == Mode.NUMERICAL:
        return "double"
    if __AUTOGEN_SCOPE__.mode == Mode.CPPAD:
        return "AD"
    return "ADCG"


def scalar(x):
    global __AUTOGEN_SCOPE__
    if __AUTOGEN_SCOPE__ is None or __AUTOGEN_SCOPE__.mode == Mode.NUMERICAL:
        return float(x)
    if __AUTOGEN_SCOPE__.mode == Mode.CPPAD:
        return ADScalar(x)
    return ADCGScalar(x)


def vector_type():
    global __AUTOGEN_SCOPE__
    if __AUTOGEN_SCOPE__ is None or __AUTOGEN_SCOPE__.mode == Mode.NUMERICAL:
        return list
    if __AUTOGEN_SCOPE__.mode == Mode.CPPAD:
        return ADVector
    return ADCGVector


def call_atomic(name: str, function, input):
    if __AUTOGEN_SCOPE__.mode != Mode.CODEGEN:
        return function(input)

    if not CodeGenData.has_trace(name):
        CodeGenData.update_call_hierarchy(name)

        ad_x = ADCGVector([ADCGScalar(to_double(x)) for x in input])
        independent(ad_x)
        ys = function(ad_x)
        ad_y = ADCGVector([ADCGScalar(y) for y in ys])

        tape = ADCGFun(ad_x, ad_y)
        CodeGenData.register_trace(name, tape)
        return ad_y

    return CodeGenData.call_bridge(name, ADCGVector(input))


def trace(fun, xs, mode: Mode = Mode.CPPAD):
    global __AUTOGEN_SCOPE__
    if __AUTOGEN_SCOPE__ is None:
        __AUTOGEN_SCOPE__ = Scope(mode)
    else:
        __AUTOGEN_SCOPE__.mode = mode

    Scalar = scalar_type()
    Vector = vector_type()

    CodeGenData.clear()

    print("Dry run...")
    # first, a "dry run" to discover the atomic functions
    CodeGenData.set_dry_run(True)

    ad_x = Vector([Scalar(x) for x in xs])
    independent(ad_x)
    # for i in range(len(xs)):
        # xs[i] = ad_x[i]
    ys = fun(ad_x)

    CodeGenData.set_dry_run(False)

    print('The following atomic functions were discovered: [%s]' % ', '.join(CodeGenData.invocation_order))

    
    print("Final run...")
    # trace top-level function where the CGAtomicFunBridges are used
    ad_x = Vector([Scalar(x) for x in xs])
    independent(ad_x)
    # for i in range(len(xs)):
        # xs[i] = ad_x[i]
    ys = fun(ad_x)

    ad_y = Vector([Scalar(y) for y in ys])
    if mode == Mode.NUMERICAL:
        raise NotImplementedError("finite diff functor not yet implemented")
    if mode == Mode.CPPAD:
        return ADFun(ad_x, ad_y)
    return ADCGFun(ad_x, ad_y)


class Generated:
    def __init__(
        self, function: Callable[[list], list], name: str, mode: Mode = Mode.CPPAD
    ):
        self.function = function
        self.name = name
        self.mode = mode
        self.__global_input_dim = 0
        self.__local_input_dim = -1
        self.__output_dim = -1
        self.__is_compiled = False

    def __call__(self, x: list) -> list:
        return self.function(x)

    def jacobian(self, x: list) -> list:
        pass

    @property
    def mode(self):
        return self.mode

    @mode.setter
    def mode(self, m):
        if m == self.mode:
            return
        print("Setting mode to", m)
        self.mode = m

    @property
    def input_dim(self):
        return self.__global_input_dim + self.__local_input_dim

    @property
    def output_dim(self):
        return self.__output_dim

    def discard_library(self):
        pass

    @property
    def is_compiled(self):
        return self.__is_compiled


# def trace(fun, xs) -> ADFun:
#     ad_x = ADVector([ADScalar(x) for x in xs])
#     independent(ad_x)
#     for i in range(len(xs)):
#         xs[i] = ad_x[i]
#     ys = fun(xs)
#     ad_y = ADVector(list(ys))
#     f = ADFun(ad_x, ad_y)
#     return f


# def trace_cg(fun, xs) -> ADCGFun:
#     ad_x = CGVector([ADCGScalar(x) for x in xs])
#     independent(ad_x)
#     for i in range(len(xs)):
#         xs[i] = ad_x[i]
#     ys = fun(xs)
#     ad_y = CGVector(list(ys))
#     f = ADCGFun(ad_x, ad_y)
#     return f
