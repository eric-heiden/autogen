from typing import Callable
from collections import namedtuple
import os
import numpy as np
from _autogen import *

init_shared_data()
# print("initialized shared data")


def scalar_type():
    """
    Retrieves the scalar type for the current mode.
    """
    if get_mode() == Mode.DOUBLE:
        return float
    if get_mode() == Mode.CPPAD:
        return ADScalar
    return ADCGScalar


def scalar(x):
    """
    Converts x to the scalar type for the current mode.
    """
    if get_mode() == Mode.DOUBLE:
        return float(to_double(x))
    if get_mode() == Mode.CPPAD:
        return ADScalar(x)
    return ADCGScalar(x)


def vector_type():
    """
    Retrieves the vector type for the current mode.
    """
    if get_mode() == Mode.DOUBLE:
        return list
    if get_mode() == Mode.CPPAD:
        return ADVector
    return ADCGVector


def vector(xs):
    """
    Converts xs to the vector type for the current mode.
    """
    if get_mode() == Mode.DOUBLE:
        return xs
    if get_mode() == Mode.CPPAD:
        return ADVector([ADScalar(x) for x in xs])
    return ADCGVector([ADCGScalar(x) for x in xs])


# data corresponding to atomic functions defined in Python
__scalar_atomics = set()  # names of atomics that return a scalar
__trace_data = {}
__trace_order = []
__TraceData = namedtuple("TraceData", ["function", "input"])
__tapes = []


def call_atomic(name: str, function, input):
    global __scalar_atomics
    global __trace_data
    global __trace_order

    if get_mode() != Mode.CODEGEN:
        return function(input)

    if CodeGenData.is_dry_run():
        if name not in __trace_data:
            CodeGenData.update_call_hierarchy(name)
            __trace_order.insert(0, name)
            raw_input = [to_double(x) for x in input]
            __trace_data[name] = __TraceData(function, raw_input)
            return function(input)

        return function(input)

    if name in __scalar_atomics:
        return CodeGenData.call_bridge(name, ADCGVector(input))[0]
    return CodeGenData.call_bridge(name, ADCGVector(input))


def __trace_python_atomics():
    """
    Trace the atomic functions implemented in Python.
    """
    global __tapes
    for name in __trace_order:
        data = __trace_data[name]
        ad_x = ADCGVector([ADCGScalar(x) for x in data.input])
        independent(ad_x)
        ys = data.function(ad_x)
        if isinstance(ys, ADCGScalar):
            # this atomic function only returns a scalar
            ad_y = ADCGVector([ys])
            __scalar_atomics.add(name)
            ys = ADCGVector([ys])
        else:
            ad_y = ADCGVector([ADCGScalar(y) for y in ys])
        tape = ADCGFun(ad_x, ad_y)
        # add tape to global list to keep it alive
        __tapes.append(tape)
        CodeGenData.register_trace(name, tape)
        print(f'Registered trace for atomic function "{name}".')


def trace(fun, xs, mode: Mode = Mode.CPPAD):
    global __scalar_atomics
    global __trace_data
    global __tapes

    set_mode(mode)

    Scalar = scalar_type()
    Vector = vector_type()

    if mode == Mode.CODEGEN:
        CodeGenData.clear()
        __scalar_atomics.clear()
        __trace_data.clear()
        __tapes.clear()

        # print("Dry run...")
        # first, a "dry run" to discover the atomic functions
        CodeGenData.set_dry_run(True)
        retrieve_tape()

        ad_x = Vector([Scalar(to_double(x)) for x in xs])
        # independent(ad_x)
        ys = fun(ad_x)

        CodeGenData.set_dry_run(False)

        print(
            "The following atomic functions were discovered: [%s]"
            % ", ".join(CodeGenData.invocation_order())
        )

        # trace existing atomics from some non-Python code
        CodeGenData.trace_existing_atomics()
        # trace atomics in Python code
        __trace_python_atomics()

        # print("Final run...")
    # trace top-level function where the CGAtomicFunBridges are used
    ad_x = Vector([Scalar(to_double(x)) for x in xs])
    independent(ad_x)
    ys = fun(ad_x)
    ad_y = Vector(ys)
    # ad_y = Vector([Scalar(y) for y in ys])
    if mode == Mode.DOUBLE:
        raise NotImplementedError("finite diff functor not yet implemented")
    if mode == Mode.CPPAD:
        return ADFun(ad_x, ad_y)
    return ADCGFun(ad_x, ad_y)


def init_vsvars():
    # https://stackoverflow.com/a/57883682
    vswhere_path = r"%ProgramFiles(x86)%/Microsoft Visual Studio/Installer/vswhere.exe"
    vswhere_path = os.path.expandvars(vswhere_path)
    if not os.path.exists(vswhere_path):
        raise EnvironmentError("vswhere.exe not found at: %s", vswhere_path)

    vs_path = (
        os.popen(f'"{vswhere_path}" -latest -property installationPath')
        .read()
        .rstrip()
    )
    vsvars_path = os.path.join(vs_path, "VC\\Auxiliary\\Build\\vcvars64.bat")
    output = os.popen(f'"{vsvars_path}" && set').read()
    assignments = output.splitlines()
    for line in assignments:
        pair = line.split("=", 1)
        if len(pair) >= 2:
            os.environ[pair[0]] = pair[1]
    print(
        f'Loaded {len(assignments)} build environment variables from "{vsvars_path}".'
    )


# if "windows" in platform.system().lower():
#     init_vsvars()


class Generated:
    def __init__(
        self,
        function: Callable[[list], list],
        name: str = None,
        mode: Mode = Mode.CODEGEN,
        codegen_target = Target.OPENMP,
    ):
        self.function = function
        if name is None:
            name = function.__name__
        self.name = name.lower().replace(" ", "_")
        self.__mode = mode
        self.__codegen_target = codegen_target
        self.__gen = None
        self.debug_mode = False
        self.__global_input_dim = 0
        self.__local_input_dim = -1
        self.__output_dim = -1

    def init(self, trace_input):
        f = trace(self.function, trace_input, self.__mode)
        if self.__mode == Mode.CODEGEN:
            self.__gen = GeneratedCodeGen(self.name, f)
            self.__gen.set_target(self.__codegen_target)
            self.__gen.debug_mode = self.debug_mode
            self.__gen.generate_code()
            self.__gen.compile()
        elif self.__mode == Mode.CPPAD:
            self.__gen = GeneratedCppAD(f)
        else:
            raise NotImplementedError(f"Mode {self.__mode} not supported")
        self.__global_input_dim = self.__gen.global_input_dim
        self.__local_input_dim = self.__gen.local_input_dim
        self.__output_dim = self.__gen.output_dim

    def __call__(self, x: list) -> list:
        if self.__gen is None:
            self.init(x)
        return self.__gen(x)

    def jacobian(self, x: list) -> list:
        if self.__gen is None:
            self.init(x)
        return self.__gen.jacobian(x)

    @property
    def mode(self):
        return self.__mode

    @mode.setter
    def mode(self, m):
        if m == self.__mode:
            return
        print("Setting mode to", m)
        self.__mode = m
        self.__gen = None

    @property
    def codegen_target(self):
        return self.__codegen_target

    @codegen_target.setter
    def codegen_target(self, m):
        if m == self.__codegen_target:
            return
        print("Setting codegen target to", m)
        self.__codegen_target = m
        self.__gen = None

    @property
    def input_dim(self):
        return self.__global_input_dim + self.__local_input_dim

    @property
    def output_dim(self):
        return self.__output_dim

    @property
    def is_compiled(self):
        return self.__is_compiled

    def get_torch_functor(self, check_nan=False):
        if self.__gen is None:
            raise RuntimeError("Generated function not initialized")
        import torch
        gen = self.__gen
        name = self.name
        input_dim = self.input_dim
        output_dim = self.output_dim

        class TorchFunctor(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input):
                ctx.batched = input.ndim == 2
                ctx.input = input.detach().cpu().numpy()
                if ctx.batched:
                    output = gen(ctx.input, [])
                else:
                    output = gen(ctx.input)
                if check_nan and np.any(np.isnan(output)):
                    gen.create_cmake_project(f"{name}_cmake", ctx.input)
                    raise RuntimeError(
                        f"Output from forward pass of function {name} contains NaN values"
                    )
                return torch.tensor(output, dtype=input.dtype, device=input.device)

            @staticmethod
            def backward(ctx, grad_output):
                # Here we must handle None grad_output tensor. In this case we
                # can skip unnecessary computations and just return None.
                if grad_output is None:
                    return None, None

                if ctx.batched:
                    jac = gen.jacobian(ctx.input, [])
                else:
                    jac = gen.jacobian(ctx.input)
                jac = np.asarray(jac).reshape(
                    (input_dim, output_dim), order="F")
                if check_nan and np.any(np.isnan(jac)):
                    gen.create_cmake_project(f"{name}_cmake", ctx.input)
                    raise RuntimeError("Dynamics Jacobian contains NaN values")
                jac = torch.tensor(
                    jac, dtype=grad_output.dtype, device=grad_output.device
                )
                if ctx.batched:
                    grad = torch.matmul(jac, grad_output.unsqueeze(1))
                else:
                    grad = torch.matmul(jac, grad_output)

                return grad, None

        torch_functor = TorchFunctor.apply
        return torch_functor


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
