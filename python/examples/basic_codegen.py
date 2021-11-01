import autogen as ag
import math
import numpy as np

use_reverse_mode = True


def simple_c(input):
    return np.cos(input[0] * input[1] * 5.23587172)


def simple_b(input):
    temp = np.zeros(3, dtype=ag.scalar_type())
    temp[0] = np.sin(input[0] * math.pi + 0.7) * \
        math.pi / 2 * input[1] * input[2]
    temp[1] = input[1] * (input[2] + math.pi / 2) * math.pi
    temp[2] = input[1] * input[2] * input[2] * input[2] * math.pi

    output = np.zeros(3, dtype=ag.scalar_type())
    output[0] = temp[0] + ag.call_atomic("cosine", simple_c, temp)
    output[1] = temp[1] + ag.call_atomic("cosine", simple_c, temp)
    output[2] = temp[2] + ag.call_atomic("cosine", simple_c, temp)
    return output


def simple_a(input):
    output_dim = 2 if use_reverse_mode else 4
    output = np.zeros(output_dim, dtype=ag.scalar_type())
    for i in range(output_dim):
        output[i] = input[i] * input[i] * 3.0
        inputs = [*input[:3]]
        temp = ag.call_atomic("sine", simple_b, inputs)
        output[i] += temp[0] + temp[1] + temp[2]
    return output


input = [0.84018771715470952, 0.39438292681909304, 0.78309922375860586,
         0.79844003347607329]

tape = ag.trace(simple_a, input, ag.Mode.CPPAD)
fun = ag.GeneratedCppAD(tape)
print('### Mode:', ag.get_mode())
print(fun(input))
print(fun.jacobian(input))

tape = ag.trace(simple_a, input, ag.Mode.CODEGEN)
fun = ag.GeneratedCodeGen(tape)
fun.compile_cpu()
print('### Mode:', ag.get_mode())
print(fun(input))
print(fun.jacobian(input))
