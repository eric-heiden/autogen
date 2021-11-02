# Getting started

Autogen supports numerical code written in C++ or Python to be traced, compiled, and executed in parallel on the CPU and GPU.

The following example demonstrates a simple function that computes the norm of a 2D vector:

=== "C++"

    ``` c++
    #include <autogen/autogen.hpp>

    template <typename Scalar>  // Scalar denotes the number type
    struct my_function {
        void operator()(const std::vector<Scalar> &input,
                        std::vector<Scalar> &output) const {
            output[0] = sqrt(input[0] * input[0] + input[1] * input[1]);
        }
    };

    int main(void) {
        std::vector<double> input, output(1);
        input = {0.5, 1.5};
        autogen::Generated<my_function> gen("my_function");
        gen.set_mode(autogen::GENERATE_CUDA);
        // the function gets compiled and executed on the GPU
        gen(input, output);
        return 0;
    }
    ```


=== "Python"

    ``` python
    import autogen as ag
    import numpy as np

    def my_function(xs):
        return np.sqrt(xs[0]**2 + xs[1]**2)

    def __main__():
        xs = np.array([0.5, 1.5])
        gen = ag.Generated("my_function", my_function)
        gen.set_mode(ag.GENERATE_CUDA)
        y = gen(xs)
        print(y)
    ```