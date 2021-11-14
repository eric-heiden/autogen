# Reference

## Generated

The `Generated` object represents the front-end class that supports different modes of computing gradients for a provided functor. The modes are: Numerical (finite  differencing), CppAD (forward-/reverse-mode AD tracing), and CodeGen (forward-/reverse-mode AD tracing + code generation for various parallel computation platforms).

=== "C++"

    ``` c++
    template <template <typename> typename Functor>
    struct Generated
    ```

    In C++, the function to be differentiated/traced needs to be given as a template parameter for a functor type that accepts the `Scalar` type as template parameter, and implements the `operator()` function:
    ``` c++
    void operator()(const std::vector<Scalar>& input,
                    std::vector<Scalar>& output) const
    ```

    ### Constructor

    ``` c++
    template <typename... Args>
    Generated(const std::string& name, Args&&... args)
    ```

    The constructor takes the name of the functor and the (optional) arguments to be passed to the constructor of the functor.


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