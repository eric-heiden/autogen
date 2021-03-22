#include "autogen/autogen.hpp"

template <typename Scalar>
Scalar simple_c(const std::vector<Scalar> &input) {
  return cos(input[0] * input[1] * 5.0);
}

template <typename Scalar>
void simple_b(const std::vector<Scalar> &input, std::vector<Scalar> &output) {
  output[0] = sin(input[0] * 2.0 + 0.7) * 5.0 * input[1] * input[2];
  output[1] = input[1] * input[2];
  output[2] = input[1] * input[2] * input[2] * input[2];
  // note we use a special `call_atomic` overload for scalar-valued functions
  std::function<Scalar(const std::vector<Scalar> &)> functor =
      &simple_c<Scalar>;
  for (auto &o : output) o += autogen::call_atomic("cosine", functor, input);
}

template <typename Scalar>
struct simple_a {
  void operator()(const std::vector<Scalar> &input,
                  std::vector<Scalar> &output) const {
    for (size_t i = 0; i < output.size(); ++i) {
      output[i] = input[i] * input[i] * 3.0;
      std::vector<Scalar> inputs = {input[0], input[1], input[2]};
      std::vector<Scalar> temp(3);
      std::function<void(const std::vector<Scalar> &, std::vector<Scalar> &)>
          functor = &simple_b<Scalar>;
      autogen::call_atomic(std::string("sine"), functor, inputs, temp);
      output[i] += temp[0] + temp[1] + temp[2];
    }
  }
};

void print(const std::vector<double> &vs) {
  for (std::size_t i = 0; i < vs.size(); ++i) {
    std::cout << vs[i];
    if (i < vs.size() - 1) std::cout << ", ";
  }
  std::cout << std::endl;
}
void print(const std::vector<std::vector<double>> &vs) {
  for (const auto &v : vs) {
    print(v);
  }
}

int main(int argc, char *argv[]) {
  int dim = 4;
  std::vector<double> input(dim), output(dim);
  for (int i = 0; i < dim; ++i) {
    input[i] = double(rand()) / RAND_MAX;
  }

  autogen::Generated<simple_a> gen("simple_a");
  gen.set_mode(autogen::GENERATE_CUDA);
  // gen.set_mode(autogen::GENERATE_NONE);
  std::vector<double> jacobian;
  std::vector<std::vector<double>> outputs(1);

  // try {
    std::cout << "### Mode: " << gen.mode() << std::endl;
    for (int i = 0; i < 2; ++i) {
      gen(input, output);
      print(output);

      gen.jacobian(input, jacobian);
      print(jacobian);
    }

    outputs[0].resize(dim);
    gen({input}, outputs);
    print(outputs[0]);
  // } catch (const std::exception &e) {
  //   std::cerr << e.what() << std::endl;
  // }

  gen.set_mode(autogen::GENERATE_NONE);
  std::cout << "### Mode: " << gen.mode() << std::endl;
  gen(input, output);
  print(output);
  gen.jacobian(input, jacobian);
  print(jacobian);

  gen.set_mode(autogen::GENERATE_CPPAD);
  std::cout << "### Mode: " << gen.mode() << std::endl;
  gen(input, output);
  print(output);
  gen.jacobian(input, jacobian);
  print(jacobian);

  gen.set_mode(autogen::GENERATE_CPU);
  std::cout << "### Mode: " << gen.mode() << std::endl;
  gen(input, output);
  print(output);
  gen.jacobian(input, jacobian);
  print(jacobian);

  return EXIT_SUCCESS;
}