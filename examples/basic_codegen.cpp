#include <iostream>

#include "autogen/autogen.hpp"

constexpr double pi = 3.1415926535;
constexpr double pi_2 = pi / 2.0;

const bool use_reverse_mode = true;

template <typename Scalar>
Scalar simple_c(const std::vector<Scalar> &input) {
  return cos(input[0] * input[1] * 5.23587172);
}

template <typename Scalar>
void simple_b(const std::vector<Scalar> &input, std::vector<Scalar> &output) {
  std::vector<Scalar> temp(3);
  temp[0] = sin(input[0] * pi + 0.7) * pi_2 * input[1] * input[2];
  temp[1] = input[1] * (input[2] + pi_2) * pi;
  temp[2] = input[1] * input[2] * input[2] * input[2] * pi;
  // note we use a special `call_atomic` overload for scalar-valued functions
  std::function functor = &simple_c<Scalar>;
  output[0] = temp[0] + autogen::call_atomic("cosine", functor, temp);
  output[1] = temp[1] + autogen::call_atomic("cosine", functor, temp);
  output[2] = temp[2] + autogen::call_atomic("cosine", functor, temp);
}

template <typename Scalar>
struct simple_a {
  void operator()(const std::vector<Scalar> &input,
                  std::vector<Scalar> &output) const {
    for (size_t i = 0; i < output.size(); ++i) {
      output[i] = input[i] * input[i] * 3.0;
      std::vector<Scalar> inputs = {input[0], input[1], input[2]};
      std::vector<Scalar> temp(3);
      std::function functor = &simple_b<Scalar>;
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
  int output_dim = use_reverse_mode ? 2 : 4;
  std::vector<double> input(dim), output(output_dim);
  input = {0.84018771715470952, 0.39438292681909304, 0.78309922375860586,
           0.79844003347607329};
  std::cout << "\nInput:  ";
  for (int i = 0; i < dim; ++i) {
    //  input[i] = double(rand()) / RAND_MAX;
    std::cout << input[i] << "  ";
  }
  std::cout << std::endl << std::endl;

  autogen::Generated<simple_a> gen("simple_a");
  std::vector<double> jacobian;
  std::vector<std::vector<double>> outputs(1);

  gen.set_mode(autogen::MODE_NUMERICAL);
  std::cout << "### Mode: " << gen.mode() << std::endl;
  gen(input, output);
  print(output);
  gen.jacobian(input, jacobian);
  print(jacobian);

  gen.set_mode(autogen::MODE_CPPAD);
  std::cout << "### Mode: " << gen.mode() << std::endl;
  gen(input, output);
  print(output);
  gen.jacobian(input, jacobian);
  print(jacobian);

  // try {
  //   gen.set_codegen_target(autogen::TARGET_OPENMP);
  //   std::cout << "### Mode: " << gen.mode()
  //             << "  Target: " << gen.codegen_target() << std::endl;
  //   gen(input, output);
  //   print(output);
  //   gen.jacobian(input, jacobian);
  //   print(jacobian);
  // } catch (const std::exception &ex) {
  //   std::cerr << "Error: " << ex.what() << std::endl;
  // }
  // gen.target()->create_cmake_project(
  //     "simple_a_" + std::to_string(gen.codegen_target()) + "_cmake", input);

  try {
    gen.set_codegen_target(autogen::TARGET_CUDA);
    std::cout << "### Mode: " << gen.mode()
              << "  Target: " << gen.codegen_target() << std::endl;
    gen(input, output);
    print(output);
    gen.jacobian(input, jacobian);
    print(jacobian);
  } catch (const std::exception &ex) {
    std::cerr << "Error: " << ex.what() << std::endl;
  }
  gen.target()->create_cmake_project(
      "simple_a_" + std::to_string(gen.codegen_target()) + "_cmake", input);

  try {
    gen.set_codegen_target(autogen::TARGET_LEGACY_C);
    std::cout << "### Mode: " << gen.mode()
              << "  Target: " << gen.codegen_target() << std::endl;
    gen(input, output);
    print(output);
    gen.jacobian(input, jacobian);
    print(jacobian);
  } catch (const std::exception &ex) {
    std::cerr << "Error: " << ex.what() << std::endl;
  }
  gen.target()->create_cmake_project(
      "simple_a_" + std::to_string(gen.codegen_target()) + "_cmake", input);

  // gen.set_mode(autogen::GENERATE_CPU);
  // std::cout << "### Mode: " << gen.mode() << std::endl;
  // gen(input, output);
  // print(output);
  // gen.jacobian(input, jacobian);
  // print(jacobian);

  return EXIT_SUCCESS;
}