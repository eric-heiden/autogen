#include <gtest/gtest.h>

#include "autogen/autogen.hpp"

constexpr double PI = 3.1415926535;
constexpr double PI_2 = PI / 2.0;

const bool use_reverse_mode = true;

template <typename Scalar>
Scalar simple_c(const std::vector<Scalar> &input) {
  return cos(input[0] * input[1] * 5.23587172);
}

template <typename Scalar>
void simple_b(const std::vector<Scalar> &input, std::vector<Scalar> &output) {
  std::vector<Scalar> temp(3);
  temp[0] = sin(input[0] * PI + 0.7) * PI_2 * input[1] * input[2];
  temp[1] = input[1] * (input[2] + PI_2) * PI;
  temp[2] = input[1] * input[2] * input[2] * input[2] * PI;
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

int dim = 4;
int output_dim = use_reverse_mode ? 2 : 4;
std::vector<double> input = {0.84018771715470952, 0.39438292681909304,
                             0.78309922375860586, 0.79844003347607329};

double kEps = 1e-4;

void check_output(const std::vector<double> &output) {
  EXPECT_NEAR(output[0], 5.87523, kEps);
  EXPECT_NEAR(output[1], 4.2241, kEps);
}
void check_jacobian(const std::vector<double> &jacobian) {
  EXPECT_NEAR(jacobian[0], -64.4636, kEps);
  EXPECT_NEAR(jacobian[1], -13.3565, kEps);
  EXPECT_NEAR(jacobian[2], -3.99235, kEps);
  EXPECT_NEAR(jacobian[3], 0.0, kEps);
  EXPECT_NEAR(jacobian[4], -69.5047, kEps);
  EXPECT_NEAR(jacobian[5], -10.9902, kEps);
  EXPECT_NEAR(jacobian[6], -3.99235, kEps);
  EXPECT_NEAR(jacobian[7], 0.0, kEps);
}

TEST(AutogenTests, FiniteDifferencing) {
  autogen::Generated<simple_a> gen("simple_a");
  std::vector<double> jacobian;
  std::vector<double> output(output_dim);

  gen.set_mode(autogen::GENERATE_NONE);
  gen(input, output);
  check_output(output);
  gen.jacobian(input, jacobian);
  check_jacobian(jacobian);
}

TEST(AutogenTests, CppAD) {
  autogen::Generated<simple_a> gen("simple_a");
  std::vector<double> jacobian;
  std::vector<double> output(output_dim);

  gen.set_mode(autogen::GENERATE_CPPAD);
  gen(input, output);
  check_output(output);
  gen.jacobian(input, jacobian);
  check_jacobian(jacobian);
}

TEST(AutogenTests, CodeGenCPU) {
  autogen::Generated<simple_a> gen("simple_a");
  std::vector<double> jacobian;
  std::vector<double> output(output_dim);

  gen.set_mode(autogen::GENERATE_CPU);
  gen(input, output);
  check_output(output);
  gen.jacobian(input, jacobian);
  check_jacobian(jacobian);
}