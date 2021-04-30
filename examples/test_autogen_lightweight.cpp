#include "autogen/autogen_lightweight.hpp"

namespace {
void print(const std::vector<double>& vs) {
  for (std::size_t i = 0; i < vs.size(); ++i) {
    std::cout << vs[i];
    if (i < vs.size() - 1) std::cout << ", ";
  }
  std::cout << std::endl;
}

template <typename Scalar>
void test_function(const std::vector<Scalar>& input,
                   std::vector<Scalar>& output) {
  output.clear();
  output.resize(input.size());
  for (size_t i = 0; i < input.size(); i++) {
    output[i] = input[i] * 2;
  }
}
}  // namespace

int main(int argc, char* argv[]) {
  autogen::GeneratedCppAD<double> gen("test", &test_function);

  std::vector<double> jacobian;
  std::vector<double> input = {1.0};
  std::vector<double> output(1);

  gen(input, output);
  print(output);

  gen.jacobian(input, jacobian);
  print(jacobian);

  return EXIT_SUCCESS;
}