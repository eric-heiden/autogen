#include <cppad/cg.hpp>
#include <cppad/cg/arithmetic.hpp>

#include "autogen/autogen_lightweight.hpp"

namespace {
using CGScalar = typename CppAD::cg::CG<double>;
using ADFun = typename CppAD::ADFun<CGScalar>;

void print(const std::vector<double>& vs) {
  for (std::size_t i = 0; i < vs.size(); ++i) {
    std::cout << vs[i];
    if (i < vs.size() - 1) std::cout << ", ";
  }
  std::cout << std::endl;
}

template <typename BaseScalar>
void test_function(const std::vector<BaseScalar>& input,
                   std::vector<BaseScalar>& output) {
  output.clear();
  output.resize(input.size());
  for (size_t i = 0; i < input.size(); i++) {
    output[i] = input[i] * 2.0;
  }
}
}  // namespace

int main(int argc, char* argv[]) {
  std::vector<double> input = {1.0, 2.0};
  std::vector<CppAD::AD<CGScalar>> ax(input.size()), ay(input.size());
  for (size_t i = 0; i < input.size(); ++i) {
    ax[i] = CGScalar(input[i]);
  }
  CppAD::Independent(ax);
  test_function<CppAD::AD<CGScalar>>(ax, ay);
  std::shared_ptr<ADFun> fun = std::make_shared<ADFun>(ax, ay);
//  ADFun fun(ax, ay);
  autogen::GeneratedLightWeight<double> gen("test", fun);

  std::vector<double> test_jacobian;
  std::vector<double> test_input = {1.0, 2.0};
  std::vector<double> test_output(2);

//  print(fun->Forward(0, test_input));

  gen(test_input, test_output);
  print(test_output);

  gen.jacobian(test_input, test_jacobian);
  print(test_jacobian);

  return EXIT_SUCCESS;
}