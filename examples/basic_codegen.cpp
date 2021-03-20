#include "autogen/autogen.hpp"

template <typename Scalar>
void simple_c(const std::vector<Scalar> &input, std::vector<Scalar> &output) {
  output[0] = cos(input[0] * 5.0);
}

template <typename Scalar>
void simple_b(const std::vector<Scalar> &input, std::vector<Scalar> &output) {
  // std::vector<Scalar> temp(1);
  // autogen::call_atomic("cosine", &simple_c<Scalar>, input, temp);
  output[0] = sin(input[0] * 2.0 + 0.7) * 5.0;  // + temp[0];
}

template <typename Scalar>
struct simple_a {
  void operator()(const std::vector<Scalar> &input,
                  std::vector<Scalar> &output) const {
    for (size_t i = 0; i < output.size(); ++i) {
      output[i] = input[i] * input[i] * 3.0;
      std::vector<Scalar> temp(1);
      autogen::call_atomic("sine", &simple_b<Scalar>, {input[i]}, temp);
      output[i] += temp[0];
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
  for (int i = 0; i < 5; ++i) {
    gen(input, output);
    print(output);

    gen.jacobian(input, jacobian);
    print(jacobian);
  }

  std::vector<std::vector<double>> outputs(1);
  outputs[0].resize(dim);
  gen({input}, outputs);
  print(outputs[0]);

  return EXIT_SUCCESS;
}