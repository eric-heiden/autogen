#include "autogen/autogen.hpp"

constexpr double PI = 3.1415926535;
constexpr double PI_2 = PI / 2.0;

const size_t kTimesteps = 10;
const size_t kParamDim = 6;
const size_t kStateDim = 4;  // q1, q2, qd1, qd2
const size_t kGlobalInputDim = kTimesteps * kStateDim;
const size_t kTotalInputDim = kGlobalInputDim + kParamDim + kStateDim + 1;

template <typename Scalar>
void printv(const std::vector<Scalar> &vs) {
  for (std::size_t i = 0; i < vs.size(); ++i) {
    std::cout << vs[i];
    if (i < vs.size() - 1) std::cout << ", ";
  }
  std::cout << std::endl;
}
template <typename Scalar>
void printv(const std::vector<std::vector<Scalar>> &vs) {
  for (const auto &v : vs) {
    printv(v);
  }
}

template <typename Scalar>
void pendulum_dynamics(const Scalar &q1, const Scalar &q2, const Scalar &qd1,
                       const Scalar &qd2, const Scalar &m1, const Scalar &m2,
                       const Scalar &l1, const Scalar &l2, const Scalar &g,
                       Scalar &qdd1, Scalar &qdd2) {
  using std::sin, std::cos;

  Scalar s12 = sin(q1 - q2);
  Scalar c12 = cos(q1 - q2);
  Scalar denom = 2.0 * m1 + m2 - m2 * cos(2.0 * (q1 - q2));

  qdd1 = -g * (2.0 * m1 + m2) * sin(q1) - m2 * g * sin(q1 - 2.0 * q2) -
         2.0 * m2 * qd2 * qd2 * l2 * s12 -
         m2 * qd1 * qd1 * l1 * sin(2.0 * (q1 - q2));
  qdd1 = qdd1 / (l1 * denom);

  qdd2 = 2.0 * s12 *
         (qd1 * qd1 * l1 * (m1 + m2) + g * (m1 + m2) * cos(q1) +
          qd2 * qd2 * l2 * m2 * c12);
  qdd2 = qdd2 / (l2 * denom);
}

template <typename Scalar>
void rollout_groundtruth(const Scalar &m1, const Scalar &m2, const Scalar &l1,
                         const Scalar &l2, const Scalar &g, const Scalar &dt,
                         std::vector<Scalar> &trajectory) {
  Scalar q1 = PI_2;
  Scalar q2 = 0.0;
  Scalar qd1 = 0.0;
  Scalar qd2 = 0.0;
  Scalar qdd1, qdd2;
  trajectory.resize(kTimesteps * 4);
  for (size_t t = 0; t < kTimesteps; ++t) {
    pendulum_dynamics(q1, q2, qd1, qd2, m1, m2, l1, l2, g, qdd1, qdd2);
    qd1 += dt * qdd1;
    qd2 += dt * qdd2;
    q1 += dt * qd1;
    q2 += dt * qd2;
    trajectory[4 * t + 0] = q1;
    trajectory[4 * t + 1] = q2;
    trajectory[4 * t + 2] = qd1;
    trajectory[4 * t + 3] = qd2;
  }
}

template <typename Scalar>
Scalar l2norm(const Scalar &a, const Scalar &b) {
  Scalar d = a - b;
  return d * d;
}

template <typename Scalar>
void step(const std::vector<Scalar> &input, std::vector<Scalar> &output) {
  std::cout << "input: ";
  printv(input);
  // 5 dynamic inputs = output dimension
  size_t i = 0;
  Scalar q1 = input[i++];
  Scalar q2 = input[i++];
  Scalar qd1 = input[i++];
  Scalar qd2 = input[i++];
  const Scalar &l = input[i++];
  Scalar qdd1, qdd2;
  // 6 parameters ("const_input")
  const Scalar &m1 = input[i++];
  const Scalar &m2 = input[i++];
  const Scalar &l1 = input[i++];
  const Scalar &l2 = input[i++];
  const Scalar &g = input[i++];
  const Scalar &dt = input[i++];
  std::cout << "m1: " << m1 << "  "
            << "m2: " << m2 << "  "
            << "l1: " << l1 << "  "
            << "l2: " << l2 << "  "
            << "g: " << g << "  "
            << "dt: " << dt << "\n";
  // 4 time-dependent inputs
  const Scalar ref_q1 = input[i++];
  const Scalar ref_q2 = input[i++];
  const Scalar ref_qd1 = input[i++];
  const Scalar ref_qd2 = input[i++];
  std::cout << "ref_q1: " << ref_q1 << "    "
            << "ref_q2: " << ref_q2 << "    "
            << "ref_qd1: " << ref_qd1 << "    "
            << "ref_qd2: " << ref_qd2 << "\n";

  pendulum_dynamics(q1, q2, qd1, qd2, m1, m2, l1, l2, g, qdd1, qdd2);
  qd1 += dt * qdd1;
  qd2 += dt * qdd2;
  q1 += dt * qd1;
  q2 += dt * qd2;

  // update loss
  Scalar loss = l;  // loss from previous time step
  loss += l2norm(q1, ref_q1);
  loss += l2norm(q2, ref_q2);
  loss += l2norm(qd1, ref_qd1);
  loss += l2norm(qd2, ref_qd2);
  // update state
  output[0] = q1;
  output[1] = q2;
  output[2] = qd1;
  output[3] = qd2;
  output[4] = loss;
  std::cout << "output: ";
  printv(output);
  std::cout << "\n";
}

template <typename Scalar>
struct cost {
  void operator()(const std::vector<Scalar> &input,
                  std::vector<Scalar> &output) const {
    std::function functor = &step<Scalar>;
    std::vector<Scalar> in(kTotalInputDim);
    size_t i = 0;
    // initial state
    in[i++] = PI_2;  //  q1
    in[i++] = 0.0;   //  q2
    in[i++] = 0.0;   //  qd1
    in[i++] = 0.0;   //  qd2
    in[i++] = 0.0;   // initial cost
    const size_t ref_traj_dim = kTimesteps * kStateDim;
    // system parameters
    for (size_t j = 0; j < kParamDim; ++j) {
      in[i++] = input[ref_traj_dim + j];
    }
    // reference trajectory (global input)
    for (size_t j = 0; j < ref_traj_dim; ++j) {
      in[i++] = input[j];
    }
    std::cout << "complete input:  ";
    printv(in);
    autogen::call_atomic(std::string("step"), functor, in, output, kTimesteps,
                         kParamDim, kStateDim);
  }
};

int main(int argc, char *argv[]) {
  // true parameters
  double m1 = 0.3;
  double m2 = 0.1;
  double l1 = 0.1;
  double l2 = 0.1;
  double g = -9.81;
  const double dt = 0.01;
  std::vector<double> ref_trajectory;
  rollout_groundtruth(m1, m2, l1, l2, g, dt, ref_trajectory);
  std::cout << "Ground-truth trajectory:\n";
  printv(ref_trajectory);

  // std::cout << "\nInput:  ";
  // for (int i = 0; i < dim; ++i) {
  //   //  input[i] = double(rand()) / RAND_MAX;
  //   std::cout << input[i] << "  ";
  // }
  // std::cout << std::endl << std::endl;

  autogen::Generated<cost> gen("cost");
  // gen.set_global_input_dim(kParamDim);
  // gen.debug_mode = true;
  gen.set_mode(autogen::GENERATE_CUDA);
  // gen.set_mode(autogen::GENERATE_NONE);
  std::vector<double> jacobian;

  std::vector<std::vector<double>> outputs(1);
  outputs[0].resize(kStateDim + 1);
  std::vector<double> global_input = ref_trajectory;
  std::vector<std::vector<double>> local_inputs(1);
  local_inputs[0].resize(kParamDim);
  size_t k = 0;
  local_inputs[0][k++] = 0.3;    //  m1
  local_inputs[0][k++] = 0.1;    //  m2
  local_inputs[0][k++] = 0.1;    //  l1
  local_inputs[0][k++] = 0.1;    //  l2
  local_inputs[0][k++] = -9.81;  //  g
  local_inputs[0][k++] = 0.01;   //  dt

  gen(local_inputs, outputs, global_input);
  printv(outputs);

  std::cout << "Double eval:\n";
  gen.set_mode(autogen::GENERATE_NONE);
  gen(local_inputs, outputs, global_input);
  printv(outputs);

  std::cout << "Input:\n";
  printv(local_inputs);

  // // try {
  // std::cout << "### Mode: " << gen.mode() << std::endl;
  // for (int i = 0; i < 1; ++i) {
  //   gen(input, output);
  //   print(output);

  //   gen.jacobian(input, jacobian);
  //   print(jacobian);
  // }

  // // outputs[0].resize(dim);
  // // gen({input}, outputs);
  // // print(outputs[0]);
  // // } catch (const std::exception &e) {
  // //   std::cerr << e.what() << std::endl;
  // // }

  // gen.set_mode(autogen::GENERATE_NONE);
  // std::cout << "### Mode: " << gen.mode() << std::endl;
  // gen(input, output);
  // print(output);
  // gen.jacobian(input, jacobian);
  // print(jacobian);

  // gen.set_mode(autogen::GENERATE_CPPAD);
  // std::cout << "### Mode: " << gen.mode() << std::endl;
  // gen(input, output);
  // print(output);
  // gen.jacobian(input, jacobian);
  // print(jacobian);

  // gen.set_mode(autogen::GENERATE_CPU);
  // std::cout << "### Mode: " << gen.mode() << std::endl;
  // gen(input, output);
  // print(output);
  // gen.jacobian(input, jacobian);
  // print(jacobian);

  return EXIT_SUCCESS;
}