#pragma once

// clang-format off
#include <vector>
#include <functional>
#include "base.hpp"
// clang-format on

namespace autogen {
class GeneratedNumerical : public GeneratedBase {
 public:
  using Functor = typename std::function<void(const std::vector<BaseScalar> &,
                                              std::vector<BaseScalar> &)>;
  using ADScalar = double;

 private:
  Functor functor_;

 protected:
  using GeneratedBase::global_input_dim_;
  using GeneratedBase::local_input_dim_;
  using GeneratedBase::output_dim_;

 public:
  /**
   * Step size to use for finite differencing.
   */
  double finite_diff_eps{1e-6};

  GeneratedNumerical(Functor functor) : functor_(functor) {}

  void operator()(const std::vector<BaseScalar> &input,
                  std::vector<BaseScalar> &output) override {
    if (local_input_dim_ < 0) {
      local_input_dim_ = static_cast<int>(input.size());
    }
    functor_(input, output);
    if (output_dim_ < 0) {
      output_dim_ = static_cast<int>(output.size());
    }
  }

  void operator()(const std::vector<std::vector<BaseScalar>> &local_inputs,
                  std::vector<std::vector<BaseScalar>> &outputs,
                  const std::vector<BaseScalar> &global_input = {}) override {
    if (local_inputs.empty()) {
      return;
    }

    if (global_input.empty()) {
      for (size_t i = 0; i < local_inputs.size(); ++i) {
        functor_(local_inputs[i], outputs[i]);
      }
    } else {
      std::vector<BaseScalar> input(global_input);
      input.resize(global_input.size() + local_inputs[0].size());
      for (size_t i = 0; i < local_inputs.size(); ++i) {
        for (size_t j = 0; j < local_inputs[i].size(); ++j) {
          input[j + global_input.size()] = local_inputs[i][j];
        }
        functor_(input, outputs[i]);
      }
    }
  }

  void jacobian(const std::vector<BaseScalar> &input,
                std::vector<BaseScalar> &output) override {
    // central difference
    assert(output_dim() > 0);
    output.resize(input_dim() * output_dim());
    std::vector<BaseScalar> left_x = input, right_x = input;
    std::vector<BaseScalar> left_y(output_dim()), right_y(output_dim());
    for (size_t i = 0; i < input.size(); ++i) {
      left_x[i] -= finite_diff_eps;
      right_x[i] += finite_diff_eps;
      BaseScalar dx = right_x[i] - left_x[i];
      functor_(left_x, left_y);
      functor_(right_x, right_y);
      for (size_t j = 0; j < output_dim(); ++j) {
        output[j * input_dim() + i] = (right_y[j] - left_y[j]) / dx;
      }
      left_x[i] = right_x[i] = input[i];
    }
  }

  void jacobian(const std::vector<std::vector<BaseScalar>> &local_inputs,
                std::vector<std::vector<BaseScalar>> &outputs,
                const std::vector<BaseScalar> &global_input = {}) override {
    outputs.resize(local_inputs.size());

    if (global_input.empty()) {
      for (size_t i = 0; i < local_inputs.size(); ++i) {
        jacobian(local_inputs[i], outputs[i]);
      }
    } else {
      std::vector<BaseScalar> input(global_input.size());
      input.insert(input.begin(), global_input.begin(), global_input.end());
      for (size_t i = 0; i < local_inputs.size(); ++i) {
        for (size_t j = 0; j < local_inputs[i].size(); ++i) {
          input[j + global_input.size()] = local_inputs[i][j];
        }
        jacobian(input, outputs[i]);
      }
    }
    return;
  }
};
}  // namespace autogen