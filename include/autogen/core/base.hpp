#pragma once

#include <memory>
#include <vector>

#include "autogen/utils/system.hpp"

namespace autogen {
using BaseScalar = double;

enum AccumulationMethod { ACCUMULATE_NONE, ACCUMULATE_SUM, ACCUMULATE_MEAN };

struct GeneratedBase {
 protected:
  int local_input_dim_{-1};
  int global_input_dim_{0};
  int output_dim_{-1};

 public:
  virtual ~GeneratedBase() {}

  virtual int local_input_dim() const { return local_input_dim_; }
  virtual int global_input_dim() const { return global_input_dim_; }
  virtual void set_global_input_dim(int dim) { global_input_dim_ = dim; }

  virtual int input_dim() const {
    return local_input_dim() + global_input_dim();
  }

  virtual int output_dim() const { return output_dim_; }

  /**
   * Forward pass.
   */
  virtual void operator()(const std::vector<BaseScalar> &input,
                          std::vector<BaseScalar> &output) {
    (*this)(input.data(), &(output[0]));
  }

  virtual void operator()(const BaseScalar *input, BaseScalar *output) {
    // if this function doesn't get overwritten we have to copy
    std::vector<BaseScalar> input_vec(input, input + input_dim());
    std::vector<BaseScalar> output_vec(output_dim());
    (*this)(input_vec, output_vec);
    for (int i = 0; i < output_dim(); ++i) {
      output[i] = output_vec[i];
    }
  }

  /**
   * Vectorized version of forward pass.
   */
  virtual void operator()(
      const std::vector<std::vector<BaseScalar>> &local_inputs,
      std::vector<std::vector<BaseScalar>> &outputs,
      const std::vector<BaseScalar> &global_input = {}) = 0;

  virtual void operator()(int num_total_threads, const BaseScalar *input,
                          BaseScalar *output) {
    const int gd = global_input_dim();
    const int ld = local_input_dim();
    const int od = output_dim();
    std::vector<BaseScalar> global_input_vec(input, input + gd);
    std::vector<std::vector<BaseScalar>> local_input_vec(num_total_threads);
    std::vector<std::vector<BaseScalar>> output_vec(
        num_total_threads, std::vector<BaseScalar>(od));
    for (int i = 0; i < num_total_threads; ++i) {
      local_input_vec[i] = std::vector<BaseScalar>(input + gd + i * ld,
                                                   input + gd + (i + 1) * ld);
    }
    (*this)(local_input_vec, output_vec, global_input_vec);
    int p = 0;
    for (int i = 0; i < num_total_threads; ++i) {
      for (int j = 0; j < od; ++j) {
        output[p++] = output_vec[i][j];
      }
    }
  }

  /**
   * Jacobian pass.
   */
  virtual void jacobian(const std::vector<BaseScalar> &input,
                        std::vector<BaseScalar> &output) = 0;

  /**
   * Vectorized version of Jacobian pass.
   */
  virtual void jacobian(
      const std::vector<std::vector<BaseScalar>> &local_inputs,
      std::vector<std::vector<BaseScalar>> &outputs,
      const std::vector<BaseScalar> &global_input = {}) = 0;
};
}  // namespace autogen