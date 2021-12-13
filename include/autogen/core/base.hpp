#pragma once

#include <algorithm>
#include <cctype>
#include <memory>
#include <string>
#include <vector>

// operating system detection
#ifndef AUTOGEN_SYSTEM_LINUX
#if defined(__linux__) || defined(__linux) || defined(linux)
#define AUTOGEN_SYSTEM_LINUX 1
#endif
#endif
#ifndef AUTOGEN_SYSTEM_APPLE
#if defined(__APPLE__)
#define AUTOGEN_SYSTEM_APPLE 1
#define AUTOGEN_SYSTEM_LINUX 1
#endif
#endif
#ifndef AUTOGEN_SYSTEM_WIN
#if defined(_WIN32) || defined(_WIN64) || defined(__WIN32__) || \
    defined(__TOS_WIN__) || defined(__WINDOWS__)
#define AUTOGEN_SYSTEM_WIN 1
#endif
#endif

namespace autogen {
using BaseScalar = double;

/**
 * Method of how the output of a vectorized function is accumulated.
 * This setting typically only applies to the vectorized version of the Jacobian
 * pass of a function.
 *
 * ACCUMULATE_NONE: No accumulation.
 * ACCUMULATE_SUM: Sum of the output vectors.
 * ACCUMULATE_MEAN: Mean of the output vectors.
 */
enum AccumulationMethod { ACCUMULATE_NONE, ACCUMULATE_SUM, ACCUMULATE_MEAN };

enum GenerationMode { MODE_NUMERICAL, MODE_CPPAD, MODE_CODEGEN };

inline std::string to_lower(const std::string &str) {
  std::string result = str;
  std::transform(result.begin(), result.end(), result.begin(), std::tolower);
  return result;
}

inline bool starts_with(const std::string &str, const std::string &query) {
  return str.rfind(query, 0) == 0;
}

static inline void ltrim(std::string &s) {
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
            return !std::isspace(ch);
          }));
}

// trim from end (in place)
static inline void rtrim(std::string &s) {
  s.erase(std::find_if(s.rbegin(), s.rend(),
                       [](unsigned char ch) { return !std::isspace(ch); })
              .base(),
          s.end());
}

// trim from both ends (in place)
static inline void trim(std::string &s) {
  ltrim(s);
  rtrim(s);
}

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