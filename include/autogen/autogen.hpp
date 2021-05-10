#pragma once

// clang-format off
#include "core/base.hpp"
#include "core/generated_numerical.hpp"
#include "core/generated_cppad.hpp"
#include "core/generated_codegen.hpp"
// clang-format on

namespace autogen {
enum GenerationMode {
  GENERATE_NONE,
  GENERATE_CPPAD,
  GENERATE_CPU,
  GENERATE_CUDA
};

static inline std::string str(const GenerationMode &mode) {
  switch (mode) {
    case GENERATE_NONE:
      return "None";
    case GENERATE_CPPAD:
      return "CppAD";
    case GENERATE_CPU:
      return "CPU";
    case GENERATE_CUDA:
      return "CUDA";
  }
  return "Unknown";
}

static inline std::ostream &operator<<(std::ostream &os,
                                       const GenerationMode &mode) {
  os << str(mode);
  return os;
}

template <template <typename> typename Functor>
struct Generated {
  static inline std::map<std::string, FunctionTrace<BaseScalar>> traces;

  const std::string name;

 protected:
  std::unique_ptr<Functor<BaseScalar>> f_double_;
  std::unique_ptr<Functor<CppADScalar>> f_cppad_;
  std::unique_ptr<Functor<CGScalar>> f_cg_;

  std::unique_ptr<GeneratedNumerical> gen_double_;
  std::unique_ptr<GeneratedCppAD> gen_cppad_;
  std::unique_ptr<GeneratedCodeGen> gen_cg_;

  GenerationMode mode_{GENERATE_CPU};

 public:
  template <typename... Args>
  Generated(const std::string &name, Args &&... args) : name(name) {
    f_double_ =
        std::make_unique<Functor<BaseScalar>>(std::forward<Args>(args)...);
    f_cppad_ =
        std::make_unique<Functor<CppADScalar>>(std::forward<Args>(args)...);
    f_cg_ = std::make_unique<Functor<CGScalar>>(std::forward<Args>(args)...);
    // TODO create Generated... "gen_..." instances
  }

  GenerationMode mode() const { return mode_; }
  void set_mode(GenerationMode mode) {
    if (mode != this->mode_) {
      // changing the mode discards the previously compiled library
      discard_library();
    }
    this->mode_ = mode;
  }

  AccumulationMethod jacobian_acc_method() const { return jac_acc_method_; }
  void set_jacobian_acc_method(AccumulationMethod jac_acc_method) {
    if (jac_acc_method != this->jac_acc_method_) {
      // changing the jac_acc_method discards the previously compiled library
      discard_library();
    }
    this->jac_acc_method_ = jac_acc_method;
  }

  size_t input_dim() const { return local_input_dim_ + global_input_dim_; }
  size_t local_input_dim() const { return local_input_dim_; }
  size_t output_dim() const { return output_dim_; }

  size_t global_input_dim() const { return global_input_dim_; }
  void set_global_input_dim(size_t global_input_dim) {
    if (global_input_dim != global_input_dim_) {
      discard_library();
    }
    global_input_dim_ = global_input_dim;
  }

  bool is_compiled() const {
    std::lock_guard<std::mutex> guard(compilation_mutex_);
    switch (mode_) {
      case GENERATE_NONE:
        return true;
      case GENERATE_CPPAD:
        return bool(tape_) && !ax_.empty() && !ay_.empty();
      case GENERATE_CPU:
      case GENERATE_CUDA:
        return !library_name_.empty();
    }
    return false;
  }

  bool is_compiling() const {
    std::lock_guard<std::mutex> guard(compilation_mutex_);
    return is_compiling_;
  }

  void compile(const FunctionTrace<BaseScalar> &main_trace) {
    {
      std::lock_guard<std::mutex> guard(compilation_mutex_);
      is_compiling_ = true;
    }

    if (mode_ == GENERATE_CPU) {
      compile_cpu(main_trace);
    } else if (mode_ == GENERATE_CUDA) {
      compile_cuda(main_trace);
    }

    {
      std::lock_guard<std::mutex> guard(compilation_mutex_);
      is_compiling_ = false;
    }
  }

  void operator()(const std::vector<BaseScalar> &input,
                  std::vector<BaseScalar> &output) {
    conditionally_compile(input, output);

    if (mode_ == GENERATE_NONE) {
      (*f_double_)(input, output);
    } else if (mode_ == GENERATE_CPU) {
#if CPPAD_CG_SYSTEM_WIN
      std::cerr << "CPU code generation is not yet available on Windows.\n";
      return;
#else
      assert(!library_name_.empty());
      GenericModelPtr &model = get_cpu_model();
      model->ForwardZero(input, output);
#endif
    } else if (mode_ == GENERATE_CUDA) {
      const auto &model = get_cuda_model();
      model.forward_zero(input, output);
    } else if (mode_ == GENERATE_CPPAD) {
      output = tape_->Forward(0, input);
    }
  }

  /**
   * Vectorized execution of the forward pass of this function, which will
   * compute the outputs for each of the inputs in parallel.
   */
  void operator()(const std::vector<std::vector<BaseScalar>> &local_inputs,
                  std::vector<std::vector<BaseScalar>> &outputs,
                  const std::vector<BaseScalar> &global_input = {}) {
    if (local_inputs.empty()) {
      return;
    }
    outputs.resize(local_inputs.size());

    conditionally_compile(local_inputs, outputs, global_input);

    if (mode_ == GENERATE_NONE || mode_ == GENERATE_CPPAD) {
      if (global_input.empty()) {
        for (size_t i = 0; i < local_inputs.size(); ++i) {
          if (mode_ == GENERATE_NONE) {
            (*f_double_)(local_inputs[i], outputs[i]);
          } else {
            outputs[i] = tape_->Forward(0, local_inputs[i]);
          }
        }
      } else {
        std::vector<BaseScalar> input(global_input);
        input.resize(global_input.size() + local_inputs[0].size());
        for (size_t i = 0; i < local_inputs.size(); ++i) {
          for (size_t j = 0; j < local_inputs[i].size(); ++j) {
            input[j + global_input.size()] = local_inputs[i][j];
          }
          if (mode_ == GENERATE_NONE) {
            (*f_double_)(input, outputs[i]);
          } else {
            outputs[i] = tape_->Forward(0, input);
          }
        }
      }
    }

    // TODO call codegen module
  }

  void jacobian(const std::vector<BaseScalar> &input,
                std::vector<BaseScalar> &output) {
    conditionally_compile(input, output);
    if (mode_ == GENERATE_NONE) {
      // central difference
      assert(output_dim() > 0);
      output.resize(input_dim() * output_dim());
      std::vector<BaseScalar> left_x = input, right_x = input;
      std::vector<BaseScalar> left_y(output_dim()), right_y(output_dim());
      for (size_t i = 0; i < input.size(); ++i) {
        left_x[i] -= finite_diff_eps;
        right_x[i] += finite_diff_eps;
        BaseScalar dx = right_x[i] - left_x[i];
        (*f_double_)(left_x, left_y);
        (*f_double_)(right_x, right_y);
        for (size_t j = 0; j < output_dim(); ++j) {
          output[j * input_dim() + i] = (right_y[j] - left_y[j]) / dx;
        }
        left_x[i] = right_x[i] = input[i];
      }
      return;
    }
    if (mode_ == GENERATE_CPPAD) {
      output = tape_->Jacobian(input);
      return;
    }

    if (!is_compiled()) {
      throw std::runtime_error("The function \"" + name +
                               "\" has not yet been compiled in " + str(mode_) +
                               " mode. You need to call the forward pass first "
                               "to trigger the compilation of the Jacobian.\n");
    }

    // TODO call codegen module
  }

  void jacobian(const std::vector<std::vector<BaseScalar>> &local_inputs,
                std::vector<std::vector<BaseScalar>> &outputs,
                const std::vector<BaseScalar> &global_input = {}) {
    outputs.resize(local_inputs.size());
    conditionally_compile(local_inputs, outputs, global_input);
    if (mode_ == GENERATE_NONE || mode_ == GENERATE_CPPAD) {
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

    if (!is_compiled()) {
      throw std::runtime_error("The function \"" + name +
                               "\" has not yet been compiled in " + str(mode_) +
                               " mode. You need to call the forward pass first "
                               "to trigger the compilation of the Jacobian.\n");
    }

    // TODO call codegen module
  }

 protected:
  void conditionally_compile(const std::vector<BaseScalar> &input,
                             std::vector<BaseScalar> &output) {
    if (input_dim() == 0 || output_dim() == 0) {
      // retrieve dimensions by evaluating double-instantiated functor on
      // provided input
      (*f_double_)(input, output);
      local_input_dim_ = input.size();
      output_dim_ = output.size();
    }
    if (is_compiled()) {
      return;
    }
    if (mode_ == GENERATE_CPPAD) {
      tape_ = std::make_shared<CppAD::ADFun<BaseScalar>>();
      ax_.resize(input.size());
      ay_.resize(output.size());
      for (size_t i = 0; i < input.size(); ++i) {
        ax_[i] = CppADScalar(input[i]);
      }
      CppAD::Independent(ax_);
      (*f_cppad_)(ax_, ay_);
      tape_->Dependent(ax_, ay_);
      return;
    }
    if (mode_ == GENERATE_CPU || mode_ == GENERATE_CUDA) {
      assert(!input.empty());
      assert(!output.empty());
      FunctionTrace<BaseScalar> t = trace(input, output);
      if (compile_in_background) {
        std::thread worker([this, &t]() { compile(t); });
        (*f_double_)(input, output);
        return;
      } else {
        compile(t);
        std::cout << "Finished compilation.\n";
      }
    }
  }

  void conditionally_compile(
      const std::vector<std::vector<BaseScalar>> &local_inputs,
      std::vector<std::vector<BaseScalar>> &outputs,
      const std::vector<BaseScalar> &global_input) {
    global_input_dim_ = global_input.size();
    std::vector<BaseScalar> compilation_input;
    compilation_input.insert(compilation_input.end(), global_input.begin(),
                             global_input.end());
    compilation_input.insert(compilation_input.end(), local_inputs[0].begin(),
                             local_inputs[0].end());
    conditionally_compile(compilation_input, outputs[0]);
  }
};

}  // namespace autogen