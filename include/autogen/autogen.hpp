#pragma once

#include <mutex>

// clang-format off
#include "core/codegen.hpp"
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

static inline std::string str(const GenerationMode& mode) {
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

static inline std::ostream& operator<<(std::ostream& os,
                                       const GenerationMode& mode) {
  os << str(mode);
  return os;
}

template <template <typename> typename Functor>
struct Generated {
  static inline std::map<std::string, FunctionTrace<BaseScalar>> traces;

  const std::string name;

  using ADScalar = typename CppAD::AD<BaseScalar>;
  using CGScalar = typename CppAD::cg::CG<BaseScalar>;
  using ADCGScalar = typename CppAD::AD<CGScalar>;
  using ADFun = typename FunctionTrace<BaseScalar>::ADFun;

  bool compile_in_background{false};
  bool is_compiling_{false};

 protected:
  std::unique_ptr<Functor<BaseScalar>> f_double_{nullptr};
  std::unique_ptr<Functor<ADScalar>> f_cppad_{nullptr};
  std::unique_ptr<Functor<ADCGScalar>> f_cg_{nullptr};

  std::unique_ptr<GeneratedNumerical> gen_double_{nullptr};
  std::unique_ptr<GeneratedCppAD> gen_cppad_{nullptr};
  std::unique_ptr<GeneratedCodeGen> gen_cg_{nullptr};

  size_t local_input_dim_{0};
  size_t global_input_dim_{0};
  size_t output_dim_{0};

  GenerationMode mode_{GENERATE_CPU};
  mutable std::mutex compilation_mutex_;

 public:
  template <typename... Args>
  Generated(const std::string& name, Args&&... args) : name(name) {
    f_double_ =
        std::make_unique<Functor<BaseScalar>>(std::forward<Args>(args)...);
    gen_double_ = std::make_unique<GeneratedNumerical>(*f_double_);
    f_cppad_ = std::make_unique<Functor<ADScalar>>(std::forward<Args>(args)...);
    f_cg_ = std::make_unique<Functor<ADCGScalar>>(std::forward<Args>(args)...);
  }

  GenerationMode mode() const { return mode_; }
  void set_mode(GenerationMode mode) {
    if (mode != this->mode_) {
      // changing the mode discards the previously compiled library
      discard_library();
    }
    this->mode_ = mode;
  }

  void discard_library() {
    if (gen_cg_) {
      // std::lock_guard<std::mutex> guard(compilation_mutex_);
      gen_cg_->discard_library();
    }
  }
  void load_precompiled_library(const std::string& path) {
    if (gen_cg_) {
      gen_cg_->load_precompiled_library(path);
    }
  }

  AccumulationMethod jacobian_acc_method() const {
    return this->jac_acc_method_;
  }
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

  bool debug_mode() const { return gen_cg_ && gen_cg_->debug_mode; }
  void set_debug_mode(bool debug_mode = true) {
    if (gen_cg_) {
      gen_cg_->debug_mode = debug_mode;
    }
  }

  size_t global_input_dim() const { return global_input_dim_; }
  void set_global_input_dim(size_t global_input_dim) {
    if (global_input_dim != global_input_dim_) {
      //      discard_library();
    }
    global_input_dim_ = global_input_dim;
  }

  bool is_compiled() const {
    switch (mode_) {
      case GENERATE_NONE:
        return true;
      case GENERATE_CPPAD:
        return (bool)gen_cppad_;
      case GENERATE_CPU:
      case GENERATE_CUDA:
        return gen_cg_ && gen_cg_->is_compiled();
    }
    return false;
  }

  bool is_compiling() const {
    //    std::lock_guard<std::mutex> guard(compilation_mutex_);
    //    return is_compiling_;
    return false;
  }

  void operator()(const std::vector<BaseScalar>& input,
                  std::vector<BaseScalar>& output) {
    conditionally_compile(input, output);

    if (mode_ == GENERATE_NONE) {
      (*gen_double_)(input, output);
    } else if (mode_ == GENERATE_CPPAD) {
      (*gen_cppad_)(input, output);
    } else {
      (*gen_cg_)(input, output);
    }
  }

  /**
   * Vectorized execution of the forward pass of this function, which will
   * compute the outputs for each of the inputs in parallel.
   */
  void operator()(const std::vector<std::vector<BaseScalar>>& local_inputs,
                  std::vector<std::vector<BaseScalar>>& outputs,
                  const std::vector<BaseScalar>& global_input = {}) {
    if (local_inputs.empty()) {
      return;
    }
    outputs.resize(local_inputs.size());

    conditionally_compile(local_inputs, outputs, global_input);

    if (mode_ == GENERATE_NONE) {
      (*gen_double_)(local_inputs, outputs, global_input);
    } else if (mode_ == GENERATE_CPPAD) {
      (*gen_cppad_)(local_inputs, outputs, global_input);
    } else {
      (*gen_cg_)(local_inputs, outputs, global_input);
    }
  }

  void jacobian(const std::vector<BaseScalar>& input,
                std::vector<BaseScalar>& output) {
    conditionally_compile(input, output);
    if (mode_ == GENERATE_NONE) {
      gen_double_->jacobian(input, output);
      return;
    }
    if (mode_ == GENERATE_CPPAD) {
      gen_cppad_->jacobian(input, output);
      return;
    }

    if (!is_compiled()) {
      throw std::runtime_error("The function \"" + name +
                               "\" has not yet been compiled in " + str(mode_) +
                               " mode. You need to call the forward pass first "
                               "to trigger the compilation of the Jacobian.\n");
    }

    gen_cg_->jacobian(input, output);
  }

  void jacobian(const std::vector<std::vector<BaseScalar>>& local_inputs,
                std::vector<std::vector<BaseScalar>>& outputs,
                const std::vector<BaseScalar>& global_input = {}) {
    outputs.resize(local_inputs.size());
    conditionally_compile(local_inputs, outputs, global_input);
    if (mode_ == GENERATE_NONE) {
      gen_double_->jacobian(local_inputs, outputs, global_input);
      return;
    }
    if (mode_ == GENERATE_CPPAD) {
      gen_cppad_->jacobian(local_inputs, outputs, global_input);
      return;
    }

    if (!is_compiled()) {
      throw std::runtime_error("The function \"" + name +
                               "\" has not yet been compiled in " + str(mode_) +
                               " mode. You need to call the forward pass first "
                               "to trigger the compilation of the Jacobian.\n");
    }

    gen_cg_->jacobian(local_inputs, outputs, global_input);
  }

 protected:
  void compile(const FunctionTrace<BaseScalar>& main_trace) {
    {
      // std::lock_guard<std::mutex> guard(compilation_mutex_);
      is_compiling_ = true;
    }

    gen_cg_->local_input_dim_ = this->local_input_dim_;
    gen_cg_->global_input_dim_ = this->global_input_dim_;
    gen_cg_->output_dim_ = this->output_dim_;
    if (mode_ == GENERATE_CPU) {
      gen_cg_->compile_cpu();
    } else if (mode_ == GENERATE_CUDA) {
      gen_cg_->compile_cuda();
    }

    {
      // std::lock_guard<std::mutex> guard(compilation_mutex_);
      is_compiling_ = false;
    }
  }

  void conditionally_compile(const std::vector<BaseScalar>& input,
                             std::vector<BaseScalar>& output) {
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
      std::vector<CppAD::AD<BaseScalar>> ax_, ay_;
      ax_.resize(input.size());
      ay_.resize(output.size());
      for (size_t i = 0; i < input.size(); ++i) {
        ax_[i] = ADScalar(input[i]);
      }
      CppAD::Independent(ax_);
      (*f_cppad_)(ax_, ay_);
      gen_cppad_ = std::make_unique<GeneratedCppAD>(
          std::make_shared<CppAD::ADFun<BaseScalar>>(ax_, ay_));
      return;
    }
    if (mode_ == GENERATE_CPU || mode_ == GENERATE_CUDA) {
      // std::lock_guard<std::mutex> guard(compilation_mutex_);

      assert(!input.empty());
      assert(!output.empty());
      std::vector<CppAD::cg::CG<BaseScalar>> cg_input(input.size());
      std::vector<CppAD::cg::CG<BaseScalar>> cg_output(output.size());
      for (size_t i = 0; i < input.size(); ++i) {
      }
      FunctionTrace<BaseScalar> t = autogen::trace(*f_cg_, name, input, output);
      gen_cg_ = std::make_unique<GeneratedCodeGen>(t);
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
      const std::vector<std::vector<BaseScalar>>& local_inputs,
      std::vector<std::vector<BaseScalar>>& outputs,
      const std::vector<BaseScalar>& global_input) {
    global_input_dim_ = global_input.size();
    std::vector<BaseScalar> compilation_input;
    compilation_input.insert(compilation_input.end(), global_input.begin(),
                             global_input.end());
    compilation_input.insert(compilation_input.end(), local_inputs[0].begin(),
                             local_inputs[0].end());
    conditionally_compile(compilation_input, outputs[0]);
    local_input_dim_ = local_inputs[0].size();
  }
};

}  // namespace autogen