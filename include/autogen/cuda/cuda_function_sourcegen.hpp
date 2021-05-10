#pragma once

#include <regex>
#include <unordered_map>

#include "autogen/core/base.hpp"
#include "cuda_language.hpp"

namespace autogen {struct CudaFunctionSourceGen {
  std::string function_name;
  size_t local_input_dim;
  size_t global_input_dim;
  size_t output_dim;
  AccumulationMethod acc_method;

  bool is_forward_one{false};
  bool is_reverse_one{false};

  CudaFunctionSourceGen(const std::string &function_name,
                        size_t local_input_dim, size_t global_input_dim,
                        size_t output_dim, AccumulationMethod acc_method)
      : function_name(function_name),
        local_input_dim(local_input_dim),
        global_input_dim(global_input_dim),
        output_dim(output_dim),
        acc_method(acc_method) {}

  void emit_header(std::ostringstream &code) const {
    // meta data retrieval function
    code << "extern \"C\" {\nMODULE_API CudaFunctionMetaData " << function_name
         << "_meta() {\n";
    code << "  CudaFunctionMetaData data;\n";
    code << "  data.output_dim = " << output_dim << ";\n";
    code << "  data.local_input_dim = " << local_input_dim << ";\n";
    code << "  data.global_input_dim = " << global_input_dim << ";\n";
    code << "  data.accumulated_output = " << std::boolalpha
         << (acc_method != ACCUMULATE_NONE) << ";\n";
    code << "  return data;\n}\n}\n";
  }

  template <typename Base>
  void emit_kernel(std::ostringstream &code, const std::ostringstream &body,
                   LanguageCuda<Base> &language,
                   bool is_function = false) const {
    std::string kernel_name = function_name;
    if (is_function) {
      code << "__device__\n";
    } else {
      code << "\n__global__\n";
      kernel_name += "_kernel";
    }
    std::string fun_head_start = "void " + kernel_name + "(";
    std::string fun_arg_pad = std::string(fun_head_start.size(), ' ');
    code << fun_head_start;
    if (!is_function) {
      code << "int num_total_threads,\n";
      code << fun_arg_pad;
    }
    if (is_forward_one) {
      code << "Float *out,\n";
      // code << fun_arg_pad << "Float const *const * in";
      code << fun_arg_pad << "const Float *x,\n";
      code << fun_arg_pad << "const Float *dx";
    } else if (is_reverse_one) {
      code << "Float *out,\n";
      // code << fun_arg_pad << "Float const *const * in";
      code << fun_arg_pad << "const Float *x,\n";
      // code << fun_arg_pad << "const Float *ty,\n";
      code << fun_arg_pad << "const Float *py";
    } else {
      code << "Float *out,\n";
      code << fun_arg_pad << "const Float *local_input";
      if (global_input_dim > 0) {
        code << ",\n" << fun_arg_pad << "const Float *global_input";
      }
    }
    code << ") {\n";
    if (is_forward_one) {
      // code << "  // independent variables\n";
      // code << "  const Float* x = in[0];\n";
      // code << "  const Float* dx = in[1];\n\n";
      // code << "  const Float* x = in;\n";
      // code << "  const Float* dx = in;\n\n";
      // code << "  const Float dx[1] = {1};\n\n";
      code << "  // dependent variables\n";
      // code << "  Float* dy = out[0];\n\n";
      code << "  Float* dy = out;\n\n";

      if (LanguageCuda<Base>::add_debug_prints) {
        code << "  printf(\"\\t" << kernel_name << ":\\n\");\n";
        code << "  printf(\"\\tx:  \"); for (unsigned long i = 0; i < "
             << local_input_dim
             << ";++i) printf(\"%f  \", x[i]); printf(\"\\n\");\n";
      }
    } else if (is_reverse_one) {
      // code << "  // independent variables\n";
      // code << "  const Float* x = in[0];\n";
      // code << "  const Float* dx = in[1];\n\n";
      // code << "  const Float* x = in;\n";
      // code << "  const Float* dx = in;\n\n";
      // code << "  const Float dx[1] = {1};\n\n";
      code << "  // dependent variables\n";
      // code << "  Float* dy = out[0];\n\n";
      code << "  Float* dw = out;\n\n";

      if (LanguageCuda<Base>::add_debug_prints) {
        code << "  printf(\"\\t" << kernel_name << ":\\n\");\n";
        code << "  printf(\"\\tx:   \"); for (unsigned long i = 0; i < "
             << local_input_dim
             << ";++i) printf(\"%f  \", x[i]); printf(\"\\n\");\n";
      }
    } else {
      if (!is_function) {
        code << "  const int ti = blockIdx.x * blockDim.x + threadIdx.x;\n";
        code << "  if (ti >= num_total_threads) {\n";
        code << "    printf(\"ERROR: thread index %i in function \\\""
             << function_name
             << "\\\" exceeded provided "
                "number of total threads %i.\\n\", ti, num_total_threads);\n";
        code << "    return;\n  }\n";
      }
      code << "\n";
      if (global_input_dim > 0) {
        code << "  const Float *x = &(global_input[0]);  // global input\n";
      }
      if (!is_function) {
        code << "  const Float *xj = &(local_input[ti * " << local_input_dim
             << "]);  // thread-local input\n";
        code << "  Float *y = &(out[ti * " << output_dim << "]);\n";
      } else {
        code << "  const Float *xj = &(local_input[0]);  // thread-local "
                "input\n";
        code << "  Float *y = &(out[0]);\n";
      }
    }

    auto &info = language.getInfo();
    code << language.generateTemporaryVariableDeclaration(
        false, false, info->atomicFunctionsMaxForward,
        info->atomicFunctionsMaxReverse);

    code << "\n";

    std::string body_str = body.str();

    // std::cout << "Replacing variables in function " << function_name << "...\n";
    // size_t consts = replace_constants(body_str);
    // if (consts > 0) {
    //   std::cout << "Introduced " << consts << " constant variable[s].\n";
    // }
    code << body_str;

    if (LanguageCuda<Base>::add_debug_prints) {
      code << "  printf(\"\\t" << kernel_name << ":\\n\");\n";
      if (is_forward_one || is_reverse_one) {
        code << "  printf(\"\\tx:   \"); for (unsigned long i = 0; i < "
             << local_input_dim
             << ";++i) printf(\"%f  \", x[i]); printf(\"\\n\");\n";
        if (is_forward_one) {
          code << "  printf(\"\\tdx:   \"); for (unsigned long i = 0; i < "
               << local_input_dim
               << ";++i) printf(\"%f  \", dx[i]); printf(\"\\n\");\n";
        } else {
          code << "  printf(\"\\tpy:   \"); for (unsigned long i = 0; i < "
               << local_input_dim
               << ";++i) printf(\"%f  \", py[i]); printf(\"\\n\");\n";
        }
        code << "  printf(\"\\tout: \"); for (unsigned long i = 0; i < "
             << output_dim
             << ";++i) printf(\"%f  \", out[i]); printf(\"\\n\");\n";
      } else {
        code << "  printf(\"\\tx:   \"); for (unsigned long i = 0; i < "
             << local_input_dim
             << ";++i) printf(\"%f  \", xj[i]); printf(\"\\n\");\n";
        code << "  printf(\"\\ty:   \"); for (unsigned long i = 0; i < "
             << output_dim
             << ";++i) printf(\"%f  \", y[i]); printf(\"\\n\");\n";
      }
    }

    code << "}\n\n";
  }

  void emit_allocation_functions(std::ostringstream &code) const {
    // global device memory pointers
    code << "Float* dev_" << function_name << "_output = nullptr;\n";
    code << "Float* dev_" << function_name << "_local_input = nullptr;\n";
    code << "Float* dev_" << function_name << "_global_input = nullptr;\n\n";

    // allocation function
    code << "extern \"C\" {\nMODULE_API void " << function_name
         << "_allocate(int num_total_threads) {\n";
    code << "  const size_t output_dim = num_total_threads * " << output_dim
         << ";\n";
    code << "  const size_t input_dim = num_total_threads * " << local_input_dim
         << ";\n\n";
    code << "  allocate((void**)&dev_" << function_name
         << "_output, output_dim * sizeof(Float));\n";
    code << "  allocate((void**)&dev_" << function_name
         << "_local_input, input_dim * "
            "sizeof(Float));\n";
    code << "  allocate((void**)&dev_" << function_name << "_global_input, "
         << global_input_dim << " * sizeof(Float));\n";
    code << "}\n\n";

    // deallocation function
    code << "MODULE_API void " << function_name << "_deallocate() {\n";
    code << "  cudaFreeHost(dev_" << function_name << "_output);\n";
    code << "  cudaFreeHost(dev_" << function_name << "_local_input);\n";
    code << "  cudaFreeHost(dev_" << function_name << "_global_input);\n";
    code << "  // cudaDeviceReset();\n";
    code << "}\n\n";
  }

  void emit_send_functions(std::ostringstream &code) const {
    // send thread-local inputs to GPU
    std::string fun_head_start =
        "MODULE_API bool " + function_name + "_send_local(";
    std::string fun_arg_pad = std::string(fun_head_start.size(), ' ');
    code << fun_head_start;
    code << "int num_total_threads,\n";
    code << fun_arg_pad << "const Float *input) {\n";
    code << "  const size_t input_dim = num_total_threads * " << local_input_dim
         << ";\n";
    code << "  cudaError status = cudaMemcpy(dev_" << function_name
         << "_local_input, input, "
            "input_dim * sizeof(Float), "
            "cudaMemcpyHostToDevice);\n";
    code << R"(  if (status != cudaSuccess) {
    fprintf(stderr, "Error %i (%s) in function \")"
         << function_name
         << R"(\" while sending thread-local input data to GPU: %s.\n",
            status, cudaGetErrorName(status), cudaGetErrorString(status));
    return false;
  }
)";
    code << "  return true;\n}\n\n";

    // send global input to GPU
    code << "MODULE_API bool " + function_name + "_send_global(";
    code << "const Float *input) {\n";
    code << "  cudaError status = cudaMemcpy(dev_" << function_name
         << "_global_input, input, " << global_input_dim
         << " * sizeof(Float), "
            "cudaMemcpyHostToDevice);\n";
    code << R"(  if (status != cudaSuccess) {
    fprintf(stderr, "Error %i (%s) in function \")"
         << function_name
         << R"(\" while sending global input data to GPU: %s.\n",
            status, cudaGetErrorName(status), cudaGetErrorString(status));
    return false;
  }
)";
    code << "  return true;\n}\n\n";
  }

  void emit_kernel_launch(std::ostringstream &code) const {
    std::string fun_head_start = "MODULE_API void " + function_name + "(";
    std::string fun_arg_pad = std::string(fun_head_start.size(), ' ');
    code << fun_head_start;
    code << "int num_total_threads,\n";
    code << fun_arg_pad << "int num_blocks,\n";
    code << fun_arg_pad << "int num_threads_per_block,\n";
    code << fun_arg_pad << "Float *output) {\n";

    code << "  const size_t output_dim = num_total_threads * " << output_dim
         << ";\n";

    std::string kernel_name = function_name + "_kernel";
    fun_head_start =
        "  " + kernel_name + "<<<num_blocks, num_threads_per_block>>>(";
    fun_arg_pad = std::string(fun_head_start.size(), ' ');
    code << fun_head_start;
    code << "num_total_threads,\n";
    code << fun_arg_pad << "dev_" << function_name << "_output,\n";
    code << fun_arg_pad << "dev_" << function_name << "_local_input";
    if (global_input_dim > 0) {
      code << ",\n"
           << fun_arg_pad << "dev_" << function_name << "_global_input";
    }
    code << ");\n";
    code << R"(
  // cudaDeviceSynchronize waits for the kernel to finish, and returns
  // any errors encountered during the launch.
  cudaDeviceSynchronize();
  cudaError status = cudaGetLastError();
  if (status != cudaSuccess) {
    fprintf(stderr, "Error %i (%s) in function \")"
         << function_name << R"(\" while executing CUDA kernel: %s.\n",
            status, cudaGetErrorName(status), cudaGetErrorString(status));
    exit((int)status);
  }

  // Copy output vector from GPU buffer to host memory.
  )";
    code << "cudaMemcpy(output, dev_" << function_name
         << "_output, output_dim * sizeof(Float), cudaMemcpyDeviceToHost);\n";
    code << R"(  status = cudaGetLastError();
  if (status != cudaSuccess) {
    fprintf(stderr, "Error %i (%s) in function \")"
         << function_name << R"(\" while retrieving output from kernel: %s.\n",
            status, cudaGetErrorName(status), cudaGetErrorString(status));
    exit((int)status);
  })";

    if (acc_method != ACCUMULATE_NONE) {
      code << "\n\n  // accumulate thread-wise outputs\n";
      code << "  for (int i = 1; i < num_total_threads; ++i) {\n";
      code << "   for (int j = 0; j < " << output_dim << "; ++j) {\n";
      code << "     output[j] += output[i*" << output_dim << " + j];\n";
      code << "   }\n  }\n";
      if (acc_method == ACCUMULATE_MEAN) {
        code << "  for (int j = 0; j < " << output_dim << "; ++j) {\n";
        code << "   output[j] /= num_total_threads;\n  }";
      }
    }

    code << "\n}\n}\n";
  }

 protected:
  // replaces repeating constants by variables to reduce the number of
  // characters
  static size_t replace_constants(std::string &code) {
    std::match_results<std::string::const_iterator> res;
    std::string::const_iterator begin = code.cbegin(), end = code.cend();
    std::unordered_map<std::string, int> counts;
    static const std::basic_regex float_regex(
        "[\\W]([\\-]?[0-9\\.]+(e[\\-0-9]+)?)");
    while (std::regex_search(begin, end, res, float_regex)) {
      const auto &r = res[1];
      // std::cout << r.length() << "  -  " << r.str() << "\n";
      begin = r.second;
      std::string match = r.str();
      if (match.size() < 4) {
        continue;  // don't bother with short numbers
      }
      if (counts.find(match) == counts.end()) {
        counts[match] = 0;
      }
      counts[match]++;
    }
    std::stringstream ss;
    size_t const_id = 0;
    for (const auto &[value, count] : counts) {
      if (count > 1) {
        std::string varname = "c" + std::to_string(const_id++);
        std::basic_regex value_regex("([\\W])(" + value + ")([\\W;])");
        code = std::regex_replace(code, value_regex, "$1" + varname + "$2");
        // bool match_found = true;
        // while (match_found) {
        //   begin = code.cbegin();
        //   end = code.cend();
        //   match_found = std::regex_search(begin, end, res, value_regex);
        //   if (match_found) {
        //     code = code.replace(res[1].first, res[1].second, varname);
        //     begin = std::advacecode.cbegin() + res[1].first;
        //   }
        // }
        ss << "  static const Float " << varname << " = " << value << ";\n";
      }
    }
    ss << code;
    std::cout << "Replaced " << const_id << " constant(s).\n";
    code = ss.str();
    return const_id;
  }
};
}  // namespace autogen