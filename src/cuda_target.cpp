#include "autogen/cg/target/cuda_target.h"

namespace autogen {
void CudaCodeGen::emit_allocation_functions(std::ostringstream &code,
                                            const std::string &function_name,
                                            size_t local_input_dim,
                                            size_t global_input_dim,
                                            size_t output_dim) const {
  // global device memory pointers
  code << "Float* dev_" << function_name << "_output = nullptr;\n";
  code << "Float* dev_" << function_name << "_local_input = nullptr;\n";
  if (global_input_dim > 0) {
    code << "Float* dev_" << function_name << "_global_input = nullptr;\n\n";
  }

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
  if (global_input_dim > 0) {
    code << "  allocate((void**)&dev_" << function_name << "_global_input, "
         << global_input_dim << " * sizeof(Float));\n";
  }
  code << "}\n\n";

  // deallocation function
  code << "MODULE_API void " << function_name << "_deallocate() {\n";
  code << "  cudaFreeHost(dev_" << function_name << "_output);\n";
  code << "  cudaFreeHost(dev_" << function_name << "_local_input);\n";
  if (global_input_dim > 0) {
    code << "  cudaFreeHost(dev_" << function_name << "_global_input);\n";
  }
  code << "  // cudaDeviceReset();\n";
  code << "}\n\n";
}

void CudaCodeGen::emit_send_functions(std::ostringstream &code,
                                      const std::string &function_name,
                                      size_t local_input_dim,
                                      size_t global_input_dim,
                                      size_t output_dim) const {
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

  if (global_input_dim > 0) {
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
}

void CudaCodeGen::emit_kernel_launch(std::ostringstream &code,
                                     const std::string &function_name,
                                     size_t local_input_dim,
                                     size_t global_input_dim,
                                     size_t output_dim) const {
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
    code << ",\n" << fun_arg_pad << "dev_" << function_name << "_global_input";
  }
  code << ");\n";
  code << R"(
  // cudaDeviceSynchronize waits for the kernel to finish, and returns
  // any errors encountered during the launch.
  cudaDeviceSynchronize();
  cudaError_t status = cudaGetLastError();
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

  if (jac_acc_method_ != ACCUMULATE_NONE) {
    code << "\n\n  // accumulate thread-wise outputs\n";
    code << "  for (int i = 1; i < num_total_threads; ++i) {\n";
    code << "    for (int j = 0; j < " << output_dim << "; ++j) {\n";
    code << "      output[j] += output[i*" << output_dim << " + j];\n";
    code << "    }\n  }\n";
    if (jac_acc_method_ == ACCUMULATE_MEAN) {
      code << "  for (int j = 0; j < " << output_dim << "; ++j) {\n";
      code << "    output[j] /= num_total_threads;\n  }";
    }
  }

  code << "\n}\n}\n";
}

void CudaCodeGen::emit_cmake_code(std::ostringstream &code,
                                  const std::string &project_name) const {
  code << "cmake_minimum_required(VERSION 3.8)\n";
  code << "project(" << project_name << " LANGUAGES CXX CUDA)\n\n";
  code << "add_executable(" << project_name << " main.cpp)\n";
  code << "target_compile_features(" << project_name
       << " PUBLIC cxx_std_11)\n\n";
  code << "set_target_properties(" << project_name
       << " PROPERTIES CUDA_SEPARABLE_COMPILATION ON)\n";
  code << "set_property(TARGET " << project_name
       << " PROPERTY CUDA_SEPARABLE_COMPILATION ON)\n";
  code << "set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)\n";
  code << "set_source_files_properties(" << project_name
       << "_CUDA.cu PROPERTIES LANGUAGE CUDA)\n\n";
  code << "if(APPLE)\n";
  code << "  # We need to add the path to the driver (libcuda.dylib) as an "
          "rpath,\n";
  code << "  # so that the static cuda runtime can find it at runtime.\n";
  code << "  set_property(TARGET " << project_name << "\n";
  code << "               PROPERTY\n";
  code << "               BUILD_RPATH "
          "${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})\n";
  code << "endif(APPLE)\n";
}

void CudaCodeGen::emit_cpp_function_call(std::ostringstream &code,
                                         const std::string &fun_name) const {
  code << "    int num_threads_per_block = min(128, num_threads);\n";
  code << "    int num_blocks = static_cast<int>(ceil(double(num_threads) / "
          "num_threads_per_block));\n";
  code << "    " << fun_name << "(num_threads, num_blocks, "
       << "num_threads_per_block, output);\n";
}

void CudaTarget::set_compiler_nvcc(std::string compiler_path) {
  if (compiler_path.empty()) {
    compiler_path = autogen::find_exe("nvcc");
  }
  compiler_ =
      std::make_shared<NvccCompiler>(compiler_path, optimization_level_);
}

std::string CudaTarget::util_header_src_() const {
  std::ostringstream code;
  code << "#ifndef CUDA_UTILS_H\n#define CUDA_UTILS_H\n\n";
  code << "#include <math.h>\n#include <stdio.h>\n\n";
  code << "#include <cuda_runtime.h>\n#include <cuda.h>\n\n";

  code << "typedef " << codegen_->scalar_type() << " Float;\n\n";

  code << R"(#ifdef _WIN32
#define MODULE_API __declspec(dllexport)
#else
#define MODULE_API
#endif

struct CompactFunctionMetaData {
  int output_dim;
  int local_input_dim;
  int global_input_dim;
  int accumulated_output;
};

void allocate(void **x, size_t size) {
  cudaError status = cudaMallocHost(x, size);
  if (status != cudaSuccess) {
    fprintf(stderr, "Error %i (%s) while allocating %zu bytes of CUDA memory: %s.\n",
            status, cudaGetErrorName(status), size, cudaGetErrorString(status));
    exit((int)status);
  }
}

#endif  // CUDA_UTILS_H)";
  return code.str();
}
}  // namespace autogen