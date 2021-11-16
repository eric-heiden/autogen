#pragma once

#include <cppad/cg/model/compiler/clang_compiler.hpp>

#include "utils/system.hpp"

namespace autogen {
/**
 * @brief Compiles CUDA code via clang.
 * https://llvm.org/docs/CompileCudaWithLLVM.html
 *
 * @tparam Base
 */
template <typename Base>
class ClangCudaCompiler : public CppAD::cg::ClangCompiler<Base> {
 public:
  std::string cuda_home_path;
  std::string cuda_lib_path;
  std::string cuda_gpu_arch;

  ClangCudaCompiler(const std::string& clangPath = "clang++",
                    const std::string& cuda_home_path = "",
                    const std::string& cuda_lib_path = "",
                    const std::string& cuda_gpu_arch = "sm_35")
      : AbstractCCompiler<Base>(clangPath), cuda_gpu_arch(cuda_gpu_arch) {
    std::string clang_path = autogen::find_exe(clangPath);
    this->setCompilerPath(clang_path);

    if (cuda_home_path.empty()) {
      if (this->cuda_home_path = std::getenv("CUDA_HOME")) {
        std::cout << "Using CUDA_HOME path " << this->cuda_home_path << '\n';
      } else {
        throw std::runtime_error(
            "Error in ClangCudaCompiler: CUDA_HOME variable is not defined as "
            "environment variable or `cuda_home_path` argument.");
      }
    } else {
      this->cuda_home_path = cuda_home_path;
    }
    
    if (cuda_lib_path.empty()) {
      if (this->cuda_lib_path = std::getenv("CUDA_LIB")) {
        std::cout << "Using CUDA_LIB path " << this->cuda_lib_path << '\n';
      } else {
        throw std::runtime_error(
            "Error in ClangCudaCompiler: CUDA_LIB variable is not defined as "
            "environment variable or `cuda_home_path` argument.");
      }
    } else {
      this->cuda_lib_path = cuda_lib_path;
    }

    this->_compileFlags.push_back("--cuda-gpu-arch=" + cuda_gpu_arch);
    this->_compileFlags.push_back("--cuda-path=\"" + this->cuda_home_path +
                                  "\"");

    this->_compileLibFlags.push_back("-L\"" + this->cuda_lib_path + "\"");
    this->_compileLibFlags.push_back("-lcudart");
    // this->_compileLibFlags.push_back("-lcuda");
    this->_compileLibFlags.push_back("-lcudart_static");
    this->_compileLibFlags.push_back("-pthread");
  }
};
}  // namespace autogen