#include "autogen/cg/target/openmp_target.h"

#include "autogen/cg/compiler/clang_compiler.hpp"

namespace autogen {
void OpenMpCodeGen::emit_kernel_launch(std::ostringstream &code,
                                       const std::string &function_name,
                                       size_t local_input_dim, 
                                       size_t global_input_dim,
                                       size_t output_dim) const {
  std::string fun_head_start = "MODULE_API void " + function_name + "(";
  std::string fun_arg_pad = std::string(fun_head_start.size(), ' ');
  code << fun_head_start;
  code << "int num_total_threads,\n";
  code << fun_arg_pad << "Float *output) {\n";

  code << "  const size_t output_dim = num_total_threads * " << output_dim
       << ";\n";
  std::string kernel_name = function_name + "_kernel";

  if (debug_mode_) {
    code << "  printf(\"Launching kernel \\\"" << kernel_name
         << "\\\" with %i thread(s).\\n\", num_total_threads);\n";
  }

  fun_head_start = "    " + kernel_name + "(";
  fun_arg_pad = std::string(fun_head_start.size(), ' ');
  code << "#pragma omp parallel for\n";
  code << "  for (int i = 0; i < num_total_threads; ++i) {\n";
  code << fun_head_start;
  code << "i,\n";
  code << fun_arg_pad << "dev_" << function_name << "_output,\n";
  code << fun_arg_pad << "dev_" << function_name << "_local_input";
  if (global_input_dim > 0) {
    code << ",\n" << fun_arg_pad << "dev_" << function_name << "_global_input";
  }
  code << ");\n";
  code << "  }\n";

  std::string outvar = "dev_" + function_name + "_output";

  if (jac_acc_method_ != ACCUMULATE_NONE) {
    code << "\n  // accumulate thread-wise outputs\n";
    if (debug_mode_) {
      code << "  printf(\"Accumulating outputs...\\n\");\n";
    }
    code << "  for (int i = 1; i < num_total_threads; ++i) {\n";
    code << "    for (int j = 0; j < " << output_dim << "; ++j) {\n";
    code << "      " << outvar << "[j] += " << outvar << "[i*" << output_dim
         << " + j];\n";
    code << "    }\n";
    code << "  }\n";
  }
  if (jac_acc_method_ == ACCUMULATE_MEAN) {
    code << "  for (int j = 0; j < " << output_dim << "; ++j) {\n";
    code << "    output[j] = " << outvar << "[j] / num_total_threads;\n";
    code << "  }\n";
  } else {
    code << "  for (int i = 0; i < num_total_threads; ++i) {\n";
    code << "    for (int j = 0; j < " << output_dim << "; ++j) {\n";
    code << "      output[j] = " << outvar << "[i * " << output_dim
         << " + j];\n";
    code << "    }\n";
    code << "  }\n";
  }

  if (debug_mode_) {
    code << "  printf(\"Output from kernel \\\"" << kernel_name
         << "\\\":\\n\");\n";
    if (jac_acc_method_ != ACCUMULATE_NONE) {
      code << "  for (int j = 0; j < " << output_dim << "; ++j) {\n";
      code << "    printf(\"%f  \", " << outvar << "[j]);\n";
      code << "  }\n";
    } else {
      code << "  for (int i = 0; i < num_total_threads; ++i) {\n";
      code << "    for (int j = 0; j < " << output_dim << "; ++j) {\n";
      code << "      printf(\"%f  \", " << outvar << "[i * " << output_dim
           << " + j];\n";
      code << "    }\n";
      code << "    printf(\"\\n\");\n";
      code << "  }\n";
    }
    code << "  printf(\"\\n\");\n";
  }

  code << "}\n";
  code << "}\n";
}

void OpenMpCodeGen::emit_allocation_functions(std::ostringstream &code,
                                              const std::string &function_name,
                                              size_t local_input_dim,
                                              size_t global_input_dim,
                                              size_t output_dim) const {
  // global device memory pointers
  code << "Float* dev_" << function_name << "_output = NULL;\n";
  code << "Float* dev_" << function_name << "_local_input = NULL;\n";
  if (global_input_dim > 0) {
    code << "Float* dev_" << function_name << "_global_input = NULL;\n\n";
  }

  // allocation function
  code << "extern \"C\" {\nMODULE_API void " << function_name
       << "_allocate(int num_total_threads) {\n";
  code << "  allocate(&dev_" << function_name << "_output, " << output_dim
       << " * num_total_threads);\n";
  // code << "  // do not allocate the inputs because we directly assign them to
  // "
  //         "the input memory pointers in the send function(s)\n";
  code << "  allocate(&dev_" << function_name << "_local_input, "
       << local_input_dim << " * num_total_threads);\n";
  if (global_input_dim > 0) {
    code << "  allocate(&dev_" << function_name << "_global_input, "
         << global_input_dim << ");\n";
  }
  code << "}\n\n";

  // deallocation function
  code << "MODULE_API void " << function_name << "_deallocate() {\n";
  code << "  free(dev_" << function_name << "_output);\n";
  code << "  free(dev_" << function_name << "_local_input);\n";
  if (global_input_dim > 0) {
    code << "  free(dev_" << function_name << "_global_input);\n";
  }
  code << "}\n\n";
}

void OpenMpCodeGen::emit_send_functions(std::ostringstream &code,
                                        const std::string &function_name,
                                        size_t local_input_dim,
                                        size_t global_input_dim,
                                        size_t output_dim) const {
  std::string fun_head_start =
      "MODULE_API bool " + function_name + "_send_local(";
  std::string fun_arg_pad = std::string(fun_head_start.size(), ' ');
  code << fun_head_start;
  code << "int num_total_threads,\n";
  code << fun_arg_pad << "const Float *input) {\n";
  // code << "  dev_" << function_name << "_local_input = input;\n";
  code << "  memcpy(dev_" << function_name
       << "_local_input, input, sizeof(Float) * num_total_threads * "
       << local_input_dim << ");\n";
  code << "  return true;\n";
  code << "}\n\n";

  if (global_input_dim > 0) {
    code << "MODULE_API bool " + function_name + "_send_global(";
    code << "const Float *input) {\n";
    // code << "  dev_" << function_name << "_global_input = input;\n";
    code << "  memcpy(dev_" << function_name
         << "_global_input, input, sizeof(Float) * " << global_input_dim
         << ");\n";
    code << "  return true;\n";
    code << "}\n\n";
  }
}

void OpenMpCodeGen::emit_cmake_code(std::ostringstream &code,
                                    const std::string &project_name) const {
  code << "cmake_minimum_required(VERSION 3.9)\n";
  code << "project(" << project_name << " LANGUAGES CXX)\n\n";
  code << "find_package(OpenMP)\n\n";
  code << "if (MSVC)\n";
  code << "  # activate floating-point exceptions\n";
  code << "  set(CMAKE_CXX_FLAGS \"/fp:except\")\n";
  code << "endif()\n\n";
  code << "add_executable(" << project_name << " main.cpp)\n\n";
  code << "if(OpenMP_CXX_FOUND)\n";
  code << "  target_link_libraries(" << project_name
       << " PUBLIC OpenMP::OpenMP_CXX)\n";
  code << "endif()\n";
}

void OpenMpTarget::set_compiler_clang(
    std::string compiler_path, const std::vector<std::string> &compile_flags,
    const std::vector<std::string> &compile_lib_flags) {
  if (compiler_path.empty()) {
    compiler_path = autogen::find_exe("clang++");
  }
  compiler_ = std::make_shared<ClangCompiler>(compiler_path);
  compiler_->addCompileFlag("-std=c++11");
  compiler_->addCompileFlag("-O0");
  // compiler_->addCompileFlag("-fopenmp");
  // compiler_->addCompileFlag("-fopenmp=libomp");
  for (const auto &flag : compile_flags) {
    compiler_->addCompileFlag(flag);
  }
  compiler_->addCompileLibFlag("-std=c++11");
  compiler_->addCompileFlag("-O0");
  for (const auto &flag : compile_lib_flags) {
    compiler_->addCompileLibFlag(flag);
  }
}

void OpenMpTarget::set_compiler_gcc(
    std::string compiler_path, const std::vector<std::string> &compile_flags,
    const std::vector<std::string> &compile_lib_flags) {
  if (compiler_path.empty()) {
    compiler_path = autogen::find_exe("g++");
  }
  compiler_ = std::make_shared<GccCompiler>(compiler_path);
  compiler_->addCompileFlag("-fopenmp");
  compiler_->addCompileFlag("-lstdc++");
  for (const auto &flag : compile_flags) {
    compiler_->addCompileFlag(flag);
  }
  for (const auto &flag : compile_lib_flags) {
    compiler_->addCompileLibFlag(flag);
  }
}

void OpenMpTarget::set_compiler_msvc(
    std::string compiler_path, std::string linker_path,
    const std::vector<std::string> &compile_flags,
    const std::vector<std::string> &compile_lib_flags) {
  if (compiler_path.empty()) {
    compiler_path = autogen::find_exe("cl.exe");
  }
  if (linker_path.empty()) {
    linker_path = autogen::find_exe("link.exe");
  }
  compiler_ = std::make_shared<MsvcCompiler>(compiler_path, linker_path);
  compiler_->addCompileFlag("/openmp");
  for (const auto &flag : compile_flags) {
    compiler_->addCompileFlag(flag);
  }
  for (const auto &flag : compile_lib_flags) {
    compiler_->addCompileLibFlag(flag);
  }
}

bool OpenMpTarget::compile_() {
  using namespace CppAD;
  using namespace CppAD::cg;

  if (compiler_ == nullptr) {
#if AUTOGEN_SYSTEM_WIN
    set_compiler_msvc();
#else
    set_compiler_clang();
#endif
  }
  compiler_->setSourcesFolder(sources_folder_);
  compiler_->setTemporaryFolder(temp_folder_);
  compiler_->setSaveToDiskFirst(true);

// TODO check type of compiler here, not operating system
#if AUTOGEN_SYSTEM_WIN
  if (debug_mode_) {
    compiler_->addCompileFlag("/DEBUG:FULL");
    compiler_->addCompileLibFlag("/DEBUG:FULL");
    compiler_->addCompileFlag("/Od");
  } else {
    compiler_->addCompileFlag("/O" + std::to_string(optimization_level_));
  }
#else
  if (debug_mode_) {
    compiler_->addCompileFlag("-O0");
    compiler_->addCompileFlag("-g");
  } else {
    compiler_->addCompileFlag("-O" + std::to_string(optimization_level_));
  }
#endif

  CppAD::cg::JobTimer *timer = new CppAD::cg::JobTimer();

  timer->startingJob("", JobTimer::DYNAMIC_MODEL_LIBRARY);

  std::map<std::string, std::string> source_files;
  for (const auto &[filename, content] : sources_) {
    if (std::find(source_filenames_.begin(), source_filenames_.end(),
                  filename) != source_filenames_.end()) {
      source_files[filename] = content;
    }
  }

  try {
    // for (const auto &p : models) {
    //   const std::map<std::string, std::string> &modelSources =
    //       this->getSources(*p.second);

    //   timer->startingJob("", JobTimer::COMPILING_FOR_MODEL);
    //   compiler_->compileSources(sources_, true, this->modelLibraryHelper_);
    //   timer->finishedJob();
    // }

    // const std::map<std::string, std::string> &sources =
    //     this->getLibrarySources();
    compiler_->compileSources(source_files, true, timer);

    // const std::map<std::string, std::string> &customSource =
    //     this->modelLibraryHelper_->getCustomSources();
    // compiler_->compileSources(customSource, true, this->modelLibraryHelper_);

    library_name_ = source_folder_prefix_ + this->canonical_name();
    std::string libname = library_name_;
    libname += system::SystemInfo<>::DYNAMIC_LIB_EXTENSION;

    compiler_->buildDynamic(libname, timer);

  } catch (...) {
    compiler_->cleanup();
    library_name_ = "";
    throw;
  }
  compiler_->cleanup();

  timer->finishedJob();

  return true;
}
}  // namespace autogen