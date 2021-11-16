#include "autogen/cg/target/compact_target.h"

#include "autogen/utils/file_utils.hpp"

namespace autogen {
template <class CodeGenT, class LibFunctionT>
bool CompactTarget<CodeGenT, LibFunctionT>::generate_code_() {
  LanguageCompact::add_debug_prints = debug_mode_;
  codegen_->set_debug_mode(debug_mode_);
  sources_.push_back(std::make_pair("util.h", util_header_src_()));
  sources_.push_back(std::make_pair("model_info.cpp", model_info_src_()));
  source_filenames_.push_back("model_info.cpp");
  std::string header_ext = codegen_->header_file_extension();
  std::string source_ext = codegen_->source_file_extension();

  // reverse order of invocation to first generate code for innermost
  // functions
  const auto &order = *CodeGenData::invocation_order;
  std::vector<CodeGenT *> models;
  for (auto it = order.rbegin(); it != order.rend(); ++it) {
    FunctionTrace &trace = (*CodeGenData::traces)[*it];
    bool is_function_only = true;
    auto source_gen = new CodeGen(*it, is_function_only, debug_mode_);
    source_gen->set_tape(*trace.tape);
    // source_gen->setCreateSparseJacobian(true);
    // source_gen->setCreateJacobian(true);
    source_gen->create_forward_zero(generate_forward_);
    source_gen->create_forward_one(generate_jacobian_);
    source_gen->create_reverse_one(generate_jacobian_);
    source_gen->create_sparse_forward_one(generate_jacobian_);
    models.push_back(source_gen);
  }
  codegen_->create_forward_zero(generate_forward_);
  codegen_->create_jacobian(generate_jacobian_);
  models.push_back(codegen_.get());

  std::string header_filename = name() + header_ext;
  std::string source_prepend = "#include \"" + header_filename + "\"\n";
  std::ostringstream header_code;
  header_code << "#ifndef " << name() << "_H\n";
  header_code << "#include \"util.h\"\n";

  for (auto &cgen : models) {
    if (cgen->is_function_only()) {
      header_code << "\n// " << cgen->name() << "\n";
      cgen->emit_function_signature(header_code);
    }
    std::string extension =
        source_ext;  // cgen->is_function_only() ? header_ext : source_ext;
    if (cgen->create_forward_zero()) {
      std::string src_name = cgen->name() + "_forward_zero" + extension;
      // generate CUDA code
      std::string source = source_prepend + cgen->forward_zero_source();
      sources_.push_back(std::make_pair(src_name, source));
      source_filenames_.push_back(src_name);
    }
    if (cgen->create_sparse_forward_one()) {
      std::string src_name = cgen->name() + "_forward_one" + extension;
      // generate CUDA code
      std::string source = source_prepend + cgen->forward_one_source(sources_);
      sources_.push_back(std::make_pair(src_name, source));
      source_filenames_.push_back(src_name);
    }
    if (cgen->create_reverse_one()) {
      std::string src_name = cgen->name() + "_reverse_one" + extension;
      // generate CUDA code
      std::string source = source_prepend + cgen->reverse_one_source(sources_);
      sources_.push_back(std::make_pair(src_name, source));
      source_filenames_.push_back(src_name);
    }
    if (cgen->create_jacobian()) {
      std::string src_name = cgen->name() + "_jacobian" + extension;
      // generate CUDA code
      std::string source = source_prepend + cgen->jacobian_source();
      sources_.push_back(std::make_pair(src_name, source));
      source_filenames_.push_back(src_name);
    }
    if (cgen->create_sparse_jacobian()) {
      std::string src_name = cgen->name() + "_sparse_jacobian" + extension;
      // generate CUDA code
      std::string source = source_prepend + cgen->sparse_jacobian_source();
      sources_.push_back(std::make_pair(src_name, source));
      source_filenames_.push_back(src_name);
    }
  }

  // delete code generators for atomic functions (they were only temporary)
  for (size_t i = 0; i < models.size() - 1; ++i) {
    delete models[i];
  }

  header_code << "#endif  // " << name() << "_H\n";
  sources_.push_back(std::make_pair(header_filename, header_code.str()));

  // generate "main" source file
  std::stringstream main_file;
  main_file << "#include \"util.h\"\n";
  for (const auto &src : source_filenames_) {
    main_file << "#include \"" << src << "\"\n";
  }
  library_name_ = name() + "_" + std::to_string(type_);
  sources_.push_back(
      std::make_pair(library_name_ + source_ext, main_file.str()));

  save_sources_();
  return true;
}

template <class CodeGenT, class LibFunctionT>
std::string CompactTarget<CodeGenT, LibFunctionT>::util_header_src_() const {
  std::ostringstream code;
  code << "#ifndef UTILS_H\n#define UTILS_H\n\n";
  code << "#include <math.h>\n";
  code << "#include <string.h>\n";
  code << "#include <stdio.h>\n";
  code << "#include <stdlib.h>\n\n";

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

inline void allocate(Float **x, int size) {
  *x = (Float*) malloc(size * sizeof(Float));
}

#endif  // UTILS_H)";
  return code.str();
}

template <class CodeGenT, class LibFunctionT>
std::string CompactTarget<CodeGenT, LibFunctionT>::model_info_src_() const {
  std::ostringstream code;
  code << "#ifndef MODEL_INFO_H\n#define MODEL_INFO_H\n\n";
  code << "#include \"util.h\"\n\n";
  code << "extern \"C\" {\n";
  code << "MODULE_API void model_info(char const *const **names, int *count) "
          "{\n";
  code << "  static const char *const models[] = {\n";
  std::vector<std::string> accessible_kernels;
  // XXX we could support multiple models in the same library in the future
  accessible_kernels.push_back(codegen_->name());
  for (std::size_t i = 0; i < accessible_kernels.size(); ++i) {
    code << "    \"" << accessible_kernels[i] << "\"";
    if (i < accessible_kernels.size() - 1) {
      code << ",";
    }
    code << "\n";
  }
  code << "  };\n";
  code << "  *names = models;\n";
  code << "  *count = " << accessible_kernels.size() << ";\n}\n";
  code << "}\n";
  code << "#endif  // MODEL_INFO_H\n";
  return code.str();
}

template <class CodeGenT, class LibFunctionT>
bool CompactTarget<CodeGenT, LibFunctionT>::create_cmake_project(
    const std::string &destination_folder,
    const std::vector<std::vector<BaseScalar>> &local_inputs,
    const std::vector<BaseScalar> &global_input) const {
  if (codegen_ == nullptr) {
    throw std::runtime_error(
        "Target::generate_code() needs to be called before a CMake project can "
        "be created.");
  }
  namespace fs = std::filesystem;
  fs::create_directories(destination_folder);

  std::string base_name = codegen_->name();
  std::string source_ext = codegen_->source_file_extension();

  std::ostringstream main_code;
  main_code << "#include \"" << library_name_ << source_ext << "\"\n";
  main_code << "#include <cmath>\n\n";
  main_code << "using namespace std;\n\n";
  main_code << "int main(int argc, char **argv) {\n";
  main_code << "  const int num_threads = " << local_inputs.size() << ";\n";
  main_code << "  // initialize local inputs\n";
  main_code << "  Float local_inputs["
            << local_inputs.size() * local_inputs[0].size() << "] = {\n    ";
  for (std::size_t i = 0; i < local_inputs.size(); ++i) {
    for (std::size_t j = 0; j < local_inputs[i].size(); ++j) {
      main_code << local_inputs[i][j];
      if (j < local_inputs[i].size() - 1) {
        main_code << ", ";
      }
    }
    if (i < local_inputs.size() - 1) {
      main_code << ",\n    ";
    } else {
      main_code << "\n";
    }
  }
  main_code << "  };\n";
  if (!global_input.empty()) {
    main_code << "  // initialize global input\n";
    main_code << "  Float global_input[" << global_input.size()
              << "] = {\n    ";
    for (std::size_t i = 0; i < global_input.size(); ++i) {
      main_code << global_input[i];
      if (i < global_input.size() - 1) {
        main_code << ", ";
      }
    }
    main_code << "\n  };\n";
  }
  if (this->generate_forward_) {
    std::string fun_name = base_name + "_forward_zero";
    codegen_->emit_cpp_function_call_block(main_code, fun_name, output_dim(),
                                           !global_input.empty());
  }
  if (this->generate_jacobian_) {
    std::string fun_name = base_name + "_jacobian";
    codegen_->emit_cpp_function_call_block(
        main_code, fun_name, input_dim() * output_dim(), !global_input.empty());
  }
  main_code << "  return 0;\n}\n";

  {
    std::ofstream file(fs::path(destination_folder) / std::string("main.cpp"));
    file << main_code.str();
    file.close();
  }

  {
    std::ostringstream cmake_code;
    codegen_->emit_cmake_code(cmake_code, base_name);
    std::ofstream file(fs::path(destination_folder) /
                       std::string("CMakeLists.txt"));
    file << cmake_code.str();
    file.close();
  }

  for (const auto &entry : sources_) {
    std::ofstream file(fs::path(destination_folder) / entry.first);
    file << entry.second;
    file.close();
  }

  std::string abs_folder = FileUtils::abs_path(destination_folder);
  std::cout << "Created CMake project files at \"" << abs_folder << "\""
            << std::endl;
  return true;
}
}  // namespace autogen