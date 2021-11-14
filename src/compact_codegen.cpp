#include "autogen/cg/compact/compact_codegen.h"

#include <regex>

// clang-format off
#include "compact_model_source_gen.cpp"
#include "compact_codegen_for0.cpp"
#include "compact_codegen_for1.cpp"
#include "compact_codegen_jacobian.cpp"
#include "compact_codegen_rev1.cpp"
#include "compact_codegen_sparse_jac.cpp"
// clang-format on

namespace autogen {
const std::string &CompactCodeGen::base_type_name() const {
  assert_model_source_gen();
  return model_source_gen_->getBaseTypeName();
}

const std::map<std::string, std::string> &CompactCodeGen::sources() const {
  assert_model_source_gen();
  auto mtt = CppAD::cg::MultiThreadingType::NONE;
  CppAD::cg::JobTimer *timer = nullptr;
  return model_source_gen_->getSources(mtt, timer);
}

bool CompactCodeGen::create_forward_zero() const {
  return model_source_gen_->isCreateForwardZero();
}
bool CompactCodeGen::create_forward_zero(bool v) {
  model_source_gen_->setCreateForwardZero(v);
  return model_source_gen_->isCreateForwardZero();
}

bool CompactCodeGen::create_forward_one() const {
  return model_source_gen_->isCreateSparseForwardOne();
}
bool CompactCodeGen::create_forward_one(bool v) {
  model_source_gen_->setCreateForwardOne(v);
  return model_source_gen_->isCreateSparseForwardOne();
}

bool CompactCodeGen::create_reverse_one() const {
  return model_source_gen_->isCreateReverseOne();
}
bool CompactCodeGen::create_reverse_one(bool v) {
  model_source_gen_->setCreateReverseOne(v);
  return model_source_gen_->isCreateReverseOne();
}

bool CompactCodeGen::create_sparse_forward_one() const {
  return model_source_gen_->isCreateSparseForwardOne();
}
bool CompactCodeGen::create_sparse_forward_one(bool v) {
  model_source_gen_->setCreateForwardOne(v);
  return model_source_gen_->isCreateSparseForwardOne();
}

bool CompactCodeGen::create_jacobian() const {
  return model_source_gen_->isCreateJacobian();
}
bool CompactCodeGen::create_jacobian(bool v) {
  model_source_gen_->setCreateJacobian(v);
  return model_source_gen_->isCreateJacobian();
}

bool CompactCodeGen::create_sparse_jacobian() const {
  return model_source_gen_->isCreateSparseJacobian();
}
bool CompactCodeGen::create_sparse_jacobian(bool v) {
  model_source_gen_->setCreateSparseJacobian(v);
  return model_source_gen_->isCreateSparseJacobian();
}

void CompactCodeGen::emit_header(std::ostringstream &code,
                                 const std::string &function_name,
                                 size_t local_input_dim,
                                 size_t global_input_dim, size_t output_dim,
                                 AccumulationMethod acc_method) const {
  // meta data retrieval function
  code << "extern \"C\" {\nMODULE_API CompactFunctionMetaData " << function_name
       << "_meta() {\n";
  code << "  CompactFunctionMetaData data;\n";
  code << "  data.output_dim = " << output_dim << ";\n";
  code << "  data.local_input_dim = " << local_input_dim << ";\n";
  code << "  data.global_input_dim = " << global_input_dim << ";\n";
  code << "  data.accumulated_output = " << int(acc_method != ACCUMULATE_NONE)
       << ";\n";
  code << "  return data;\n}\n}\n";
}

void CompactCodeGen::emit_function(
    std::ostringstream &code, const std::string &function_name,
    size_t local_input_dim, size_t global_input_dim, size_t output_dim,
    const std::ostringstream &body, LanguageCompact &language, bool is_function,
    bool is_forward_one, bool is_reverse_one) const {
  std::string kernel_name = function_name;
  if (is_function) {
    code << function_type_prefix(is_function);
  } else {
    code << "\n" << function_type_prefix(is_function);
    kernel_name += "_kernel";
  }
  std::string fun_head_start = "void " + kernel_name + "(";
  std::string fun_arg_pad = std::string(fun_head_start.size(), ' ');
  code << fun_head_start;
  if (!is_function) {
    if (kernel_requires_id_argument_) {
      code << "int ti,\n";
    } else {
      code << "int num_total_threads,\n";
    }
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

    if (add_debug_prints_) {
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

    if (add_debug_prints_) {
      code << "  printf(\"\\t" << kernel_name << ":\\n\");\n";
      code << "  printf(\"\\tx:   \"); for (unsigned long i = 0; i < "
           << local_input_dim
           << ";++i) printf(\"%f  \", x[i]); printf(\"\\n\");\n";
    }
  } else {
    if (!is_function) {
      if (!kernel_requires_id_argument()) {
        emit_thread_id_getter(code);
        code << "  if (ti >= num_total_threads) {\n";
        code << "    printf(\"ERROR: thread index %i in function \\\""
             << function_name
             << "\\\" exceeded provided "
                "number of total threads %i.\\n\", ti, num_total_threads);\n";
        code << "    return;\n  }\n";
      }
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

  // std::cout << "Replacing variables in function " << function_name <<
  // "...\n"; size_t consts = replace_constants(body_str); if (consts > 0) {
  //   std::cout << "Introduced " << consts << " constant variable[s].\n";
  // }
  code << body_str;

  if (add_debug_prints_) {
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
           << output_dim << ";++i) printf(\"%f  \", y[i]); printf(\"\\n\");\n";
    }
  }

  code << "}\n\n";
}

void CompactCodeGen::emit_debug_print(std::ostringstream &code,
                                      const std::string &function_name,
                                      const std::string &var_vec_name,
                                      const std::string &var_size_name,
                                      const std::string &format) const {
  code << "printf(\"" << function_name << "  " << var_vec_name << ":\\n\");\n";
  code << "for (unsigned long debug_i = 0; debug_i < " << var_size_name
       << "; debug_i++) {\n";
  code << "  printf(\"" << format << "  \", " << var_vec_name
       << "[debug_i]);\n";
  code << "}\n";
  code << "printf(\"\\n\");\n";
}

void CompactCodeGen::emit_cmake_code(std::ostringstream &code,
                                     const std::string &project_name) const {
  code << "cmake_minimum_required(VERSION 3.0)\n";
  code << "project(" << project_name << ")\n";
  code << "add_executable(" << project_name << " main.cpp)\n";
}

void CompactCodeGen::emit_cpp_function_call_block(std::ostringstream &code,
                                                  const std::string &fun_name,
                                                  int output_dim,
                                                  bool has_global_input) const {
  code << "  // calling " << fun_name << "\n";
  code << "  {\n";
  code << "    " << fun_name << "_allocate(num_threads);\n";
  code << "    " << fun_name << "_send_local(num_threads, local_inputs);\n";
  if (has_global_input) {
    code << "    " << fun_name << "_send_global(global_input);\n";
  }
  code << "    Float output[" << output_dim << " * num_threads];\n";
  emit_cpp_function_call(code, fun_name);
  code << "    printf(\"" << fun_name << ":\\n\");\n";
  code << "    for (int t = 0; t < num_threads; ++t) {\n";
  code << "      for (int i = 0; i < " << output_dim << "; ++i) {\n";
  code << "        printf(\"%f\", output[t * " << output_dim << " + i]);\n";
  code << "        if (i < " << output_dim - 1 << ") {\n";
  code << "          printf(\", \");\n";
  code << "        }\n";
  code << "      }\n";
  code << "      printf(\"\\n\");\n";
  code << "    }\n";
  code << "    " << fun_name << "_deallocate();\n";
  code << "  }\n";
}

void CompactCodeGen::emit_cpp_function_call(std::ostringstream &code,
                                            const std::string &fun_name) const {
  code << "    " << fun_name << "(num_threads, output);\n";
}

void CompactCodeGen::emit_function_signature(std::ostringstream &code) const {
  if (!function_only_) {
    return;
  }
  if (create_forward_zero()) {
    code << "MODULE_API void " << model_name_;
    code << "_forward_zero(Float *out, const Float *local_input);\n";
  }
  if (create_forward_one()) {
    code << "MODULE_API int " << model_name_;
    code << "_forward_one(Float *out, const Float *x, const Float *dx, "
            "unsigned long nnzTx, const unsigned long *idx);\n";
    code << "MODULE_API void " << model_name_;
    code << "_forward_one_sparsity(unsigned long pos, unsigned long const** "
            "elements, unsigned long* nnz);\n";
  }
  if (create_reverse_one()) {
    code << "MODULE_API int " << model_name_;
    code << "_reverse_one(Float *out, const Float *x, const Float *py, "
            "unsigned long nnzTx, const unsigned long *idx);\n";
    code << "MODULE_API void " << model_name_;
    code << "_reverse_one_sparsity(unsigned long pos, unsigned long const** "
            "elements, unsigned long* nnz);\n";
  }
}

size_t CompactCodeGen::replace_constants(std::string &code) {
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
      ss << "  static const Float " << varname << " = " << value << ";\n";
    }
  }
  ss << code;
  std::cout << "Replaced " << const_id << " constant(s).\n";
  code = ss.str();
  return const_id;
}

void CompactCodeGen::assert_model_source_gen() const {
  if (!model_source_gen_) {
    throw std::runtime_error(
        "Error in CompactCodeGen: the gradient tape has not been set.");
  }
}

}  // namespace autogen