#pragma once

#include "autogen/cg/compact/compact_codegen.h"

namespace autogen {
void CompactCodeGen::generateSparseForwardOneSourcesWithAtomics(
    const std::map<size_t, std::vector<size_t>>& elements,
    std::ostringstream& code,
    std::vector<std::pair<std::string, std::string>>& sources) {
  using std::vector;

  /**
   * Generate one function for each dependent variable
   */
  size_t n = fun_->Domain();

  vector<CGBase> dxv(n);

  const std::string jobName = "model (forward one)";
  model_source_gen_->startingJob("'" + jobName + "'",
                                 CppAD::cg::JobTimer::SOURCE_GENERATION);

  for (const auto& it : elements) {
    size_t j = it.first;
    const std::vector<size_t>& rows = it.second;

    cache_.str("");
    cache_ << "model (forward one, indep " << j << ")";
    const std::string subJobName = cache_.str();

    model_source_gen_->startingJob("'" + subJobName + "'",
                                   CppAD::cg::JobTimer::GRAPH);

    CppAD::cg::CodeHandler<BaseScalar> handler;
    handler.setJobTimer(model_source_gen_->getJobTimer());

    vector<CGBase> indVars(n);
    handler.makeVariables(indVars);
    const auto& x = model_source_gen_->getTypicalIndependentValues();
    if (x.size() > 0) {
      for (size_t i = 0; i < n; i++) {
        indVars[i].setValue(x[i]);
      }
    }

    CGBase dx;
    handler.makeVariable(dx);
    if (x.size() > 0) {
      dx.setValue(BaseScalar(1.0));
    }

    // TODO: consider caching the zero order coefficients somehow between calls
    fun_->Forward(0, indVars);
    dxv[j] = dx;
    vector<CGBase> dy = fun_->Forward(1, dxv);
    dxv[j] = BaseScalar(0);
    CPPADCG_ASSERT_UNKNOWN(dy.size() == fun_->Range());

    vector<CGBase> dyCustom;
    for (size_t it2 : rows) {
      dyCustom.push_back(dy[it2]);
    }

    model_source_gen_->finishedJob();

    std::ostringstream fun_body;

    LanguageCompact langC(false);
    langC.setMaxAssignmentsPerFunction(model_source_gen_->_maxAssignPerFunc,
                                       &model_source_gen_->_sources);
    langC.setMaxOperationsPerAssignment(
        model_source_gen_->_maxOperationsPerAssignment);
    langC.setParameterPrecision(model_source_gen_->_parameterPrecision);
    // cache_.str("");
    // cache_ << model_name_ << "_sparse_for1_indep" << j;
    // langC.setGenerateFunction(cache_.str());
    langC.setGenerateFunction("");

    // CudaVariableNameGenerator<BaseScalar> nameGen(global_input_dim_);
    // handler.generateCode(fun_body, langC, dyCustom, nameGen,
    //                      this->_atomicFunctions, subJobName);

    std::unique_ptr<CppAD::cg::VariableNameGenerator<BaseScalar>> nameGen(
        model_source_gen_->createVariableNameGenerator("dy"));
    CppAD::cg::LangCDefaultHessianVarNameGenerator<BaseScalar> nameGenHess(
        nameGen.get(), "dx", n);
    handler.generateCode(fun_body, langC, dyCustom, nameGenHess,
                         model_source_gen_->getAtomicFunctions(), subJobName);
    langC.print_constants(fun_body);

    std::string function_name = std::string(model_name_) +
                                "_sparse_forward_one_indep" + std::to_string(j);
    size_t out_dim = output_dim();  // TODO verify
    bool is_forward_one = true;
    bool is_reverse_one = false;

    std::ostringstream complete;
    complete << "#include \"util.h\"\n\n";

    if (!function_only_) {
      emit_header(complete, function_name, local_input_dim(),
                  global_input_dim(), out_dim, ACCUMULATE_NONE);
    }
    //   LanguageCompact langC;
    emit_function(complete, function_name, local_input_dim(),
                  global_input_dim(), out_dim, fun_body, langC, function_only_,
                  is_forward_one, is_reverse_one);
    if (!function_only_) {
      emit_allocation_functions(complete, function_name, local_input_dim(),
                                global_input_dim(), out_dim);
      emit_send_functions(complete, function_name, local_input_dim(),
                          global_input_dim(), out_dim);
      emit_kernel_launch(complete, function_name, local_input_dim(),
                         global_input_dim(), out_dim);
    }

    std::string filename = function_name + header_file_extension();
    sources.push_back(std::make_pair(filename, complete.str()));

    code << "#include \"" << filename << "\"\n";
  }
}

void CompactCodeGen::generateSparseForwardOneSourcesNoAtomics(
    const std::map<size_t, std::vector<size_t>>& elements,
    std::ostringstream& code,
    std::vector<std::pair<std::string, std::string>>& sources) {
  using std::vector;

  /**
   * Jacobian
   */
  size_t n = fun_->Domain();

  CppAD::cg::CodeHandler<BaseScalar> handler;
  handler.setJobTimer(model_source_gen_->getJobTimer());

  const auto& typical_x = model_source_gen_->getTypicalIndependentValues();

  vector<CGBase> x(n);
  handler.makeVariables(x);
  if (typical_x.size() > 0) {
    for (size_t i = 0; i < n; i++) {
      x[i].setValue(typical_x[i]);
    }
  }

  CGBase dx;
  handler.makeVariable(dx);
  if (typical_x.size() > 0) {
    dx.setValue(BaseScalar(1.0));
  }

  const auto& sparsity = model_source_gen_->getJacobianSparsity();
  vector<CGBase> jacFlat(sparsity.rows.size());

  CppAD::sparse_jacobian_work work;  // temporary structure for CPPAD
  fun_->SparseJacobianForward(x, sparsity.sparsity, sparsity.rows,
                              sparsity.cols, jacFlat, work);

  /**
   * organize results
   */
  std::map<size_t, vector<CGBase>> jac;                  // by column
  std::map<size_t, std::map<size_t, size_t>> positions;  // by column

  for (const auto& it : elements) {
    size_t j = it.first;
    const std::vector<size_t>& column = it.second;

    jac[j].resize(column.size());
    std::map<size_t, size_t>& pos = positions[j];

    for (size_t e = 0; e < column.size(); e++) {
      size_t i = column[e];
      pos[i] = e;
    }
  }

  for (size_t el = 0; el < sparsity.rows.size(); el++) {
    size_t i = sparsity.rows[el];
    size_t j = sparsity.cols[el];
    size_t e = positions[j].at(i);

    vector<CGBase>& column = jac[j];
    column[e] = jacFlat[el] * dx;
  }

  /**
   * Create source for each independent/column
   */
  typename std::map<size_t, vector<CGBase>>::iterator itJ;
  for (itJ = jac.begin(); itJ != jac.end(); ++itJ) {
    size_t j = itJ->first;
    vector<CGBase>& dyCustom = itJ->second;

    cache_.str("");
    cache_ << "model (forward one, indep " << j << ")";
    const std::string subJobName = cache_.str();

    std::ostringstream fun_body;

    LanguageCompact langC;
    langC.setMaxAssignmentsPerFunction(model_source_gen_->_maxAssignPerFunc,
                                       &model_source_gen_->_sources);
    langC.setMaxOperationsPerAssignment(
        model_source_gen_->_maxOperationsPerAssignment);
    langC.setParameterPrecision(model_source_gen_->_parameterPrecision);
    // cache_.str("");
    // cache_ << model_name_ << "_sparse_for1_indep" << j;
    langC.setGenerateFunction("");  // cache_.str());

    // CudaVariableNameGenerator<BaseScalar> nameGen(global_input_dim_);
    // handler.generateCode(fun_body, langC, dyCustom, nameGen,
    //                      this->_atomicFunctions, subJobName);

    std::unique_ptr<CppAD::cg::VariableNameGenerator<BaseScalar>> nameGen(
        model_source_gen_->createVariableNameGenerator("dy"));
    CppAD::cg::LangCDefaultHessianVarNameGenerator<BaseScalar> nameGenHess(
        nameGen.get(), "dx", n);
    handler.generateCode(fun_body, langC, dyCustom, nameGenHess,
                         model_source_gen_->getAtomicFunctions(), subJobName);
    langC.print_constants(fun_body);

    // std::cout << code.str() << std::endl;

    std::string function_name = std::string(model_name_) +
                                "_sparse_forward_one_indep" + std::to_string(j);
    size_t out_dim = output_dim();  // TODO verify
    bool is_forward_one = true;
    bool is_reverse_one = false;

    std::ostringstream complete;

    if (!function_only_) {
      emit_header(complete, function_name, local_input_dim(),
                  global_input_dim(), out_dim, ACCUMULATE_NONE);
    }
    //   LanguageCompact langC;
    emit_function(complete, function_name, local_input_dim(),
                  global_input_dim(), out_dim, fun_body, langC, function_only_,
                  is_forward_one, is_reverse_one);
    if (!function_only_) {
      emit_allocation_functions(complete, function_name, local_input_dim(),
                                global_input_dim(), out_dim);
      emit_send_functions(complete, function_name, local_input_dim(),
                          global_input_dim(), out_dim);
      emit_kernel_launch(complete, function_name, local_input_dim(),
                         global_input_dim(), out_dim);
    }

    std::string filename = function_name + header_file_extension();
    sources.push_back(std::make_pair(filename, complete.str()));

    code << "#include \"" << filename << "\"\n";
  }
}

std::string CompactCodeGen::directional_forward_function_source(
    const std::string& function,
    const std::map<size_t, std::vector<size_t>>& elements,
    size_t input_dim) const {
  std::stringstream code;
  std::string fun_title = "int " + function + "_sparse_forward_one(";
  code << function_type_prefix(true);
  code << fun_title << "unsigned long pos,\n";
  code << std::string(fun_title.size(), ' ') << "Float* out,\n";
  code << std::string(fun_title.size(), ' ') << "const Float* x,\n";
  code << std::string(fun_title.size(), ' ') << "const Float* dx) {\n";
  // code << "  Float compressed[" << input_dim << "];\n";
  // code << "  unsigned long const* idx;\n";
  // code << "  unsigned long nnz;\n\n";

  code << "  switch(pos) {\n";
  for (const auto& it : elements) {
    // the size of each sparsity row
    code << "    case " << it.first
         << ":\n"
            "      "
         << function << "_sparse_forward_one_indep"
         << it.first
         //<< "(compressed, x, dx);\n"
         << "(out, x, dx);\n"
            "      return 0;\n";
  }
  code
      << "    default:\n"
         "      // printf(\"Error: cannot compute functional derivative %u for "
         "\\\""
      << function << "_sparse_forward_one\\\".\\n\", pos);\n";
  code << "      return 1;\n"
          "  };\n";

  // code << "  " << function << "_forward_one_sparsity(pos, &idx, &nnz);\n\n";

  // code << "  for (unsigned long ePos = 0; ePos < nnz; ePos++) {\n";
  // code << "    out[idx[ePos]] += compressed[ePos];\n";
  // code << "  }\n";

  code << "}\n";
  return code.str();
}

/**
 * Generate CUDA library code for the forward one pass.
 */
std::string CompactCodeGen::forward_one_source(
    std::vector<std::pair<std::string, std::string>>& sources) {
  const std::string jobName = "model (first-order forward)";

  model_source_gen_->startingJob("'" + jobName + "'",
                                 CppAD::cg::JobTimer::GRAPH);

  CppAD::cg::CodeHandler<BaseScalar> handler;
  handler.setJobTimer(model_source_gen_->getJobTimer());

  if (global_input_dim_ > fun_->Domain()) {
    throw std::runtime_error(
        "CUDA codegen failed: global data input size must not be "
        "larger than the provided input vector size.");
  }

  const std::size_t local_input_dim = fun_->Domain() - global_input_dim_;
  const std::size_t out_dim = fun_->Range();

#ifdef DEBUG
  std::cout << "Generating first-order forward code for function \""
            << model_name_ << "\" with input dimension " << local_input_dim
            << " and output dimension " << out_dim << "...\n";
#endif

  //   std::vector<CGBase> indVars(local_input_dim + global_input_dim_);
  //   handler.makeVariables(indVars);
  //   if (this->_x.size() > 0) {
  //     for (std::size_t i = 0; i < indVars.size(); i++) {
  //       indVars[i].setValue(this->_x[i]);
  //     }
  //   }

  //   std::vector<CGBase> dep;

  model_source_gen_->determineJacobianSparsity();
  const auto& sparsity = model_source_gen_->getJacobianSparsity();
  // elements[var]{equations}
  std::map<size_t, std::vector<size_t>> elements;
  for (size_t e = 0; e < sparsity.rows.size(); e++) {
    elements[sparsity.cols[e]].push_back(sparsity.rows[e]);
  }

  // std::cout << model_name_ << " uses atomics? " << std::boolalpha
  //           << this->isAtomicsUsed() << std::endl;

  std::ostringstream code;
  code << "#include \"util.h\"\n\n";

  if (model_source_gen_->isAtomicsUsed()) {
    generateSparseForwardOneSourcesWithAtomics(elements, code, sources);
  } else {
    generateSparseForwardOneSourcesNoAtomics(elements, code, sources);
  }

  code << "\n";

  const std::string sparsity_function =
      std::string(model_name_) + "_forward_one_sparsity";
  model_source_gen_->_cache.str("");
  model_source_gen_->generateSparsity1DSource2(sparsity_function, elements);
  code << "\n"
       << function_type_prefix(true) << model_source_gen_->_cache.str() << "\n";

  code << directional_forward_function_source(model_name_, elements,
                                              fun_->Domain());

  size_t m = fun_->Range();
  size_t n = fun_->Domain();

  std::string sparse_for1_function =
      std::string(model_name_) + "_sparse_forward_one";
  const std::string model_function = std::string(model_name_) + "_forward_one";

  // logic from functor_generic_model.hpp:466 (ForwardOne)

  code << "\n" << function_type_prefix(true);
  LanguageCompact::printFunctionDeclaration(
      code, "int", model_function,
      {"Float *out", "const Float *x", "const Float *dx", "unsigned long nnzTx",
       "const unsigned long *idx"});
  code << " {\n"
       << "  unsigned long ePos, ej, j, nnz;\n"
       << "  int ret;\n"
       << "  unsigned long const* pos;\n"
       << "  Float compressed[" << m << "];\n"
       << "  const Float *dx_in;\n\n";

  if (LanguageCompact::add_debug_prints) {
    emit_debug_print(code, model_function, "idx", "nnzTx");
    // code << "  printf(\"" << model_function
    //      << " idx:  \"); for (ej = 0; ej < nnzTx; ej++) printf(\"%u  \", "
    //         "idx[ej]); printf(\"\\n\");\n";
  }

  code << "  for (ej = 0; ej < nnzTx; ej++) {\n"
       << "    j = idx[ej];\n"
       << "    " << sparsity_function << "(j, &pos, &nnz);\n"
       << "    if (nnz == 0) continue;\n"
       << "    dx_in = &dx[ej];\n"
       << "    ret = " << sparse_for1_function
       << "(j, compressed, x, dx_in);\n\n"
       << "    if (ret != 0) return ret;\n"
       << "    for (ePos = 0; ePos < nnz; ePos++) {\n"
       << "      if (ej == 0) out[pos[ePos]] = 0;\n"
       << "      out[pos[ePos]] += compressed[ePos];\n"
       << "    }\n"
       << "  }\n"
       << "  return 0;\n"
       << "}\n";

  return code.str();
}

}  // namespace autogen