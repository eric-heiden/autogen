namespace autogen {

template <class Base>
void CudaModelSourceGen<Base>::generateSparseReverseOneSourcesWithAtomics(
    const std::map<size_t, std::vector<size_t>>& elements,
    std::ostringstream& code,
    std::vector<std::pair<std::string, std::string>>& sources) {
  using std::vector;

  /**
   * Generate one function for each dependent variable
   */
  size_t m = this->_fun.Range();
  size_t n = this->_fun.Domain();

  vector<CGBase> w(m);

  /**
   * Generate one function for each dependent variable
   */
  const std::string jobName = "model (reverse one)";
  this->startingJob("'" + jobName + "'",
                    CppAD::cg::JobTimer::SOURCE_GENERATION);

  for (const auto& it : elements) {
    size_t i = it.first;
    const std::vector<size_t>& cols = it.second;

    this->_cache.str("");
    this->_cache << "model (reverse one, dep " << i << ")";
    const std::string subJobName = this->_cache.str();

    this->startingJob("'" + subJobName + "'", CppAD::cg::JobTimer::GRAPH);

    CppAD::cg::CodeHandler<Base> handler;
    handler.setJobTimer(this->_jobTimer);

    vector<CGBase> indVars(this->_fun.Domain());
    handler.makeVariables(indVars);
    if (this->_x.size() > 0) {
      for (size_t i = 0; i < n; i++) {
        indVars[i].setValue(this->_x[i]);
      }
    }

    CGBase py;
    handler.makeVariable(py);
    if (this->_x.size() > 0) {
      py.setValue(Base(1.0));
    }

    // TODO: consider caching the zero order coefficients somehow between calls
    this->_fun.Forward(0, indVars);

    w[i] = py;
    vector<CGBase> dw = this->_fun.Reverse(1, w);
    CPPADCG_ASSERT_UNKNOWN(dw.size() == n);
    w[i] = Base(0);

    vector<CGBase> dwCustom;
    for (size_t it2 : cols) {
      dwCustom.push_back(dw[it2]);
    }

    this->finishedJob();

    std::ostringstream fun_body;

    LanguageCuda<Base> langC(false);
    langC.setMaxAssignmentsPerFunction(this->_maxAssignPerFunc,
                                       &this->_sources);
    langC.setMaxOperationsPerAssignment(this->_maxOperationsPerAssignment);
    langC.setParameterPrecision(this->_parameterPrecision);
    // this->_cache.str("");
    // this->_cache << this->_name << "_sparse_rev1_indep" << j;
    // langC.setGenerateFunction(this->_cache.str());
    langC.setGenerateFunction("");

    // CudaVariableNameGenerator<Base> nameGen(global_input_dim_);
    // handler.generateCode(fun_body, langC, dwCustom, nameGen,
    //                      this->_atomicFunctions, subJobName);

    std::unique_ptr<CppAD::cg::VariableNameGenerator<Base>> nameGen(
        this->createVariableNameGenerator("dw"));
    CppAD::cg::LangCDefaultHessianVarNameGenerator<Base> nameGenHess(
        nameGen.get(), "py", n);
    handler.generateCode(fun_body, langC, dwCustom, nameGenHess,
                         this->_atomicFunctions, subJobName);
    langC.print_constants(fun_body);

    std::string fun_name = std::string(this->_name) +
                           "_sparse_reverse_one_dep" + std::to_string(i);
    CudaFunctionSourceGen generator(fun_name, local_input_dim(),
                                    global_input_dim_, output_dim(),
                                    ACCUMULATE_NONE);
    generator.is_reverse_one = true;

    std::ostringstream complete;
    // complete << "__device__\n";
    // complete << fun_body.str();

    if (!kernel_only_) {
      generator.emit_header(complete);
    }
    //   LanguageCuda<Base> langC;
    generator.emit_kernel(complete, fun_body, langC, kernel_only_);
    if (!kernel_only_) {
      generator.emit_allocation_functions(complete);
      generator.emit_send_functions(complete);
      generator.emit_kernel_launch(complete);
    }

    std::string filename = fun_name + ".cuh";
    sources.push_back(std::make_pair(filename, complete.str()));

    code << "#include \"" << filename << "\"\n";
  }
}

template <class Base>
void CudaModelSourceGen<Base>::generateSparseReverseOneSourcesNoAtomics(
    const std::map<size_t, std::vector<size_t>>& elements,
    std::ostringstream& code,
    std::vector<std::pair<std::string, std::string>>& sources) {
  using std::vector;

  /**
   * Jacobian
   */
  size_t m = this->_fun.Range();
  size_t n = this->_fun.Domain();

  CppAD::cg::CodeHandler<Base> handler;
  handler.setJobTimer(this->_jobTimer);

  vector<CGBase> x(n);
  handler.makeVariables(x);
  if (this->_x.size() > 0) {
    for (size_t i = 0; i < n; i++) {
      x[i].setValue(this->_x[i]);
    }
  }

  CGBase py;
  handler.makeVariable(py);
  if (this->_x.size() > 0) {
    py.setValue(Base(1.0));
  }

  vector<CGBase> jacFlat(this->_jacSparsity.rows.size());

  CppAD::sparse_jacobian_work work;  // temporary structure for CPPAD
  this->_fun.SparseJacobianReverse(x, this->_jacSparsity.sparsity,
                                   this->_jacSparsity.rows,
                                   this->_jacSparsity.cols, jacFlat, work);

  /**
   * organize results
   */
  std::map<size_t, vector<CGBase>> jac;                // by row
  std::vector<std::map<size_t, size_t>> positions(m);  // by row

  for (const auto& it : elements) {
    size_t i = it.first;
    const std::vector<size_t>& row = it.second;

    jac[i].resize(row.size());
    std::map<size_t, size_t>& pos = positions[i];

    for (size_t e = 0; e < row.size(); e++) {
      size_t j = row[e];
      pos[j] = e;
    }
  }

  for (size_t el = 0; el < this->_jacSparsity.rows.size(); el++) {
    size_t i = this->_jacSparsity.rows[el];
    size_t j = this->_jacSparsity.cols[el];
    size_t e = positions[i].at(j);

    vector<CGBase>& row = jac[i];
    row[e] = jacFlat[el] * py;
  }

  /**
   * Create source for each equation/row
   */
  typename std::map<size_t, vector<CGBase>>::iterator itI;
  for (itI = jac.begin(); itI != jac.end(); ++itI) {
    size_t i = itI->first;
    vector<CGBase>& dwCustom = itI->second;

    this->_cache.str("");
    this->_cache << "model (reverse one, dep " << i << ")";
    const std::string subJobName = this->_cache.str();

    std::ostringstream fun_body;

    LanguageCuda<Base> langC;
    langC.setMaxAssignmentsPerFunction(this->_maxAssignPerFunc,
                                       &this->_sources);
    langC.setMaxOperationsPerAssignment(this->_maxOperationsPerAssignment);
    langC.setParameterPrecision(this->_parameterPrecision);
    // this->_cache.str("");
    // this->_cache << this->_name << "_sparse_rev1_indep" << j;
    langC.setGenerateFunction("");  // this->_cache.str());

    // CudaVariableNameGenerator<Base> nameGen(global_input_dim_);
    // handler.generateCode(fun_body, langC, dwCustom, nameGen,
    //                      this->_atomicFunctions, subJobName);

    std::unique_ptr<CppAD::cg::VariableNameGenerator<Base>> nameGen(
        this->createVariableNameGenerator("dw"));
    CppAD::cg::LangCDefaultHessianVarNameGenerator<Base> nameGenHess(
        nameGen.get(), "py", n);
    handler.generateCode(fun_body, langC, dwCustom, nameGenHess,
                         this->_atomicFunctions, subJobName);
    langC.print_constants(fun_body);

    // std::cout << code.str() << std::endl;

    std::string fun_name = std::string(this->_name) +
                           "_sparse_reverse_one_dep" + std::to_string(i);
    CudaFunctionSourceGen generator(fun_name, local_input_dim(),
                                    global_input_dim_, output_dim(),
                                    ACCUMULATE_NONE);
    generator.is_reverse_one = true;

    std::ostringstream complete;

    if (!kernel_only_) {
      generator.emit_header(complete);
    }
    //   LanguageCuda<Base> langC;
    generator.emit_kernel(complete, fun_body, langC, kernel_only_);
    if (!kernel_only_) {
      generator.emit_allocation_functions(complete);
      generator.emit_send_functions(complete);
      generator.emit_kernel_launch(complete);
    }

    std::string filename = fun_name + ".cuh";
    sources.push_back(std::make_pair(filename, complete.str()));

    code << "#include \"" << filename << "\"\n";
  }
}

std::string directional_reverse_function_source(
    const std::string& function,
    const std::map<size_t, std::vector<size_t>>& elements, size_t input_dim) {
  std::stringstream code;
  std::string fun_title = "int " + function + "_sparse_reverse_one(";
  code << "__device__\n";
  code << fun_title << "unsigned long pos,\n";
  code << std::string(fun_title.size(), ' ') << "Float* out,\n";
  code << std::string(fun_title.size(), ' ') << "const Float* x,\n";
  // code << std::string(fun_title.size(), ' ') << "const Float* ty,\n";
  code << std::string(fun_title.size(), ' ') << "const Float* py) {\n";
  // code << "  Float compressed[" << input_dim << "];\n";
  // code << "  unsigned long const* idx;\n";
  // code << "  unsigned long nnz;\n\n";

  code << "  switch(pos) {\n";
  for (const auto& it : elements) {
    // the size of each sparsity row
    code << "    case " << it.first
         << ":\n"
            "      "
         << function << "_sparse_reverse_one_dep"
         << it.first
         //<< "(compressed, x, dx);\n"
         << "(out, x, py);\n"
            "      return 0;\n";
  }
  code
      << "    default:\n"
         "      // printf(\"Error: cannot compute functional derivative %u for "
         "\\\""
      << function << "_sparse_reverse_one\\\".\\n\", pos);\n";
  code << "      return 1;\n"
          "  };\n";

  // code << "  " << function << "_reverse_one_sparsity(pos, &idx, &nnz);\n\n";

  // code << "  for (unsigned long ePos = 0; ePos < nnz; ePos++) {\n";
  // code << "    out[idx[ePos]] += compressed[ePos];\n";
  // code << "  }\n";

  code << "}\n";
  return code.str();
}

/**
 * Generate CUDA library code for the reverse one pass.
 */
template <class Base>
std::string CudaModelSourceGen<Base>::reverse_one_source(
    std::vector<std::pair<std::string, std::string>>& sources) {
  const std::string jobName = "model (first-order reverse)";

  this->startingJob("'" + jobName + "'", CppAD::cg::JobTimer::GRAPH);

  CppAD::cg::CodeHandler<Base> handler;
  handler.setJobTimer(this->_jobTimer);

  if (global_input_dim_ > this->_fun.Domain()) {
    throw std::runtime_error(
        "CUDA codegen failed: global data input size must not be "
        "larger than the provided input vector size.");
  }

  const std::size_t local_input_dim = this->_fun.Domain() - global_input_dim_;
  const std::size_t output_dim = this->_fun.Range();

  std::cout << "Generating first-order reverse code for function \""
            << this->_name << "\" with input dimension " << local_input_dim
            << " and output dimension " << output_dim << "...\n";

  //   std::vector<CGBase> indVars(local_input_dim + global_input_dim_);
  //   handler.makeVariables(indVars);
  //   if (this->_x.size() > 0) {
  //     for (std::size_t i = 0; i < indVars.size(); i++) {
  //       indVars[i].setValue(this->_x[i]);
  //     }
  //   }

  //   std::vector<CGBase> dep;

  this->determineJacobianSparsity();
  // elements[var]{equations}
  std::map<size_t, std::vector<size_t>> elements;
  for (size_t e = 0; e < this->_jacSparsity.rows.size(); e++) {
    elements[this->_jacSparsity.rows[e]].push_back(this->_jacSparsity.cols[e]);
  }

  // std::cout << this->_name << " uses atomics? " << std::boolalpha
  //           << this->isAtomicsUsed() << std::endl;

  std::ostringstream code;

  if (this->isAtomicsUsed()) {
    generateSparseReverseOneSourcesWithAtomics(elements, code, sources);
  } else {
    generateSparseReverseOneSourcesNoAtomics(elements, code, sources);
  }

  code << "\n";

  const std::string sparsity_function =
      std::string(this->_name) + "_reverse_one_sparsity";
  this->_cache.str("");
  this->generateSparsity1DSource2(sparsity_function, elements);
  code << "\n__device__\n" << this->_cache.str() << "\n";

  code << directional_reverse_function_source(this->_name, elements,
                                              this->_fun.Domain());

  size_t m = this->_fun.Range();
  size_t n = this->_fun.Domain();

  std::string sparse_rev1_function =
      std::string(this->_name) + "_sparse_reverse_one";
  const std::string model_function = std::string(this->_name) + "_reverse_one";

  // logic from functor_generic_model.hpp:466 (reverseOne)

  code << "\n__device__\n";
  LanguageCuda<Base>::printFunctionDeclaration(
      code, "int", model_function,
      {"Float *out", "const Float *x", "const Float *py", "unsigned long nnzTx",
       "const unsigned long *idx"});
  code << " {\n"
       << "  unsigned long ePos, ej, j, nnz;\n"
       << "  int ret;\n"
       << "  unsigned long const* pos;\n"
       << "  Float compressed[" << n << "];\n"
       << "  const Float *py_in;\n\n";

  if (LanguageCuda<Base>::add_debug_prints) {
    code << "  printf(\"" << model_function
         << " idx:  \"); for (ej = 0; ej < nnzTx; ej++) printf(\"%u  \", "
            "idx[ej]); printf(\"\\n\");\n";
  }

  code << "  for (ej = 0; ej < nnzTx; ej++) {\n"
       << "    j = idx[ej];\n"
       << "    " << sparsity_function << "(j, &pos, &nnz);\n"
       << "    if (nnz == 0) continue;\n"
       << "    py_in = &py[ej];\n"
       << "    ret = " << sparse_rev1_function
       << "(j, compressed, x, py_in);\n\n"
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