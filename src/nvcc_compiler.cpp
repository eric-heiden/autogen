#include "autogen/cg/compiler/nvcc_compiler.h"

namespace autogen {

NvccCompiler::NvccCompiler(const std::string& compilerPath, int optimization_level)
    : AbstractCCompiler<BaseScalar>(compilerPath) {
  this->_compileFlags.push_back("/O2");         // Optimization level
  this->_compileFlags.push_back("/nologo");     // Suppress startup banner
  this->_compileLibFlags.push_back("/O2");      // Optimization level
  this->_compileLibFlags.push_back("/nologo");  // Suppress startup banner

  this->_compileFlags.push_back("--ptxas-options=-O" +
                                std::to_string(optimization_level));
  this->_compileFlags.push_back(",-v ");
  this->_compileFlags.push_back("--ptxas-options=-v ");
  this->_compileFlags.push_back("-rdc=true ");
  // if (debug_mode_) {
  this->_compileFlags.push_back("-G ");
  // }
}
}  // namespace autogen