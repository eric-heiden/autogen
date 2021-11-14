#pragma once

#include "autogen/core/base.hpp"

namespace autogen {
class NvccCompiler : public CppAD::cg::AbstractCCompiler<BaseScalar> {
 public:
  NvccCompiler(const std::string& compilerPath = "/usr/bin/nvcc",
               int optimization_level = 2);

  NvccCompiler(const NvccCompiler& orig) = delete;
  NvccCompiler& operator=(const NvccCompiler& rhs) = delete;

  /**
   * Creates a dynamic library from a set of object files
   *
   * @param library the path to the dynamic library to be created
   */
  void buildDynamic(const std::string& library,
                    CppAD::cg::JobTimer* timer = nullptr) override {
    std::vector<std::string> args;
    args.push_back("/DLL");  // Create a DLL (no debug symbols)
    args.push_back("/out:\"" + library + "\"");  // Output file name
    args.insert(args.end(), this->_compileLibFlags.begin(),
                this->_compileLibFlags.end());
    for (const std::string& it : this->_ofiles) {
      args.push_back(it);
    }

    if (timer != nullptr) {
      timer->startingJob("'" + library + "'",
                         CppAD::cg::JobTimer::COMPILING_DYNAMIC_LIBRARY);
    } else if (this->_verbose) {
      std::cout << "building library '" << library << "'" << std::endl;
    }

#if AUTOGEN_SYSTEM_WIN
    this->_compileFlags.push_back("-o " + library + ".dll ");
#else
    this->_compileFlags.push_back("--compiler-options ");
    this->_compileFlags.push_back("-fPIC ");
    this->_compileFlags.push_back("-o " + library + ".so ");
#endif
    this->_compileFlags.push_back("--shared ");

    // TODO implement separate linking process
    // std::string out;
    // int return_code = CppAD::cg::system::callExecutable(this->_linkerPath,
    // args, &out); if (return_code) {
    //   std::cerr << "\nError (" << return_code
    //             << ") linking library '" + library + "' via MSVC:\n" + out
    //             << std::endl;
    //   throw CppAD::cg::CGException("Error linking library '" + library +
    //                                "' via MSVC:\n" + out);
    // }

    if (timer != nullptr) {
      timer->finishedJob();
    }
  }

  virtual ~NvccCompiler() = default;

 protected:
  /**
   * Compiles a single source file into an object file
   *
   * @param source the content of the source file
   * @param output the compiled output file name (the object file path)
   */
  void compileSource(const std::string& source, const std::string& output,
                     bool posIndepCode) override {
    throw CppAD::cg::CGException(
        "MSVC cannot compile source files from stdin, "
        "make sure to save them to disk first via "
        "the corresponding option in CppADCodeGen.");
  }

  void compileFile(const std::string& path, const std::string& output,
                   bool posIndepCode) override {
    std::vector<std::string> args;
    args.push_back("/LD");  // Prepare for DLL linking
    args.push_back("/c");   // Compile without linking
    args.push_back(path);
    args.insert(args.end(), this->_compileFlags.begin(),
                this->_compileFlags.end());
    args.push_back("/Fo\"" + output + "\"");

    std::string out;
    int return_code =
        CppAD::cg::system::callExecutable(this->_path, args, &out);
    if (return_code) {
      std::cerr << "\nError (" << return_code
                << ") compiling file '" + path + "' via MSVC:\n" + out
                << std::endl;
      throw CppAD::cg::CGException("Error compiling file '" + path +
                                   "' via MSVC:\n" + out);
    }
  }
};
}  // namespace autogen