#pragma once

#include "autogen/core/base.hpp"

namespace autogen {
/**
 * C++ code compilation with Clang.
 */
class ClangCompiler : public CppAD::cg::AbstractCCompiler<BaseScalar> {
 protected:
  std::set<std::string> _bcfiles;  // bitcode files
  std::string _version;

 public:
  ClangCompiler(const std::string& clangPath = "/usr/bin/clang++")
      : AbstractCCompiler<BaseScalar>(clangPath) {
    this->_compileFlags.push_back("-O2");         // Optimization level
    this->_compileFlags.push_back("-c");          // compile only
    this->_compileLibFlags.push_back("-O2");      // Optimization level
    this->_compileLibFlags.push_back("-shared");  // Make shared object
#if !CPPAD_CG_SYSTEM_WIN
    // add all symbols to the dynamic symbol table
    this->_compileLibFlags.push_back("-rdynamic");
#endif
  }

  ClangCompiler(const ClangCompiler& orig) = delete;
  ClangCompiler& operator=(const ClangCompiler& rhs) = delete;

  const std::string& getVersion() {
    using namespace CppAD::cg;
    if (_version.empty()) {
      std::vector<std::string> args{"--version"};
      std::string output;
      system::callExecutable(this->_path, args, &output);

      std::string vv = "version ";
      size_t is = output.find(vv);
      if (is == std::string::npos) {
        throw CGException("Failed to determine Clang version");
      }
      is += vv.size();
      size_t i = is;
      while (i < output.size() && output[i] != ' ' && output[i] != '\n') {
        i++;
      }

      _version = output.substr(is, i - is);
    }
    return _version;
  }

  virtual const std::set<std::string>& getBitCodeFiles() const {
    return _bcfiles;
  }

  virtual void generateLLVMBitCode(
      const std::map<std::string, std::string>& sources,
      CppAD::cg::JobTimer* timer = nullptr) {
    bool posIndepCode = false;
    this->_compileFlags.push_back("-emit-llvm");
    try {
      this->compileSources(sources, posIndepCode, timer, ".bc", this->_bcfiles);
    } catch (...) {
      this->_compileFlags.pop_back();
      throw;
    }
  }

  /**
   * Creates a dynamic library from a set of object files
   *
   * @param library the path to the dynamic library to be created
   */
  void buildDynamic(const std::string& library,
                    CppAD::cg::JobTimer* timer = nullptr) override {
    using namespace CppAD::cg;
#if CPPAD_CG_SYSTEM_APPLE
    std::string linkerName =
        ",-install_name," + system::filenameFromPath(library);
#elif CPPAD_CG_SYSTEM_LINUX
    std::string linkerName = ",-soname," + system::filenameFromPath(library);
#else
    std::string linkerName = "";
#endif
    std::string linkerFlags = "-Wl" + linkerName;
    for (size_t i = 0; i < this->_linkFlags.size(); i++)
      linkerFlags += "," + this->_linkFlags[i];

    std::vector<std::string> args;
    args.insert(args.end(), this->_compileLibFlags.begin(),
                this->_compileLibFlags.end());
    args.push_back(linkerFlags);  // Pass suitable options to linker
    args.push_back("-o");         // Output file name
    args.push_back(library);      // Output file name

    std::cout << "\n\nCommand:\n" << this->_path;
    for (const std::string& it : this->_ofiles) {
      args.push_back(it);
      std::cout << " " << it;
    }
    std::cout << std::endl;

    if (timer != nullptr) {
      timer->startingJob("'" + library + "'",
                         JobTimer::COMPILING_DYNAMIC_LIBRARY);
    } else if (this->_verbose) {
      std::cout << "building library '" << library << "'" << std::endl;
    }

    system::callExecutable(this->_path, args);

    if (timer != nullptr) {
      timer->finishedJob();
    }
  }

  void cleanup() override {
    using namespace CppAD::cg;
    // clean up
    for (const std::string& it : _bcfiles) {
      if (remove(it.c_str()) != 0)
        std::cerr << "Failed to delete temporary file '" << it << "'"
                  << std::endl;
    }
    _bcfiles.clear();

    // other files and temporary folder
    AbstractCCompiler<BaseScalar>::cleanup();
  }

  virtual ~ClangCompiler() { cleanup(); }

  static std::vector<std::string> parseVersion(const std::string& version) {
    using namespace CppAD::cg;
    auto vv = explode(version, ".");
    if (vv.size() > 2) {
      auto vv2 = explode(vv[2], "-");
      if (vv2.size() > 1) {
        vv.erase(vv.begin() + 2);
        vv.insert(vv.begin() + 2, vv2.begin(), vv2.end());
      }
    }
    return vv;
  }

 protected:
  /**
   * Compiles a single source file into an output file
   * (e.g. object file or bit code file)
   *
   * @param source the content of the source file
   * @param output the compiled output file name (the object file path)
   */
  void compileSource(const std::string& source, const std::string& output,
                     bool posIndepCode) override {
    using namespace CppAD::cg;
    std::vector<std::string> args;
    args.insert(args.end(), this->_compileFlags.begin(),
                this->_compileFlags.end());
    args.push_back("-");
#if !CPPAD_CG_SYSTEM_WIN
    if (posIndepCode) {
      args.push_back("-fPIC");  // position-independent code for dynamic linking
    }
#endif
    args.push_back("-o");
    args.push_back(output);

    std::cout << "\n\nCommand:\n" << this->_path;
    for (const std::string& arg : args) {
      std::cout << " " << arg;
    }
    std::cout << std::endl;

    if (this->_verbose) {
      std::cout << "compiling source '" << output << "'" << std::endl;
      std::cout << "command:  " << this->_path << " ";
      for (const std::string& it : args) {
        std::cout << it << " ";
      }
      std::cout << std::endl;
    }

    system::callExecutable(this->_path, args, nullptr, &source);
  }

  void compileFile(const std::string& path, const std::string& output,
                   bool posIndepCode) override {
    using namespace CppAD::cg;
    std::vector<std::string> args;
    args.insert(args.end(), this->_compileFlags.begin(),
                this->_compileFlags.end());
#if !CPPAD_CG_SYSTEM_WIN
    if (posIndepCode) {
      args.push_back("-fPIC");  // position-independent code for dynamic linking
    }
#endif
    args.push_back(path);
    args.push_back("-o");
    args.push_back(output);

    std::cout << "\n\nCommand:\n" << this->_path;
    for (const std::string& arg : args) {
      std::cout << " " << arg;
    }
    std::cout << std::endl;

    if (this->_verbose) {
      std::cout << "compiling file '" << path << "'" << std::endl;
      std::cout << "command:  " << this->_path << " ";
      for (const std::string& it : args) {
        std::cout << it << " ";
      }
      std::cout << std::endl;
    }

    system::callExecutable(this->_path, args);
  }
};
}  // namespace autogen
