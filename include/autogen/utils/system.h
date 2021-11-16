#pragma once

#include "autogen/core/base.hpp"
#include "filesystem.hpp"

namespace autogen {
inline std::string exec(const std::string &cmd,
                        const std::vector<std::string> &args,
                        bool throw_exception_on_error = true,
                        int *return_code = nullptr);

inline bool file_exists(const std::string &filename) {
  namespace fs = std::filesystem;
  return fs::exists(filename);
}

inline bool directory_exists(const std::string &dirname) {
  namespace fs = std::filesystem;
  return fs::is_directory(dirname);
}

// returns the absolute path of the executable
inline std::string find_exe(const std::string &name,
                            bool throw_exception_on_error = true);
}  // namespace autogen