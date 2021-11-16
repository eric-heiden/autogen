#include "autogen/utils/system.h"

#include <array>
#include <fstream>

#include "autogen/utils/filesystem.hpp"

namespace autogen {
std::string exec(const std::string &cmd, const std::vector<std::string> &args,
                 bool throw_exception_on_error, int *return_code) {
  std::string msg;
  int code = CppAD::cg::system::callExecutable(cmd, args, &msg, nullptr);
  if (return_code != nullptr) {
    *return_code = code;
  }
  msg.erase(std::remove(msg.begin(), msg.end(), '\n'), msg.end());
  msg.erase(std::remove(msg.begin(), msg.end(), '\r'), msg.end());
  if (code != 0 && throw_exception_on_error) {
    std::stringstream ss;
    ss << cmd;
    for (const auto &arg : args) {
      ss << " " << arg;
    }
    throw std::runtime_error("Error: command \"" + ss.str() +
                             "\" returned exit code " + std::to_string(code) +
                             ".\n" + msg);
  }
  return msg;
}

std::string find_exe(const std::string &name, bool throw_exception_on_error) {
  try {
#if AUTOGEN_SYSTEM_WIN
    std::string path =
        autogen::exec("powershell",
                      std::vector<std::string>{
                          "-command", "\"(get-command " + name + ").Path\""},
                      throw_exception_on_error);
#else
    std::string path =
        autogen::exec("/usr/bin/which", std::vector<std::string>{name},
                      throw_exception_on_error);
#endif
    if (!file_exists(path)) {
      if (throw_exception_on_error) {
        throw std::runtime_error("Error: could not find executable \"" + name +
                                 "\"");
      }
      return "";
    }
    return path;
  } catch (CppAD::cg::CGException &e) {
    throw std::runtime_error("Error: could not find executable \"" + name +
                             "\"");
  }
}
}  // namespace autogen