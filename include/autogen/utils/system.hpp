#pragma once

// operating system detection
#ifndef AUTOGEN_SYSTEM_LINUX
#if defined(__linux__) || defined(__linux) || defined(linux)
#define AUTOGEN_SYSTEM_LINUX 1
#endif
#endif
#ifndef AUTOGEN_SYSTEM_APPLE
#if defined(__APPLE__)
#define AUTOGEN_SYSTEM_APPLE 1
#define AUTOGEN_SYSTEM_LINUX 1
#endif
#endif
#ifndef AUTOGEN_SYSTEM_WIN
#if defined(_WIN32) || defined(_WIN64) || defined(__WIN32__) || \
    defined(__TOS_WIN__) || defined(__WINDOWS__)
#define AUTOGEN_SYSTEM_WIN 1
#endif
#endif

#include <array>
#include <cppad/cg.hpp>
#include <fstream>

namespace autogen {
static std::string exec(const std::string &cmd,
                        const std::vector<std::string> &args,
                        bool throw_exception_on_error = true,
                        int *return_code = nullptr) {
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

static bool file_exists(const std::string &filename) {
  std::ifstream file(filename);
  return file.good();
}

// returns the absolute path of the executable
static std::string find_exe(const std::string &name,
                            bool throw_exception_on_error = true) {
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