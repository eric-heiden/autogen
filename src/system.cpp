#include "autogen/utils/system.h"

#include <array>
#include <fstream>

#ifdef AUTOGEN_SYSTEM_WIN
#undef UNICODE
#include <stdio.h>
#include <stdlib.h>
#include <tchar.h>
#include <windows.h>
#endif

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

#ifdef AUTOGEN_SYSTEM_WIN
void printError(TCHAR *msg) {
  DWORD eNum;
  TCHAR sysMsg[256];
  TCHAR *p;

  eNum = GetLastError();
  FormatMessage(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                NULL, eNum, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), sysMsg,
                256, NULL);

  // Trim the end of the line and terminate it with a null
  p = sysMsg;
  while ((*p > 31) || (*p == 9)) ++p;
  do {
    *p-- = 0;
  } while ((p >= sysMsg) && ((*p == '.') || (*p < 33)));

  // Display the message
  _tprintf(TEXT("\n\t%s failed with error %d (%s)"), msg, eNum, sysMsg);
}
#endif

void load_windows_build_variables() {
#ifdef AUTOGEN_SYSTEM_WIN
#define INFO_BUFFER_SIZE 32767
  TCHAR infoBuf[INFO_BUFFER_SIZE];
  DWORD bufCharCount;
  bufCharCount =
      ExpandEnvironmentStrings(TEXT("%ProgramFiles(x86)%\\Microsoft Visual "
                                    "Studio\\Installer\\vswhere.exe"),
                               infoBuf, INFO_BUFFER_SIZE);
  if (!bufCharCount) {
    printError(TEXT("ExpandEnvironmentStrings"));
    return;
  }
  std::string vs_path = autogen::exec(
      std::string(infoBuf),
      std::vector<std::string>{"-latest", "-property", "installationPath"});
  // std::cout << "vs_path: \"" << vs_path << "\"" << std::endl;

  std::string vsvars_path = vs_path + "\\VC\\Auxiliary\\Build\\vcvars64.bat";
  // std::cout << "vsvars_path: \"" << vsvars_path << "\"" << std::endl;
  // read lines from file vsvars_path and set environment variables
  std::vector<std::string> args{"/C \"" + vsvars_path + "\"", " && set"};
  std::string output;
  int code =
      CppAD::cg::system::callExecutable("cmd.exe", args, &output, nullptr);
  if (code != 0) {
    throw std::runtime_error("Failed to load build variables from \"" +
                             vsvars_path + "\"");
  }
  std::istringstream ss(output);
  int counter = 0;
  for (std::string line; std::getline(ss, line);) {
    if (line.find("=") != std::string::npos) {
      std::string var_name = line.substr(0, line.find("="));
      // only load these specific build variables that are necessary for MSVC,
      // otherwise some PowerShell errors appear
      if (to_lower(var_name) == "include" || to_lower(var_name) == "lib" ||
          to_lower(var_name) == "libpath") {
        std::string var_value = line.substr(line.find("=") + 1);
        // std::cout << "setting " << var_name << " = " << var_value <<
        // std::endl;
        SetEnvironmentVariable(var_name.c_str(), var_value.c_str());
        ++counter;
      }
    }
  }
  std::cout << "Loaded " << counter
            << " build variables from \"" + vsvars_path + "\"." << std::endl;
#endif
}
}  // namespace autogen