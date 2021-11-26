#pragma once

#include <ostream>
#include <string>

namespace autogen {
enum TargetType { TARGET_CUDA, TARGET_OPENMP, TARGET_LEGACY_C, TARGET_PYTORCH };

inline TargetType str2type(const std::string &str) {
  std::string s = to_lower(str);
  if (s == "cuda") {
    return TARGET_CUDA;
  } else if (s == "openmp") {
    return TARGET_OPENMP;
  } else if (s == "legacy_c") {
    return TARGET_LEGACY_C;
  } else if (s == "pytorch") {
    return TARGET_PYTORCH;
  } else {
    throw std::runtime_error("Unknown target type: \"" + str + "\"");
  }
}
}  // namespace autogen

namespace std {
inline std::string to_string(autogen::TargetType type) {
  switch (type) {
    case autogen::TARGET_CUDA:
      return "CUDA";
    case autogen::TARGET_OPENMP:
      return "OpenMP";
    case autogen::TARGET_LEGACY_C:
      return "Legacy_C";
    default:
      throw std::runtime_error("Unknown target type");
      return "Unknown";
  }
}
}  // namespace std

inline std::ostream &operator<<(std::ostream &os,
                                const autogen::TargetType &type) {
  return os << std::to_string(type);
}