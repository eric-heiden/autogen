#pragma once

#include <ostream>
#include <string>

namespace autogen {
enum TargetType { TARGET_CUDA, TARGET_OPENMP, TARGET_LEGACY_C, TARGET_PYTORCH };
}

namespace std {
inline std::string to_string(autogen::TargetType type) {
  switch (type) {
    case autogen::TARGET_CUDA:
      return "CUDA";
    case autogen::TARGET_OPENMP:
      return "OpenMP";
    case autogen::TARGET_LEGACY_C:
      return "LegacyC";
    default:
      throw std::runtime_error("Unknown target type");
      return "Unknown";
  }
}
}  // namespace std

inline std::ostream &operator<<(std::ostream &os, const autogen::TargetType &type) {
  return os << std::to_string(type);
}