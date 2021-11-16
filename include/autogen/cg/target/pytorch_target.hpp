#pragma once

#include "target.hpp"

namespace autogen {

struct PyTorchTarget : public Target {
  PyTorchTarget() : Target(TargetType::PyTorch) {
    // the generated pytorch code is compiled externally via ninja
    this->code_can_be_compiled_ = false;
  }
};
}  // namespace autogen