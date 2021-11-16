#pragma once

#include "compact_lib_function.hpp"

namespace autogen {
/**
 * Defines a model that is loaded from a dynamic library.
 */
template <class LibFunction>
struct CompactModel {
  template <class LibFunction2>
  friend class CompactLibrary;

 protected:
  const std::string model_name_;

 public:
  LibFunction forward_zero;
  LibFunction jacobian;
  // LibFunction sparse_jacobian;

  const std::string &model_name() { return model_name_; }

  CompactModel(const std::string &model_name, void *lib_handle)
      : model_name_(model_name) {
    forward_zero.load(model_name + "_forward_zero", lib_handle);
    jacobian.load(model_name + "_jacobian", lib_handle);
    // sparse_jacobian.load(model_name + "_sparse_jacobian", lib_handle);
  }

  virtual ~CompactModel() = default;
};
}  // namespace autogen