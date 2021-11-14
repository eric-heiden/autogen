#pragma once

#include "autogen/utils/file_utils.hpp"
#include "autogen/utils/system.h"
#include "compact_model.hpp"

#if AUTOGEN_SYSTEM_WIN
#include <windows.h>

namespace {
static std::string GetLastErrorAsString() {
  // https://stackoverflow.com/a/17387176
  // Get the error message ID, if any.
  DWORD errorMessageID = ::GetLastError();
  if (errorMessageID == 0) {
    return std::string();  // No error message has been recorded
  }

  LPSTR messageBuffer = nullptr;

  // Ask Win32 to give us the string version of that message ID.
  // The parameters we pass in, tell Win32 to create the buffer that holds the
  // message for us (because we don't yet know how long the message string will
  // be).
  size_t size = FormatMessageA(
      FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
          FORMAT_MESSAGE_IGNORE_INSERTS,
      NULL, errorMessageID, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
      (LPSTR)&messageBuffer, 0, NULL);

  // Copy the error message into a std::string.
  std::string message(messageBuffer, size);

  // Free the Win32's string's buffer.
  LocalFree(messageBuffer);

  return message;
}
}  // namespace
#endif

namespace autogen {
template <class LibFunction = CompactLibFunction>
class CompactLibrary {
 protected:
  using ModelInfoFunctionPtr = void (*)(char const *const **names, int *count);

  void *lib_handle_{nullptr};

  std::map<std::string, std::unique_ptr<CompactModel<LibFunction>>> models_;

#if !AUTOGEN_SYSTEM_WIN
  int dl_open_mode_ = RTLD_NOW;
#endif

#if AUTOGEN_SYSTEM_WIN
  std::string library_ext_{".dll"};
#else
  std::string library_ext_{".so"};
#endif

 public:
  /**
   * Opens the dynamic library with the given basename (filename without
   * extension), and loads the models.
   */
  virtual void load(const std::string &library_basename,
                    std::string path = "") {
    path += library_basename + library_ext_;
    std::string abs_path;
    bool found = autogen::FileUtils::find_file(path, abs_path);
    assert(found);
#if AUTOGEN_SYSTEM_WIN
    lib_handle_ = LoadLibrary(abs_path.c_str());
    if (lib_handle_ == nullptr) {
      throw std::runtime_error("Failed to dynamically load library '" +
                               library_basename +
                               "': " + GetLastErrorAsString());
    }
#else
    lib_handle_ = dlopen(abs_path.c_str(), dlOpenMode);
    // _dynLibHandle = dlmopen(LM_ID_NEWLM, path.c_str(), RTLD_NOW);
    if (lib_handle_ == nullptr) {
      throw std::runtime_error("Failed to dynamically load library '" +
                               library_basename +
                               "': " + std::string(dlerror()));
    }
#endif
    auto model_info_fun =
        AbstractLibFunction::template load_function<ModelInfoFunctionPtr>(
            "model_info", lib_handle_);
    const char *const *names;
    int count;
    model_info_fun(&names, &count);
    std::cout << "Found " << count << " model";
    if (count != 1) std::cout << "s";
    std::cout << ": ";
    for (int i = 0; i < count; ++i) {
      std::cout << names[i];
      if (i < count - 1) std::cout << ", ";
      models_.emplace(std::make_pair(
          std::string(names[i]),
          std::make_unique<CompactModel<LibFunction>>(names[i], lib_handle_)));
    }
    std::cout << std::endl;
  }

  void clear() {
    // clear models, which will deallocate memory
    // before the library is unloaded
    models_.clear();
#if AUTOGEN_SYSTEM_WIN
    FreeLibrary((HMODULE)lib_handle_);
#else
    dlclose(lib_handle_);
#endif
    lib_handle_ = nullptr;
  }

  virtual ~CompactLibrary() { clear(); }

  CompactModel<LibFunction>* get_model(
      const std::string &model_name) const {
    return models_.at(model_name).get();
  }
  bool has_model(const std::string &model_name) const {
    return models_.find(model_name) != models_.end();
  }

  std::vector<std::string> model_names() const {
    std::vector<std::string> names(models_.size());
    for (const auto &[key, value] : models_) {
      names.push_back(key);
    }
    return names;
  }

#if !AUTOGEN_SYSTEM_WIN
  int dl_open_mode() const { return dl_open_mode_; }
  void set_dl_open_mode(int dl_open_mode) { dl_open_mode_ = dl_open_mode; }
#endif
};
}  // namespace autogen