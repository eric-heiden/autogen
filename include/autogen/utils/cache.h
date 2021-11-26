#pragma once

#include <map>

#include "autogen/core/base.hpp"
#include "autogen/core/target_type.hpp"

namespace autogen {
struct Cache {
  static void clear();

  static bool exists(
      const std::vector<std::pair<std::string, std::string>>& sources,
      std::string* name = nullptr);

  static std::string get_cache_folder();

  static std::string get_project_name(TargetType type,
                                      const std::string& name = "");

  static std::string get_project_folder(TargetType type,
                                        const std::string& name = "");
  static std::string get_source_folder(TargetType type,
                                       const std::string& name = "");
  static std::string get_temp_folder(TargetType type,
                                     const std::string& name = "");
  static std::string get_cmake_folder(TargetType type,
                                     const std::string& name = "");
  static std::string get_library_file(TargetType type,
                                      const std::string& name = "");

  static void save_sources(
      const std::vector<std::pair<std::string, std::string>>& sources,
      TargetType type, const std::string& name = "");

  struct CacheEntry {
    std::string name;
    std::string created;
    std::string crc;
    TargetType type;

    friend std::ostream& operator<<(std::ostream& os, const CacheEntry& entry);
    friend std::istream& operator>>(std::istream& is, CacheEntry& entry);
  };

 private:
  static std::unique_ptr<Cache> instance_;
  static inline std::unique_ptr<Cache>& instance() {
    if (!instance_) {
      instance_ = std::unique_ptr<Cache>(new Cache);
    }
    return instance_;
  }

  std::string cache_folder_;
  std::string cache_file_;
  std::map<std::string, CacheEntry> data_;

  Cache(const std::string& cache_folder = ".autogen/");

  void load_cache_();
  void save_cache_();
};
}  // namespace autogen