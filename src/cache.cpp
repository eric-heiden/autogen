#include "autogen/utils/cache.h"

#include <fstream>
#include <iostream>
#include <sstream>

#include "autogen/utils/crc.hpp"
#include "autogen/utils/file_utils.hpp"

const char cache_delimiter = ';';
namespace fs = std::filesystem;

std::string get_crc(
    const std::vector<std::pair<std::string, std::string>>& sources) {
  std::uint32_t crc = 0;
  auto table = CRC::CRC_32().MakeTable();
  for (const auto& source : sources) {
    auto& filename = source.first;
    auto& content = source.second;
    crc = CRC::Calculate(content.c_str(), content.size(), table, crc);
  }
  std::stringstream ss;
  ss << std::hex << crc;
  return ss.str();
}

namespace autogen {
std::unique_ptr<Cache> Cache::instance_ = nullptr;

void Cache::clear() {
  if (fs::exists(instance()->cache_folder_)) {
    fs::remove_all(instance()->cache_folder_);
  }
  instance()->data_.clear();
}

bool Cache::exists(
    const std::vector<std::pair<std::string, std::string>>& sources,
    std::string* name) {
  std::string crc = get_crc(sources);
  if (instance()->data_.find(crc) != instance()->data_.end()) {
    if (name) {
      auto entry = instance()->data_[crc];
      *name = entry.name;
    }
    return true;
  }
  return false;
}

std::string Cache::get_project_name(TargetType type, const std::string& name) {
  std::string project_name;
  if (name.empty()) {
    project_name = "gen_" + std::to_string(instance()->data_.size());
  } else {
    project_name = name;
    std::replace(project_name.begin(), project_name.end(), ' ', '_');
    std::replace(project_name.begin(), project_name.end(), '/', '_');
    std::replace(project_name.begin(), project_name.end(), '\\', '_');
    std::replace(project_name.begin(), project_name.end(), '.', '_');
    std::replace(project_name.begin(), project_name.end(), '?', '_');
    std::replace(project_name.begin(), project_name.end(), '*', '_');
    std::replace(project_name.begin(), project_name.end(), ':', '_');
    std::replace(project_name.begin(), project_name.end(), ';', '_');
    std::replace(project_name.begin(), project_name.end(), '\t', '_');
    std::replace(project_name.begin(), project_name.end(), '%', '_');
  }
  project_name += "_" + to_lower(std::to_string(type));
  return project_name;
}

std::string Cache::get_cache_folder() { return instance()->cache_folder_; }

std::string Cache::get_project_folder(TargetType type,
                                      const std::string& name) {
  return fs::path(get_cache_folder())
      .append(get_project_name(type, name))
      .string();
}
std::string Cache::get_source_folder(TargetType type, const std::string& name) {
  return fs::path(get_project_folder(type, name)).append("src").string();
}
std::string Cache::get_temp_folder(TargetType type, const std::string& name) {
  return fs::path(get_project_folder(type, name)).append("tmp").string();
}
std::string Cache::get_cmake_folder(TargetType type, const std::string& name) {
  return fs::path(get_project_folder(type, name)).append("cmake").string();
}
std::string Cache::get_library_file(TargetType type, const std::string& name) {
  return fs::path(get_project_folder(type, name))
      .append(get_project_name(type, name))
      .string();
}

void Cache::save_sources(
    const std::vector<std::pair<std::string, std::string>>& sources, TargetType type,
    const std::string& name) {
  std::string folder = get_source_folder(type, name);
  fs::create_directories(folder);
  std::cout << "Saving source files at " << FileUtils::abs_path(folder) << "\n";
  for (const auto& entry : sources) {
    std::ofstream file(fs::path(folder) / entry.first);
    file << entry.second;
    file.close();
  }
  CacheEntry entry;
  entry.name = name;
  entry.type = type;
  entry.crc = get_crc(sources);
  auto time = std::time(nullptr);
  std::stringstream buffer;
  buffer << std::put_time(std::localtime(&time), "%Y%m%d %H:%M:%S");
  entry.created = buffer.str();
  instance()->data_[entry.crc] = entry;
  instance()->save_cache_();
}

std::ostream& operator<<(std::ostream& os, const Cache::CacheEntry& entry) {
  os << entry.name << cache_delimiter << entry.created << cache_delimiter
     << entry.crc << cache_delimiter << std::to_string(entry.type)
     << cache_delimiter;
  return os;
}
std::istream& operator>>(std::istream& is, Cache::CacheEntry& entry) {
  std::getline(is, entry.name, cache_delimiter);
  std::getline(is, entry.created, cache_delimiter);
  std::getline(is, entry.crc, cache_delimiter);
  std::string type_str;
  std::getline(is, type_str, cache_delimiter);
  entry.type = str2type(type_str);
  return is;
}

void Cache::load_cache_() {
  std::ifstream input(cache_file_);
  std::istringstream ss;
  Cache::CacheEntry entry;
  data_.clear();
  for (std::string line; std::getline(input, line);) {
    ss.str(line);
    ss >> entry;
    data_[entry.crc] = entry;
  }
}

void Cache::save_cache_() {
  std::ofstream output(cache_file_);
  for (const auto& entry : data_) {
    output << entry.second << std::endl;
  }
}

Cache::Cache(const std::string& cache_folder) {
  cache_folder_ = FileUtils::abs_path(cache_folder);
  fs::path cache_path = cache_folder_;
  cache_file_ = (cache_path / "autogen.csv").string();
  if (!fs::exists(cache_path)) {
    fs::create_directories(cache_path);
  } else if (fs::exists(cache_file_)) {
    load_cache_();
  }
  std::cout << "Using autogen cache at \"" << cache_folder_
            << "\" (entries: " << data_.size() << ")." << std::endl;
}

}  // namespace autogen