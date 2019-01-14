/*
 * File System Path Helper
 *
 */

#include <utils/debug/log.hpp>
#include <utils/files/file_path.hpp>

#include <filesystem>
#include <string>

static std::filesystem::path runtime_path;

namespace utils {

std::string getRuntimeFolder() {
  return runtime_path.parent_path().generic_string();
}

std::string fixRelativePath(const char *path) {
  return getRuntimeFolder() + path;
}

std::string fixRelativePath(std::string path) {
  return getRuntimeFolder() + path;
}

void setRuntimeFolder(const char *path) {
  LOG("Getting Runtime Folder. Using: " << std::string(path));
  runtime_path = std::filesystem::u8path(path);
  LOG("Runtime Folder: " << runtime_path.parent_path().generic_string());
  LOG("Runtime Name: " << runtime_path.filename().generic_string());
}

}  // namespace utils