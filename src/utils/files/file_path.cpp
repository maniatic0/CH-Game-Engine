/*
 * File System Path Helper
 *
 */

#include <utils/debug/log.hpp>
#include <utils/files/file_path.hpp>
#include <utils/files/files_config.h>


#include <filesystem>
#include <string>

static std::filesystem::path runtime_path;
static std::filesystem::path assets_path;

namespace utils {

std::string getRuntimeFolder() {
  return runtime_path.parent_path().generic_string();
}

std::filesystem::path getRuntimeFolderPath() {
  return runtime_path.parent_path();
}

std::string getAssetsFolder() { return assets_path.generic_string(); }

std::filesystem::path getAssetsFolderPath() { return assets_path; }

std::string fixRelativePath(const char *path) {
  return (getRuntimeFolderPath() / path).generic_string();
}

std::string fixRelativePath(const std::string &path) {
  return (getRuntimeFolderPath() / path).generic_string();
}

std::string fixRelativeAssetPath(const std::string &path) {
  return (getAssetsFolderPath() / path).generic_string();
}

std::string fixRelativeAssetPath(const char *path) {
  return (getAssetsFolderPath() / path).generic_string();
}

void setRuntimeFolder(const char *path) {
  // See this for upath deprectation
  // https://stackoverflow.com/questions/54004000/why-is-stdfilesystemu8path-deprecated-in-c20
  LOG("Getting Runtime Folder. Using: " << std::string(path));
  runtime_path = std::filesystem::u8path(path);
  assets_path = runtime_path.parent_path() / ASSETS_FOLDER_NAME;
  LOG("Runtime Folder: " << runtime_path.parent_path().generic_string());
  LOG("Runtime Name: " << runtime_path.filename().generic_string());
  LOG("Assets Folder: " << assets_path.generic_string());
}

} // namespace utils