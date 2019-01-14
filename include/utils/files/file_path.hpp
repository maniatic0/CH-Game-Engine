/*
 * File System Path Helper
 *
 */

#ifndef FILE_PATH_HEADER_GUARD
#define FILE_PATH_HEADER_GUARD

#include <filesystem>
#include <string>

namespace utils {

std::string getRuntimeFolder();

std::filesystem::path getRuntimeFolderPath();

std::string getAssetsFolder();

std::filesystem::path getAssetsFolderPath();

std::string fixRelativePath(const char *path);

std::string fixRelativePath(const std::string &path);

std::string fixRelativeAssetPath(const std::string &path);

std::string fixRelativeAssetPath(const char *path);

void setRuntimeFolder(const char *path);
} // namespace utils

#endif // FILE_PATH_HEADER_GUARD