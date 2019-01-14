/*
 * File System Path Helper
 *
 */

#ifndef FILE_PATH_HEADER_GUARD
#define FILE_PATH_HEADER_GUARD

#include <string>

namespace utils {

std::string getRuntimeFolder();

std::string fixRelativePath(const char *path);

std::string fixRelativePath(std::string path);

void setRuntimeFolder(const char *path);
} // namespace utils

#endif // FILE_PATH_HEADER_GUARD