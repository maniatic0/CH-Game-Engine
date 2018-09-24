/*
 * File System Path Helper
 *
 */

#ifndef FILE_PATH_HEADER_GUARD
#define FILE_PATH_HEADER_GUARD

#include <string>

namespace utils {

std::string getRuntimeFolder();

void setRuntimeFolder(const char *path);
}  // namespace utils

#endif  // FILE_PATH_HEADER_GUARD