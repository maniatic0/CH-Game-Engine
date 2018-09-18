/*
 * Binary File Loading Helper
 *
 */

#ifndef BINARY_LOADER_HEADER_GUARD
#define BINARY_LOADER_HEADER_GUARD

#include <string>
#include <vector>

namespace utils {
std::vector<char> readFile(const std::string &filename);
}  // namespace utils

#endif  // BINARY_LOADER_HEADER_GUARD