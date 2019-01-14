/*
 * Binary File Loading Helper
 *
 */

#ifndef BINARY_LOADER_HEADER_GUARD
#define BINARY_LOADER_HEADER_GUARD

#include <string>
#include <vector>

namespace utils {

std::vector<char> readFile(const std::string &filename,
                           bool use_absolute_path = false);

std::vector<char> readAssetFile(const std::string &filename,
                                bool is_shader = false);
}  // namespace utils

#endif  // BINARY_LOADER_HEADER_GUARD