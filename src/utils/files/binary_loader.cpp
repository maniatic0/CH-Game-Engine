/*
 * Binary File Loading Helper
 *
 */

#include <utils/files/binary_loader.hpp>

#include <fstream>

namespace utils {
std::vector<char> readFile(const std::string &filename) {
  // std::ios::ate makes us read from end to start of file. It gives us the size
  // at the start
  std::ifstream file(filename, std::ios::ate | std::ios::binary);

  if (!file.is_open()) {
    throw std::runtime_error("failed to open file!");
  }

  size_t fileSize = (size_t)file.tellg();
  std::vector<char> buffer(fileSize);

  file.seekg(0);
  file.read(buffer.data(), fileSize);

  file.close();

  return buffer;
}
} // namespace utils