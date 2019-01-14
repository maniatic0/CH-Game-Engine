/*
 * Binary File Loading Helper
 *
 */

#include <utils/files/files_config.h>
#include <utils/debug/log.hpp>
#include <utils/files/file_path.hpp>

#include <fstream>

#include <utils/files/binary_loader.hpp>

namespace utils {

std::vector<char> loadFile(std::ifstream &file) {
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

std::vector<char> readFile(const std::string &filename,
                           bool use_absolute_path /* = false */) {
  // std::ios::ate makes us read from end to start of file. It gives us the size
  // at the start
  std::ifstream file(
      !use_absolute_path ? utils::fixRelativePath(filename) : filename,
      std::ios::ate | std::ios::binary);

  return utils::loadFile(file);
}

std::vector<char> readAssetFile(const std::string &filename,
                                bool is_shader /* = false */) {
  // std::ios::ate makes us read from end to start of file. It gives us the size
  // at the start
  std::ifstream file(!is_shader ? utils::fixRelativeAssetPath(filename)
                                : utils::fixRelativeAssetPath(
                                      SHADERS_FOLDER_NAME"/" + filename),
                     std::ios::ate | std::ios::binary);

  return utils::loadFile(file);
}

}  // namespace utils