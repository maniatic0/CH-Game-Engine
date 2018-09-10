# CH-Game-Engine
Game Engine Project based on Game Engine Architecture book by Jason Gregory

# Requires
* CMake
* Clang 6.0.0 (it was made with this compiler in mind and probably only tested on Windows)
* GLFW (automatic install)
* GLM (automatic install)
* Vulkan

# Windows Build
Requires https://github.com/plasmacel/llvm-vs2017-integration to use Clang 6.0.0

To start CMake:
mkdir build\ && cd build\ && cmake ..\ -G "Visual Studio 15 2017 Win64" -T LLVM-vs2017
or:
cmake -H. -Bbuilds/ -T"LLVM-vs2017" -G"Visual Studio 15 2017 Win64"

To build use on "build" directory:
cmake --build builds\