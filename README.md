# CH-Game-Engine
Game Engine Project based on Game Engine Architecture book by Jason Gregory

# Requires
* CMake
* Clang 6.0.0 (it was made with this compiler in mind and probably only tested on Windows)
* GLFW (automatic install)
* GLM (automatic install)
* Vulkan

# Windows Build
Requires https://github.com/plasmacel/llvm-vs2017-integration to use Clang 6.0.0.


To start CMake:
```Batchfile
cmake -H. -DCMAKE_BUILD_TYPE=%1 -Bbuild/ -T"LLVM-vs2017" -G "Visual Studio 15 2017 Win64"
```

To build use on "build" directory:
```Batchfile
cmake --build builds\ --target install --config %1
```

Where %1 is either Debug or Release

# VS Code Help
Use this command to get all the clang/gcc compiler include headers folders
```Batchfile
clang -v -E -x c++ -
gcc -v -E -x c++ -
```