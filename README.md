# CH-Game-Engine

Game Engine Project based on Game Engine Architecture book by Jason Gregory

## Requires

- CMake
- Clang 7.0.1 (it was made with this compiler in mind and probably only tested on Windows)
- GLFW (automatic install)
- GLM (automatic install)
- STB (automatic install)
- tinyobjloader (automatic install)
- Vulkan and LunarG Vulkan Tools

## Windows Build

Requires <https://marketplace.visualstudio.com/items?itemName=LLVMExtensions.llvm-toolchain> or <https://github.com/plasmacel/llvm-vs2017-integration> to use Clang 7.0.1.

To start CMake:

```Batchfile
cmake -H. -DCMAKE_BUILD_TYPE=%1 -Bbuild/ -T"llvm" -G "Visual Studio 15 2017 Win64"
```

Or

```Batchfile
cmake -H. -DCMAKE_BUILD_TYPE=%1 -Bbuild/ -T"LLVM-vs2017" -G "Visual Studio 15 2017 Win64"
```

To build use on "build" directory:

```Batchfile
cmake --build builds\ --target install --config %1
```

Where %1 is either Debug or Release

## VS Code Help

Use this command to get all the clang/gcc compiler include headers folders

```Batchfile
clang++ -E -x c++ - -v < nul
gcc -v -E -x c++ - -v < nul
clang++ -E -x c++ - -v < /dev/null
g++ -E -x c++ - -v < /dev/null
```

Use this command to get all the clang-cl macro dump. For other clang compiler check <http://nadeausoftware.com/articles/2011/12/c_c_tip_how_list_compiler_predefined_macros>

```Batchfile
clang-cl ... /d1PP /E > file.txt
```

Use this command to get the clang(-cl) call graph. It requires a .dot graph processor

```Batchfile
clang(-cl) ... -Xclang -analyze -Xclang -analyzer-checker=debug.ViewCallGraph
```
