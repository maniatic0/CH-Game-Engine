@echo off
IF %1.==. GOTO No1

cmake -H. -DCMAKE_BUILD_TYPE=%1 -Bbuilds/ -T"LLVM-vs2017" -G "Visual Studio 15 2017 Win64"
cmake --build builds\ --target install --config %1
GOTO End

:No1
  ECHO Requires Param Release^|^Debug^|RelWithDebInfo^|etc... check CMAKE_CONFIGURATION_TYPES
GOTO End

:End
@echo on