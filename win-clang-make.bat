@echo off
IF %1.==. GOTO No1
IF %2.==. (
  SET OFFLINE_BUILD=OFF
) ELSE (
  SET OFFLINE_BUILD=%2
)
echo Performing OFFLINE_BUILD=%OFFLINE_BUILD% Note: it should be ON or OFF

cmake -H. -DCMAKE_BUILD_TYPE=%1 -Bbuild/ -T"llvm" -G "Visual Studio 15 2017 Win64"
cmake -DBUILD_OFFLINE=%OFFLINE_BUILD% --build build\ --target install --config %1
GOTO End

:No1
  ECHO Requires Param Release^|^Debug^|RelWithDebInfo^|etc... check CMAKE_CONFIGURATION_TYPES
GOTO End

:End
@echo on