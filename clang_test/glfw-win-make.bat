
@echo ###############################
@echo GENERATING BUILD SYSTEM

cmake -H. -Bbuilds/ -T"LLVM-vs2017" -G "Visual Studio 15 2017 Win64" -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/install 

@echo ###############################
@echo BUILDING LIBRARIES
cmake -DBUILD_SHARED_LIBS=ON -DLIB_SUFFIX=C:\Users\Christian\Desktop\CH_Engine\CH-Game-Engine\libraries --build builds\

@echo ###############################
@echo BUILDING EXAMPLES
cmake --build builds\
