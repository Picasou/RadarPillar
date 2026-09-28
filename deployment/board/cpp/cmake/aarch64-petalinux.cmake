# ⑤ 交叉编译工具链(PetaLinux SDK): 先 source <sdk>/environment-setup-aarch64-xilinx-linux 再 cmake
# SDK env 导出 CC/CXX(含 flags) 与 SDKTARGETSYSROOT; 此处取编译器路径并剥 flags, sysroot 走 CMAKE_FIND_ROOT_PATH
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

if(NOT DEFINED ENV{SDKTARGETSYSROOT})
  message(FATAL_ERROR "先 source SDK: environment-setup-aarch64-xilinx-linux (缺 SDKTARGETSYSROOT)")
endif()
set(CMAKE_SYSROOT $ENV{SDKTARGETSYSROOT})

# CC 形如 "aarch64-xilinx-linux-gcc -mbranch-protection=standard ..." → 取首 token
separate_arguments(_CC_ARGS NATIVE_COMMAND "$ENV{CC}")
separate_arguments(_CXX_ARGS NATIVE_COMMAND "$ENV{CXX}")
list(GET _CC_ARGS 0 CMAKE_C_COMPILER)
list(GET _CXX_ARGS 0 CMAKE_CXX_COMPILER)

set(CMAKE_FIND_ROOT_PATH ${CMAKE_SYSROOT})
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
