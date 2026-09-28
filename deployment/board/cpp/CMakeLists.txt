cmake_minimum_required(VERSION 3.16)
project(radar_pipeline CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ONNX Runtime C++ 预编译库(PC 构建; BOARD 构建关 ORT)
set(ORT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/onnxruntime-linux-x64-1.30.0" CACHE PATH "onnxruntime root")
set(WEIGHTS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../weights" CACHE PATH "deploy weights dir")
set(VARTX86_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/vart-x86" CACHE PATH "vart x86 libs root")
option(USE_VARTCPU "接入 VART cpu-runner 后端(PC 跑量化 xmodel)" ON)
option(BOARD "板端交叉编译(USE_VART 真 runner, 关 ORT, 链 sysroot VART)" OFF)

add_library(radar_core STATIC
    src/point_io.cpp
    src/pillarize.cpp
    src/pillar_vfe.cpp
    src/pillar_attention.cpp
    src/scatter.cpp
    src/weights_loader.cpp
    src/dpu_backend.cpp
    src/decode_nms.cpp
)
target_compile_options(radar_core PRIVATE -O3 -Wall)

if(BOARD)
    # 板端: compiled xmodel + VART GraphRunner; VART 头/库在 SDK sysroot(CMAKE_FIND_ROOT_PATH 内)
    set(USE_VARTCPU OFF)
    target_compile_definitions(radar_core PUBLIC USE_VART)
    target_include_directories(radar_core PUBLIC src)
    # A53 定向优化(-O3 已有); FASTMATH 放宽求和序换 SIMD 向量化; FRONT_THREADS 双线程分段
    # BOARD 模式默认全开(=板上 67.8ms 达标配置); 需对比实验时 -DFASTMATH=OFF / -DFRONT_THREADS=OFF 显式关
    option(FASTMATH "板端 -ffast-math 向量化放宽" ON)
    option(FRONT_THREADS "front 双线程(std::thread 分段, 板上无 libgomp 故不用 OpenMP)" ON)
    target_compile_options(radar_core PRIVATE -mcpu=cortex-a53)
    if(FASTMATH)
        target_compile_options(radar_core PRIVATE -ffast-math)
    endif()
    if(FRONT_THREADS)
        target_compile_definitions(radar_core PUBLIC FRONT_THREADS)
        find_package(Threads REQUIRED)
        target_link_libraries(radar_core PUBLIC Threads::Threads)
    endif()
    target_link_libraries(radar_core PUBLIC vart-runner vart-mem-manager xir unilog target-factory glog)
    add_executable(radar_pipeline_board src/main.cpp)
    target_link_libraries(radar_pipeline_board PRIVATE radar_core)
    set_target_properties(radar_pipeline_board PROPERTIES OUTPUT_NAME radar_pipeline_board)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_decode.cpp")
        add_executable(test_decode tests/test_decode.cpp)
        target_link_libraries(test_decode PRIVATE radar_core)
    endif()
    return()
endif()

# ---- PC 构建(现状不变): ORT 常驻 + 可选 VARTCPU ----
target_include_directories(radar_core PUBLIC src ${ORT_ROOT}/include)

if(USE_VARTCPU)
    target_compile_definitions(radar_core PUBLIC USE_VARTCPU)
    target_include_directories(radar_core PUBLIC
        ${VARTX86_ROOT}/include/vart_cpu ${VARTX86_ROOT}/include)
    target_link_directories(radar_core PUBLIC ${VARTX86_ROOT}/lib)
    target_link_libraries(radar_core PUBLIC
        vart-cpu-runner vart-runner vart-util vart-mem-manager
        xir unilog target-factory glog boost_filesystem boost_system protobuf)
    set(CMAKE_BUILD_RPATH "${VARTX86_ROOT}/lib")
    set(CMAKE_INSTALL_RPATH "${VARTX86_ROOT}/lib")
endif()

add_executable(radar_pipeline src/main.cpp)
target_link_options(radar_pipeline PRIVATE -Wl,--disable-new-dtags)  # RPATH 对传递依赖生效
target_link_libraries(radar_pipeline PRIVATE radar_core)
target_link_directories(radar_pipeline PRIVATE ${ORT_ROOT}/lib)
target_link_libraries(radar_pipeline PRIVATE onnxruntime)
set_target_properties(radar_pipeline PROPERTIES
    BUILD_RPATH "${ORT_ROOT}/lib"
    INSTALL_RPATH "${ORT_ROOT}/lib")

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_decode.cpp")
    add_executable(test_decode tests/test_decode.cpp)
    target_link_libraries(test_decode PRIVATE radar_core)
endif()
