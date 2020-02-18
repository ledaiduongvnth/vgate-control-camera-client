cmake_minimum_required(VERSION 3.2)

PROJECT (camera_client)

option (USE_ARM64    "Set switch to build at ARM 64 platform mode"   OFF)
set(USE_TENSORRT ON)
add_definitions(-DUSE_TENSORRT)
set(USE_NPP ON)
add_definitions(-DUSE_NPP)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")

if(USE_ARM64)
        add_definitions(-DUSE_ARM64)
        SET(CMAKE_SYSTEM_NAME Linux)
        SET(CMAKE_C_COMPILER "aarch64-linux-gnu-gcc")
        SET(CMAKE_CXX_COMPILER "aarch64-linux-gnu-g++")
endif()

if(USE_ARM64)
    add_definitions (-std=c++11 -O2 -fomit-frame-pointer -g -Wall)
    MESSAGE (STATUS "Build Option: -std=c++11 -O2 -fomit-frame-pointer -g -Wall")
else()
    add_definitions (-std=c++11 -O2 -fomit-frame-pointer -g -Wall)
    MESSAGE (STATUS "Build Option: -std=c++11 -O2 -fomit-frame-pointer -g -Wall")
endif()

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};
        -O3 -gencode arch=compute_50,code=sm_50;
        -gencode arch=compute_52,code=sm_52;
        -gencode arch=compute_53,code=sm_53;
        -gencode arch=compute_60,code=sm_60;
        -gencode arch=compute_61,code=sm_61;
        -gencode arch=compute_62,code=sm_62;
        -gencode arch=compute_70,code=sm_70;
        )

find_package(Protobuf REQUIRED)
find_package(gRPC CONFIG REQUIRED)
set(GRPC_CPP_LIB gRPC::grpc++_unsecure)
set(GRPC_PLUGIN_EXEC $<TARGET_FILE:gRPC::grpc_cpp_plugin>)

include_directories (
        "facedetect"
        "facedetect/tensorrt"
        "/usr/local/TensorRT/include"
        "/usr/local/cuda/include"
        proto
        utils
        tracking
)

AUX_SOURCE_DIRECTORY(facedetect DIR_SRCS)
AUX_SOURCE_DIRECTORY(facedetect/tensorrt DIR_SRCS_CUDA)
AUX_SOURCE_DIRECTORY(./proto PROTO)
AUX_SOURCE_DIRECTORY(./utils UTILS)
AUX_SOURCE_DIRECTORY(./tracking TRACKING)

file( GLOB  core_cuda_files  "facedetect/*.cu")
cuda_add_executable(
        camera_client
        main.cpp
        ${DIR_SRCS}
        ${DIR_SRCS_CUDA}
        ${core_cuda_files}
        ${PROTO}
        ${UTILS}
        ${TRACKING}
)

target_link_libraries(
        camera_client
        -L/usr/local/cuda/lib64
        -L/usr/local/TensorRT/lib
        -lnvinfer -lnvcaffe_parser -ljsoncpp -lX11
        -lcuda -lcudart -lcublas -lcudnn -lcurand
        -lboost_thread -lboost_system -lnppig
        -lnppicc -lnppc -lnppidei -lnppist
        ${GRPC_CPP_LIB}
        ${Protobuf_LIBRARIES}
        ${OpenCV_LIBS}
)

