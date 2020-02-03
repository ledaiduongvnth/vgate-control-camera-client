cmake_minimum_required(VERSION 3.2)

option (USE_ARM64           "Set switch to build at ARM 64 platform mode"   OFF)
option (USE_DEBUG           "Set switch to build at Debug mode"             OFF)
option (USE_CAFFE           "Set switch to build at USE_CAFFE mode"         OFF)
option (USE_TENSORRT        "Set switch to build at USE_TENSORRT mode"      ON)
option (USE_NPP             "Set switch to build at USE_NPP mode"           ON)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")

set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")


if(USE_ARM64)
    SET(CMAKE_SYSTEM_NAME Linux)
    SET(CMAKE_C_COMPILER "aarch64-linux-gnu-gcc")
    SET(CMAKE_CXX_COMPILER "aarch64-linux-gnu-g++")
endif()

PROJECT (retinaFaceReImp)

if(USE_ARM64)
    add_definitions (-std=c++11 -O2 -fomit-frame-pointer -g -Wall)
    MESSAGE (STATUS "Build Option: -std=c++11 -O2 -fomit-frame-pointer -g -Wall")
else()
    add_definitions (-std=c++11 -O2 -fomit-frame-pointer -g -Wall)
    MESSAGE (STATUS "Build Option: -std=c++11 -O2 -fomit-frame-pointer -g -Wall")
endif()

find_package(OpenCV REQUIRED)

if(USE_TENSORRT)
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
endif()

if(USE_ARM64)
    add_definitions(-DUSE_ARM64)
    MESSAGE (STATUS "Build Option: -DUSE_ARM64")
endif()

if(USE_DEBUG)
    add_definitions(-D_DEBUG)
    MESSAGE (STATUS "Build Option: -D_DEBUG")
endif()

if(USE_TENSORRT)
    add_definitions(-DUSE_TENSORRT)
    MESSAGE (STATUS "Build Option: -DUSE_TENSORRT")
elseif(USE_CAFFE)
    add_definitions(-DUSE_CAFFE)
    MESSAGE (STATUS "Build Option: -DUSE_CAFFE")
endif()

if(USE_NPP)
    add_definitions(-DUSE_NPP)
    MESSAGE(STATUS "Build Option: -DUSE_NPP")
endif()

find_package(Protobuf REQUIRED)
find_package(gRPC CONFIG REQUIRED)
message(STATUS "Using gRPC ${gRPC_VERSION}")
set(GRPC_CPP_LIB gRPC::grpc++_unsecure)
set(GRPC_PLUGIN_EXEC $<TARGET_FILE:gRPC::grpc_cpp_plugin>)

include_directories (
    "./retinaface"
    "./retinaface/tensorrt"
    "/usr/local/TensorRT/include"
    "/usr/local/cuda/include"
        proto
        utils
)

AUX_SOURCE_DIRECTORY(./retinaface DIR_SRCS)

if(USE_TENSORRT)
    if(USE_NPP)
        file( GLOB  core_cuda_files  "./retinaface/*.cu")
    endif()
    AUX_SOURCE_DIRECTORY(./retinaface/tensorrt DIR_SRCS_CUDA)
    cuda_add_executable(
            retinaface
            ${DIR_SRCS}
            ${DIR_SRCS_CUDA}
            ${core_cuda_files}
            proto/multiple_camera_server.grpc.pb.cc
            proto/multiple_camera_server.pb.cc
            utils/base64.cpp
    )
else()
    add_executable(retinaface ${DIR_SRCS})
endif()

target_link_libraries(retinaface -lboost_system -lglog -lopencv_core  -lopencv_videoio -lopencv_imgproc -lopencv_highgui -lopencv_video -lopencv_imgcodecs)

if(USE_TENSORRT)
    target_link_libraries(retinaface -L/usr/local/cuda/lib64 -L/usr/local/TensorRT/lib
        -lnvinfer -lnvcaffe_parser -lcuda -lcudart -lcublas -lcudnn -lcurand  -lboost_thread -lnppig
        -lnppicc -lnppc -lnppidei -lnppist
            ${GRPC_CPP_LIB}
            ${Protobuf_LIBRARIES}
            ${OpenCV_LIBS})
elseif(USE_CAFFE)
    target_link_libraries(retinaface -lcaffe -lboost_system -lboost_thread)
endif()

