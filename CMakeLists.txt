cmake_minimum_required(VERSION 3.2)

PROJECT (camera_client)

option (USE_ARM64    "Set switch to build at ARM 64 platform mode"   ON)
add_definitions(-DUSE_TENSORRT)
add_definitions(-DUSE_NPP)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")

if(USE_ARM64)
        add_definitions(-DUSE_ARM64)
        SET(CMAKE_SYSTEM_NAME Linux)
        SET(CMAKE_C_COMPILER "aarch64-linux-gnu-gcc")
        SET(CMAKE_CXX_COMPILER "aarch64-linux-gnu-g++")
        set(CMAKE_CXX_FLAGS  "-lpthread")
        set(CMAKE_CXX_FLAGS  "-O3")
endif()

execute_process ( COMMAND sh ${CMAKE_CURRENT_SOURCE_DIR}/proto/gen.sh
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/proto )

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

find_package(Freetype REQUIRED)
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
        ${FREETYPE_INCLUDE_DIRS}
        /usr/local/include/jetson-utils
        /usr/include/gstreamer-1.0
        /usr/include/glib-2.0
        /usr/include/libxml2
        /usr/lib/aarch64-linux-gnu/gstreamer-1.0/include
        /usr/lib/x86_64-linux-gnu/gstreamer-1.0/include
        /usr/lib/aarch64-linux-gnu/glib-2.0/include
        /usr/lib/x86_64-linux-gnu/glib-2.0/include
        /usr/include/gstreamer-1.0
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
        -lnvinfer -lnvonnxparser -ljsoncpp -lX11
        -lcuda -lcudart -lcublas -lcudnn -lcurand
        -lboost_thread -lboost_system -lnppig
        -lnppicc -lnppc -lnppidei -lnppist -ljetson-utils
        ${GRPC_CPP_LIB}
        ${Protobuf_LIBRARIES}
        ${OpenCV_LIBS}
        ${FREETYPE_LIBRARIES}
)

