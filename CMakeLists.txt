cmake_minimum_required(VERSION 3.2)

#set(CMAKE_CXX_STANDARD 14)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CXX_EXTENSIONS OFF)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")


PROJECT(camera_client CXX)

find_package( OpenCV REQUIRED )

find_package(Protobuf REQUIRED)
find_package(gRPC CONFIG REQUIRED)
message(STATUS "Using gRPC ${gRPC_VERSION}")
set(GRPC_CPP_LIB gRPC::grpc++_unsecure)
set(GRPC_PLUGIN_EXEC $<TARGET_FILE:gRPC::grpc_cpp_plugin>)

include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_executable(
        camera_client
        main.cpp
        proto/multiple_camera_server.grpc.pb.cc
        proto/multiple_camera_server.pb.cc
        utils/base64.cpp
)

target_include_directories(
        camera_client
        PRIVATE
        proto
        utils
)

target_link_libraries(
        camera_client
        ${GRPC_CPP_LIB}
        ${Protobuf_LIBRARIES}
        ${OpenCV_LIBS}
)