### install grpc c++ 

# currently we use the 1.23.0 version of grpc lib in c++ as following:
https://github.com/grpc/grpc/commit/7c0764918b9f33cab507ff483b4be849b0203ec4

# follow the instruction to build grpc from source
https://github.com/grpc/grpc/blob/master/BUILDING.md

# if we get the error "all warnings being treated as errors" while compiling boringssl:
remove flag: "-Werror" from boringssl"s CMakeLists.txt 

# if we get the error :
CMake Error at /usr/local/lib/cmake/grpc/gRPCConfig.cmake:8 (include):
  include could not find load file:

    /usr/local/lib/cmake/grpc/gRPCTargets.cmake
# use the cmake command
cmake -DgRPC_INSTALL=ON -DgRPC_BUILD_TESTS=OFF -DgRPC_PROTOBUF_PROVIDER=package -DgRPC_ZLIB_PROVIDER=package -DgRPC_CARES_PROVIDER=package -DgRPC_SSL_PROVIDER=package -DCMAKE_BUILD_TYPE=Release ../.. 


# install libjsoncpp:
sudo apt-get install libjsoncpp-dev 

# install libboots
sudo apt-get install libboost-all-dev
