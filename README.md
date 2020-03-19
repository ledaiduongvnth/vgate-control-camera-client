### install grpc c++

### on Jetson nano

# currently we use the 1.23.0 version of grpc lib in c++ as following:
https://github.com/grpc/grpc/commit/7c0764918b9f33cab507ff483b4be849b0203ec4

# follow the instruction to build grpc from source
https://github.com/grpc/grpc/blob/master/BUILDING.md

# install protobuf from source:
grpc/third_party/protobuf

# install borring_ssl:


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

### on Jetson TX2
the same instruction with jetson nano. In additional:

# install protobuf from source :
https://stackoverflow.com/questions/43937682/upgrading-protobuf-tensorflow-on-jetson-tx2

# error
c-aresConfig.cmake
c-ares-config.cmake

cmake -DgRPC_INSTALL=ON -DgRPC_BUILD_TESTS=OFF -DgRPC_PROTOBUF_PROVIDER=package -DgRPC_ZLIB_PROVIDER=package -DgRPC_CARES_PROVIDER=package -DgRPC_SSL_PROVIDER=package -D_gRPC_CARES_LIBRARIES=cares -DgRPC_CARES_PROVIDER=kludge -DCMAKE_BUILD_TYPE=Release ../..
