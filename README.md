#BUILD INSTRUCTION

## Install default OS for the device
1, For Jetson nano:

https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit

2, For Jetson tx2

https://docs.nvidia.com/sdk-manager/install-with-sdkm-jetson/index.html#install-with-sdkm-jetson

## Install libjsoncpp:
sudo apt-get install libjsoncpp-dev

## Install libboots
sudo apt-get install libboost-all-dev

## Install Freetype to support display Vietnamese language:
tar xvfz freetype-2.9.1.tar.gz
cd freetype-2.9.1
sudo apt build-dep libfreetype6
./configure
make
sudo make install
sudo ldconfig

## Install Grpc c++

#### 1, Clone and checkout following commit from grpc c++ source
https://github.com/grpc/grpc/commit/7c0764918b9f33cab507ff483b4be849b0203ec4

#### 2, Follow the instruction bellow to build grpc from source
https://github.com/grpc/grpc/blob/master/BUILDING.md

#### 3, Install protobuf from source
grpc/third_party/protobuf


# if we get the error "all warnings being treated as errors" while compiling boringssl:
remove flag: "-Werror" from boringssl"s CMakeLists.txt

# if we get the error :
CMake Error at /usr/local/lib/cmake/grpc/gRPCConfig.cmake:8 (include):
  include could not find load file:

    /usr/local/lib/cmake/grpc/gRPCTargets.cmake
# use the cmake command
cmake -DgRPC_INSTALL=ON -DgRPC_BUILD_TESTS=OFF -DgRPC_PROTOBUF_PROVIDER=package -DgRPC_ZLIB_PROVIDER=package -DgRPC_CARES_PROVIDER=package -DgRPC_SSL_PROVIDER=package -DCMAKE_BUILD_TYPE=Release ../..




### on Jetson TX2
the same instruction with jetson nano. In additional:

# install protobuf from source :
https://stackoverflow.com/questions/43937682/upgrading-protobuf-tensorflow-on-jetson-tx2

# error
c-aresConfig.cmake
c-ares-config.cmake

cmake -DgRPC_INSTALL=ON -DgRPC_BUILD_TESTS=OFF -DgRPC_PROTOBUF_PROVIDER=package -DgRPC_ZLIB_PROVIDER=package -DgRPC_CARES_PROVIDER=package -DgRPC_SSL_PROVIDER=package -D_gRPC_CARES_LIBRARIES=cares -DgRPC_CARES_PROVIDER=kludge -DCMAKE_BUILD_TYPE=Release ../..
