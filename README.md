#BUILD INSTRUCTION

## Install default OS for the device
1, For Jetson nano:

https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit

2, For Jetson tx2

https://docs.nvidia.com/sdk-manager/install-with-sdkm-jetson/index.html#install-with-sdkm-jetson

## Install libjsoncpp:
sudo apt-get install libjsoncpp-dev

## Install libboots
sudo apt-get install libboost-all-dev libc-ares-dev libglew-dev

## instal Openssl
sudo apt-get install libssl-dev 


## Install Freetype to support display Vietnamese language:
cd
curl -L  https://download.savannah.gnu.org/releases/freetype/freetype-2.9.1.tar.gz > freetype-2.9.1.tar.gz 
tar xvfz freetype-2.9.1.tar.gz
cd freetype-2.9.1
./configure
make
sudo make install
sudo ldconfig

## Install Grpc c++
cd
sudo apt-get install build-essential autoconf libtool pkg-config
sudo apt-get install cmake
git clone -b v1.23.0 https://github.com/grpc/grpc
cd grpc
git submodule update --init

#### 3, Install protobuf from source
cd third_party/protobuf
mkdir -p cmake/build
cd cmake/build
cmake  -Dprotobuf_BUILD_TESTS=OFF ..
make -j4
sudo make install
sudo ldconfig


#### To avoid "all warnings being treated as errors" while compiling boringssl:
cd ~/grpc/third_party/boringssl
sudo apt install nano
nano CMakeLists.txt
remove flag: "-Werror" on CMakeLists.txt

#### Build and install Grpc
cd ~/grpc
mkdir -p cmake/build
cd cmake/build
cmake -DgRPC_INSTALL=ON -DgRPC_BUILD_TESTS=OFF -DgRPC_PROTOBUF_PROVIDER=package -DgRPC_ZLIB_PROVIDER=package -DgRPC_CARES_PROVIDER=package -DgRPC_SSL_PROVIDER=package -D_gRPC_CARES_LIBRARIES=cares -DgRPC_CARES_PROVIDER=kludge -DCMAKE_BUILD_TYPE=Release ../..
make -j4
sudo make install 
sudo ldconfig

## Build the project
#### 1, Clone the project source 
cd
git clone https://github.com/ledaiduongvnth/vgate-control-camera-client.git

#### 2, Install jetson-utils lib:
cd ~/vgate-control-camera-client/jetson-utils
mkdir build
cd build
cmake ..
make -j4
sudo make install 
sudo ldconfig

#### 3, Compile the source code:
cd ~/vgate-control-camera-client
mkdir build
cd build
cmake ..
make -j4

#### 4, Run the binary:
./camera_client