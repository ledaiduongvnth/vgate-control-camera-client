#BUILD INSTRUCTION

## Install default OS for the device
1, For Jetson nano:

https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit

2, For Jetson tx2

https://docs.nvidia.com/sdk-manager/install-with-sdkm-jetson/index.html#install-with-sdkm-jetson

## Install dependencies:

````shell script
sudo apt-get install -y libjsoncpp-dev libboost-all-dev libc-ares-dev libglew-dev libssl-dev cmake build-essential autoconf libtool pkg-config
````

#### Install Freetype to support display Vietnamese language:
````shell script
cd
curl -L  https://download.savannah.gnu.org/releases/freetype/freetype-2.9.1.tar.gz > freetype-2.9.1.tar.gz 
tar xvfz freetype-2.9.1.tar.gz
cd freetype-2.9.1
./configure
make
sudo make install
sudo ldconfig
````

## Install Grpc c++
````shell script
cd
git clone -b v1.23.0 https://github.com/grpc/grpc
cd grpc
git submodule update --init
`````

#### Install protobuf from source
````shell script
cd third_party/protobuf
mkdir -p cmake/build
cd cmake/build
cmake  -Dprotobuf_BUILD_TESTS=OFF ..
make -j4
sudo make install
sudo ldconfig
````


#### To avoid "all warnings being treated as errors" while compiling boringssl:
````shell script
cd ~/grpc/third_party/boringssl
sudo apt install nano
nano CMakeLists.txt
````
remove flag: "-Werror" on CMakeLists.txt

#### Build and install Grpc
````shell script
cd ~/grpc
mkdir -p cmake/build
cd cmake/build
cmake -DgRPC_INSTALL=ON -DgRPC_BUILD_TESTS=OFF -DgRPC_PROTOBUF_PROVIDER=package -DgRPC_ZLIB_PROVIDER=package -DgRPC_CARES_PROVIDER=package -DgRPC_SSL_PROVIDER=package -D_gRPC_CARES_LIBRARIES=cares -DgRPC_CARES_PROVIDER=kludge -DCMAKE_BUILD_TYPE=Release ../..
make -j4
sudo make install 
sudo ldconfig
````

## Build the project
#### 1, Clone the project source 
````shell script
cd
git clone https://github.com/ledaiduongvnth/vgate-control-camera-client.git
````

#### 2, Install jetson-utils lib:
````shell script
cd ~/vgate-control-camera-client/jetson-utils
mkdir build
cd build
cmake ..
make -j4
sudo make install 
sudo ldconfig
````

#### 3, Compile the source code:
````shell script
cd ~/vgate-control-camera-client
mkdir build
cd build
cmake ..
make -j4
````

#### 4, Run the binary:
````shell script
./camera_client
````


# Customize the display screen by the customer's requirements:
All the customization is performed inside of the vgate-control-camera-client/utils/image_proc.h
Given a list of face's boxes and face's labels Which is listFaces.
````C++
void WriteTextAndBox(cv::Mat &displayImage, DrawText &drawer, vector<KalmanTracker> listFaces) {
    for (auto it = trackers.begin(); it != trackers.end();) {
        /* Box contains face's coordinates Which is an Opencv Rect_, see more https://docs.opencv.org/3.4/d2/d44/classcv_1_1Rect__.html */
        Rect_<float> pBox = (*it).box;
        // label of the face
        it->name
        it++;
    }
}
````