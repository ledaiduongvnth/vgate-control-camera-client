#!/bin/bash
protoc -I . --grpc_out=. --plugin=protoc-gen-grpc=`which grpc_cpp_plugin` multiple_camera_server.proto
protoc -I . --cpp_out=. multiple_camera_server.proto
