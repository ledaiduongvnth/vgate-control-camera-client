#include <chrono>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <thread>

#include <grpc/grpc.h>
#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>
#include "multiple_camera_server.grpc.pb.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::ClientReader;
using grpc::ClientReaderWriter;
using grpc::ClientWriter;
using grpc::Status;
using multiple_camera_server::FaceProcessing;
using multiple_camera_server::JSReq;
using multiple_camera_server::JSResp;
using multiple_camera_server::LabeledFace;
using multiple_camera_server::UnlabeledFace;

UnlabeledFace MakeUnlabeledFace(){
    UnlabeledFace face;
    face.set_track_id("sfsfsdf");
    face.set_image_bytes("sfsfdg");
    for( int i = 0; i < 10; i = i + 1 ) {
        face.set_landmarks(0, 32);
    }
    return face;
}

//JSReq MakeJSReq(){
//    JSReq jsReq;
//    *jsReq.mutable_faces() = {MakeUnlabeledFace(), MakeUnlabeledFace()};
//    return jsReq;
//}

class CameraClient {
public:
    CameraClient(std::shared_ptr<Channel> channel):stub_(FaceProcessing::NewStub(channel)){}
    void RecognizeFace() {
        ClientContext context;
        std::shared_ptr<ClientReaderWriter<JSReq, JSResp> > stream(stub_->recognize_face_js(&context));
        std::thread writer([stream]() {
            std::vector<JSReq> reqs{};
            for (const JSReq& req :reqs) {
                stream->Write(req);
            }
            stream->WritesDone();
        });

        JSResp jSResp;
        while (stream->Read(&jSResp)) {
            std::cout << "Got message " << jSResp.faces_size();
        }
        writer.join();
        Status status = stream->Finish();
        if (!status.ok()) {
            std::cout << "recognize_face_js rpc failed." << std::endl;
        }
    }

private:
    std::unique_ptr<FaceProcessing::Stub> stub_{};
};

int main(int argc, char** argv) {
    CameraClient client(grpc::CreateChannel(
            "localhost:50051", grpc::InsecureChannelCredentials()));
    client.RecognizeFace();
    return 0;
}