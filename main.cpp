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
#include <opencv2/opencv.hpp>
#include "utils/base64.h"

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


JSReq MakeJSReq(){
    JSReq jsReq;
    cv::Mat img = cv::imread("/home/d/CLionProjects/vgate-control-camera-client/9cf798c3-07ec-4cda-a90c-3c37c0d3a492.jpeg");
    std::vector<uchar> buf;
    bool success = cv::imencode(".jpg", img, buf);
    auto *enc_msg = reinterpret_cast<unsigned char*>(buf.data());
    std::string encoded = base64_encode(enc_msg, buf.size());

    for (int i = 0; i < 2; ++i){
        UnlabeledFace* face = jsReq.add_faces();
        face->set_track_id("sfsfsfsdfsdf");
        face->set_image_bytes(encoded);
        for(int k = 0; k < 10; ++k){
            face->add_landmarks(k*10);
        }
        face->add_landmarks(0);
    }
    return jsReq;
}

class CameraClient {
public:
    CameraClient(std::shared_ptr<Channel> channel):stub_(FaceProcessing::NewStub(channel)){}
    void RecognizeFace() {
        ClientContext context;
        std::shared_ptr<ClientReaderWriter<JSReq, JSResp> > stream(stub_->recognize_face_js(&context));
        std::thread writer([stream]() {
            std::vector<JSReq> reqs{
                MakeJSReq(),
                MakeJSReq(),
                MakeJSReq(),

                MakeJSReq()
            };
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