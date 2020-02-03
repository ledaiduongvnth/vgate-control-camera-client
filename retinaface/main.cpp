#include <iostream>
#include <RetinaFace.h>
#include "timer.h"
#include <opencv2/videoio.hpp>

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
#include "base64.h"

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

using namespace std;

std::tuple<cv::Mat, int, int> CropFaceImageWithMargin(cv::Mat srcImg, int x1, int y1, int x2, int y2, float expanded_face_scale){
    int new_width = (int)((float)(x2 - x1) / 2 * expanded_face_scale);
    int new_height = (int)((float)(y2 - y1) / 2 * expanded_face_scale);
    int x_center = (x1 + x2) / 2;
    int y_center = (y1 + y2) / 2;
    int new_top = y_center - new_height > 0 ? (y_center - new_height) : 0;
    int new_left = x_center - new_width > 0 ? (x_center - new_width) : 0;
    cv::Mat face_image = srcImg(Rect(new_left, new_top, new_width * 2, new_height * 2));
    return std::make_tuple(face_image, new_left, new_top);
}

class CameraClient {
public:
    CameraClient(std::shared_ptr<Channel> channel):stub_(FaceProcessing::NewStub(channel)){}
    void RecognizeFace(RetinaFace* rf) {
        ClientContext context;
        std::shared_ptr<ClientReaderWriter<JSReq, JSResp> > stream(stub_->recognize_face_js(&context));
        std::thread writer([stream, rf]() {
            cv::VideoCapture cap(0);
            cv::Mat img;
            vector<FaceDetectInfo> faceInfo;
            cv::Mat cropedImage ;
            bool success;
            int new_left;
            int new_top;
            float scale;
            while(cap.read(img)) {
                tie(faceInfo, scale) = rf->detect(img, 0.9);
                tie(cropedImage, new_left, new_top) = CropFaceImageWithMargin(img,
                        faceInfo[0].rect.x1 * scale,
                        faceInfo[0].rect.y1 * scale,
                        faceInfo[0].rect.x2 * scale,
                        faceInfo[0].rect.y2 * scale,
                        1.3);
                JSReq jsReq;
                std::vector<uchar> buf;
                success = cv::imencode(".jpg", cropedImage, buf);
                auto* enc_msg = reinterpret_cast<unsigned char*>(buf.data());
                std::string encoded = base64_encode(enc_msg, buf.size());
                UnlabeledFace* face = jsReq.add_faces();
                face->set_track_id("sfsfsfsdfsdf");
                face->set_image_bytes(encoded);
                for(size_t j = 0; j < 5; j++) {
                    face->add_landmarks(faceInfo[0].pts.y[j] * scale - new_top);
                }
                for(size_t j = 0; j < 5; j++) {
                    face->add_landmarks(faceInfo[0].pts.x[j] * scale - new_left);
                }
                face->add_landmarks(0);
                stream->Write(jsReq);
                imshow("dst", img);
                waitKey(1);
            }
            stream->WritesDone();
        });
        JSResp jSResp;
        while (stream->Read(&jSResp)) {
            std::cout << "Got message \n" << jSResp.faces_size();
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
    string path = "../model";
    RetinaFace* rf = new RetinaFace(path, "net3");
    CameraClient client(grpc::CreateChannel(
            "localhost:50051", grpc::InsecureChannelCredentials()));
    client.RecognizeFace(rf);
    return 0;
}

