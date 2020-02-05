#include <iostream>
#include <RetinaFace.h>
#include <opencv2/videoio.hpp>
#include <chrono>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>
#include "multiple_camera_server.grpc.pb.h"
#include <opencv2/opencv.hpp>
#include "base64.h"
#include "image_proc.h"
#include "queue.h"

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

class CameraClient {
public:
    CameraClient(std::shared_ptr<Channel> channel):stub_(FaceProcessing::NewStub(channel)){}
    void RecognizeFace(RetinaFace* rf) {
        ClientContext context;
        std::shared_ptr<ClientReaderWriter<JSReq, JSResp> > stream(stub_->recognize_face_js(&context));
        std::thread writer([stream, rf, this]() {
            cv::VideoCapture cap(0);
            cv::Mat img;
            vector<FaceDetectInfo> faceInfo;
            cv::Mat cropedImage ;
            vector<LabeledFaceIn> facesOut;
            bool success;
            int new_left;
            int new_top;
            float scale;
            int is_send = 0;
            while(cap.read(img)) {
                JSReq jsReq;
                tie(faceInfo, scale) = rf->detect(img.clone(), 0.9);
                if (!faceInfo.empty()){
                    is_send = is_send + 1;
                    if (is_send == 4){
                        is_send = 0;
                    }
                    facesOut = this->work_queue.pop();
                    for (auto & t : faceInfo){
                        tie(cropedImage, new_left, new_top) = CropFaceImageWithMargin(img,
                                                                                      t.rect.x1 * scale,
                                                                                      t.rect.y1 * scale,
                                                                                      t.rect.x2 * scale,
                                                                                      t.rect.y2 * scale,1.3);
                        UnlabeledFace* face = jsReq.add_faces();
                        std::vector<uchar> buf;
                        success = cv::imencode(".jpg", cropedImage, buf);
                        if (success){
                            auto* enc_msg = reinterpret_cast<unsigned char*>(buf.data());
                            std::string encoded = base64_encode(enc_msg, buf.size());
                            char ch = 'A' + random()%26;
                            string track_id ;
                            track_id = ch;
                            face->set_track_id(track_id);
                            face->set_image_bytes(encoded);
                            for(size_t j = 0; j < 5; j++) {
                                face->add_landmarks(t.pts.y[j] * scale - new_top);
                            }
                            for(size_t j = 0; j < 5; j++) {
                                face->add_landmarks(t.pts.x[j] * scale - new_left);
                            }
                            face->add_landmarks(0);
                        }

                        cv::Rect rect = cv::Rect(cv::Point2f(t.rect.x1 * scale, t.rect.y1 * scale),
                                                 cv::Point2f(t.rect.x2 * scale, t.rect.y2 * scale));

                        cv::rectangle(img, rect, Scalar(0, 0, 255), 2);

                        for(size_t j = 0; j < 5; j++) {
                            cv::Point2f pt = cv::Point2f(t.pts.x[j] * scale, t.pts.y[j] * scale);
                            cv::circle(img, pt, 1, Scalar(0, 255, 0), 2);
                        }
                        if (!facesOut.empty()){
                            printf("result: %s\n", facesOut[0].person_name.c_str());
                            cv::putText(img,
                                        facesOut[0].person_name,
                                        cv::Point(t.rect.x1 * scale, t.rect.y1 * scale),
                                        cv::FONT_HERSHEY_DUPLEX,
                                        1.0,
                                        CV_RGB(118, 185, 0),
                                        2);
                        }

                    }
                    if (is_send == 0){
                        stream->Write(jsReq);
                    }
                }
                imshow("dst", img);
                waitKey(1);
            }
            stream->WritesDone();
        });
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        JSResp jSResp;
        vector<LabeledFaceIn> faces;
        LabeledFaceIn labeledFaceIn;
        LabeledFace labeledFace;
        while (stream->Read(&jSResp)) {
            if (!jSResp.faces().empty()){
                for (int i = 0; i < jSResp.faces().size(); ++i){
                    labeledFace = jSResp.faces(i);
                    labeledFaceIn.track_id = labeledFace.track_id();
                    labeledFaceIn.registration_id = labeledFace.registration_id();
                    labeledFaceIn.person_name = labeledFace.person_name();
                    labeledFaceIn.confidence = labeledFace.confidence();
                    faces.emplace_back(labeledFaceIn);
                }
                this->work_queue.push_work(faces);
            }
        }
        writer.join();
        Status status = stream->Finish();
        if (!status.ok()) {
            std::cout << "recognize_face_js rpc failed." << std::endl;
        }
    }
private:
    std::unique_ptr<FaceProcessing::Stub> stub_{};
    WorkQueue work_queue;

};

int main(int argc, char** argv) {
    string path = "../model";
    RetinaFace* rf = new RetinaFace(path, "net3");
    CameraClient client(grpc::CreateChannel(
            "localhost:50052", grpc::InsecureChannelCredentials()));
    client.RecognizeFace(rf);
    return 0;
}