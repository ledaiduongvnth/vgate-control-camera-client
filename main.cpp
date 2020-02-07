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
#include "base64.h"
#include "image_proc.h"
#include "queue.h"
#include "sort.h"

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
            cv::Mat img, cropedImage;
            vector<FaceDetectInfo> faceInfo;
            vector<LabeledFaceIn> facesOut;
            vector<TrackingBox> detFrameData;
            bool success;
            int new_left, new_top, frame_count, is_send = 0;
            float scale;
            vector<KalmanTracker> trackers;
            while(cap.read(img)) {
                detFrameData.clear();
                JSReq jsReq;
                tie(faceInfo, scale) = rf->detect(img.clone(), 0.9);
                /* if there is any face in the image */
                if (!faceInfo.empty()){
                    is_send = is_send + 1;
                    if (is_send == 4){
                        is_send = 0;
                    }
                    facesOut = this->work_queue.pop();
                    for (auto & t : faceInfo){
                        TrackingBox trackingBox;
                        trackingBox.box.x = t.rect.x1*scale;
                        trackingBox.box.y = t.rect.y1*scale;
                        trackingBox.box.width = (t.rect.x2 - t.rect.x1)*scale;
                        trackingBox.box.height = (t.rect.y2 - t.rect.y1)*scale;
                        for(size_t j = 0; j < 5; j++) {
                            trackingBox.landmarks.push_back(t.pts.y[j] * scale);
                        }
                        for(size_t j = 0; j < 5; j++) {
                            trackingBox.landmarks.push_back(t.pts.x[j] * scale);
                        }
                        detFrameData.push_back(trackingBox);
                    }
                    // update trackers
                    std::tie(trackers, frame_count) = update_trackers(trackers, 1, 1, frame_count, img.size(), detFrameData, 0.2);
                    for (auto it = trackers.begin(); it != trackers.end();)
                    {
                        Rect_<float> pBox = (*it).predict();
                        if (pBox.x >= 0 && pBox.y >= 0 && pBox.x + pBox.width < img.size().width && pBox.y + pBox.height < img.size().height)
                        {
                            // attach detection results to the trackers
                            if (!facesOut.empty()){
                                for (auto & k : facesOut){
                                    if (k.track_id == it->source_track_id){
                                        it->name = k.person_name;
                                    }
                                }
                            }
                            // end attach detection results to the trackers
                            // put text and draw rectangle
                            cv::putText(img, it->name,cv::Point(pBox.x, pBox.y),cv::FONT_HERSHEY_DUPLEX,1.0,CV_RGB(0, 255, 0),2);
                            cv::Rect rect = cv::Rect(pBox.x, pBox.y, pBox.width, pBox.height);
                            cv::rectangle(img, rect, Scalar(0, 0, 255), 2);
                            // end put text and draw rectangle
                            // get face image and landmarks to make request
                            tie(cropedImage, new_left, new_top) = CropFaceImageWithMargin(img, pBox.x, pBox.y, pBox.x + pBox.width, pBox.y + pBox.height,1.3);
                            UnlabeledFace* face = jsReq.add_faces();
                            std::vector<uchar> buf;
                            success = cv::imencode(".jpg", cropedImage, buf);
                            if (success){
                                auto* enc_msg = reinterpret_cast<unsigned char*>(buf.data());
                                std::string encoded = base64_encode(enc_msg, buf.size());
                                face->set_track_id(it->source_track_id);
                                face->set_image_bytes(encoded);
                                for(size_t j = 0; j < 5; j++) {
                                    face->add_landmarks(it->landmarks[j]-(float)new_top);
                                }
                                for(size_t j = 5; j < 10; j++) {
                                    face->add_landmarks(it->landmarks[j]-(float)new_left);
                                }
                                face->add_landmarks(0);
                            }
                            // end get face iamge and landmarks to make request
                            it++;
                        }
                        else
                        {
                            printf("delete track 1\n");
                            it = trackers.erase(it);
                        }
                    }
                    if (is_send == 0){
                        stream->Write(jsReq);
                    }
                }
                /* end if there is any face in the image */
                printf("number of trackers:%zu\n", trackers.size());
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