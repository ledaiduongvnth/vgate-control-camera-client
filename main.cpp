#include <RetinaFace.h>
#include <opencv2/videoio.hpp>
#include <memory>
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
#include "SORTtracker.h"
#include <jsoncpp/json/value.h>
#include "jsoncpp/json/json.h"
#include "X11/Xlib.h"


using grpc::Channel;
using grpc::ClientContext;
using grpc::ClientReader;
using grpc::ClientReaderWriter;
using grpc::ClientWriter;
using multiple_camera_server::FaceProcessing;
using multiple_camera_server::JSReq;
using multiple_camera_server::JSResp;
using multiple_camera_server::LabeledFace;
using multiple_camera_server::UnlabeledFace;

class CameraClient {
public:
    CameraClient(std::shared_ptr<Channel> channel) : stub_(FaceProcessing::NewStub(channel)) {}

    void RecognizeFace(RetinaFace *rf, string camera_source, Screen* screen) {
        ClientContext context;
        std::shared_ptr<ClientReaderWriter<JSReq, JSResp> > stream(stub_->recognize_face_js(&context));
        std::thread writer([stream, rf, camera_source, screen, this]() {
            cv::VideoCapture cap(camera_source);
            cv::Mat origin_image, display_image, cropedImage;
            vector<FaceDetectInfo> faceInfo;
            vector<LabeledFaceIn> facesOut;
            vector<TrackingBox> tmp_det;
            int max_age = 2;
            int min_hits = 4;
            SORTtracker tracker(max_age, min_hits, 0.05);
            bool success, first_detections = true;
            int new_left, new_top, is_send = 0;
            float scale;
            while (cap.read(origin_image)) {
                display_image = origin_image.clone();
                tmp_det.clear();
                JSReq jsReq;
                tie(faceInfo, scale) = rf->detect(origin_image.clone(), 0.4);
                /* if there is any face in the image */
                is_send = is_send + 1;
                if (is_send == 4) {
                    is_send = 0;
                }
                facesOut = this->work_queue.pop();
                for (auto &t : faceInfo) {
                    TrackingBox trackingBox;
                    trackingBox.box.x = t.rect.x1 * scale;
                    trackingBox.box.y = t.rect.y1 * scale;
                    trackingBox.box.width = (t.rect.x2 - t.rect.x1) * scale;
                    trackingBox.box.height = (t.rect.y2 - t.rect.y1) * scale;
                    for (size_t j = 0; j < 5; j++) {
                        trackingBox.landmarks.push_back(t.pts.y[j] * scale);
                    }
                    for (size_t j = 0; j < 5; j++) {
                        trackingBox.landmarks.push_back(t.pts.x[j] * scale);
                    }
                    tmp_det.push_back(trackingBox);
                }
                if (first_detections) {
                    tracker.init(tmp_det);
                    first_detections = false;
                }
                tracker.step(tmp_det, origin_image.size());
                if (!faceInfo.empty()) {
                    for (auto it = tracker.trackers.begin(); it != tracker.trackers.end();) {
                        Rect_<float> pBox = (*it).box;
                        if (pBox.x > 0 && pBox.y > 0 && pBox.x + pBox.width < origin_image.size().width &&
                            pBox.y + pBox.height < origin_image.size().height) {
                            // attach detection results to the trackers
                            if (!facesOut.empty()) {
                                for (auto &k : facesOut) {
                                    if (k.track_id == it->source_track_id) {
                                        it->name = k.person_name;
                                    }
                                }
                            }
                            // end attach detection results to the trackers
                            // put text and draw rectangle
                            cv::putText(display_image, it->name, cv::Point(pBox.x, pBox.y), cv::FONT_ITALIC, 5.0,
                                        CV_RGB(0, 255, 0), 5);
                            cv::Rect rect = cv::Rect(pBox.x, pBox.y, pBox.width, pBox.height);
                            cv::rectangle(display_image, rect, Scalar(0, 0, 255), 5);
                            // end put text and draw rectangle
                            // get face image and landmarks to make request
                            tie(cropedImage, new_left, new_top) = CropFaceImageWithMargin(origin_image, pBox.x, pBox.y,
                                                                                          pBox.x + pBox.width,
                                                                                          pBox.y + pBox.height, 1.3);
                            UnlabeledFace *face = jsReq.add_faces();
                            std::vector<uchar> buf;
                            success = cv::imencode(".jpg", cropedImage, buf);
                            if (success) {
                                auto *enc_msg = reinterpret_cast<unsigned char *>(buf.data());
                                std::string encoded = base64_encode(enc_msg, buf.size());
                                face->set_track_id(it->source_track_id);
                                face->set_image_bytes(encoded);
                                for (size_t j = 0; j < 5; j++) {
                                    face->add_landmarks(it->landmarks[j] - (float) new_top);
                                }
                                for (size_t j = 5; j < 10; j++) {
                                    face->add_landmarks(it->landmarks[j] - (float) new_left);
                                }
                                face->add_landmarks(0);
                            }
                        }
                        it++;
                    }
                    if (is_send == 0) {
                        stream->Write(jsReq);
                    }
                }
                namedWindow("camera_client", WND_PROP_FULLSCREEN);
                setWindowProperty("camera_client", WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);
                resize(display_image, display_image, cv::Size(screen->width, screen->height));

                imshow("camera_client", display_image);
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
            if (!jSResp.faces().empty()) {
                for (int i = 0; i < jSResp.faces().size(); ++i) {
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
    }

private:
    std::unique_ptr<FaceProcessing::Stub> stub_{};
    WorkQueue work_queue;

};

int main(int argc, char **argv) {
    std::ifstream file_input("../config/config.json");
    Json::Reader reader;
    Json::Value configs;
    reader.parse(file_input, configs);
    string multiple_camera_host = configs["multiple-camera-host"].asString();
    string camera_source = configs["camera_source"].asString();
    string model_path = configs["model_path"].asString();
    Display* d = XOpenDisplay(NULL);
    Screen*  screen = DefaultScreenOfDisplay(d);
    RetinaFace *rf = new RetinaFace(model_path, "net3");
    CameraClient client(grpc::CreateChannel(multiple_camera_host, grpc::InsecureChannelCredentials()));
    client.RecognizeFace(rf, camera_source, screen);
    return 0;
}