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

    string camera_source;
    string multiple_camera_host;
    Screen *screen;

    CameraClient(string camera_source, string multiple_camera_host, string model_path) {
        this->camera_source = camera_source;
        this->multiple_camera_host = multiple_camera_host;
        this->rf = new RetinaFace(model_path, "net3");
        Display *d = XOpenDisplay(NULL);
        this->screen = DefaultScreenOfDisplay(d);
        CreateConnectionStream();
    }

    void CreateConnectionStream() {

        this->channel = grpc::CreateChannel(this->multiple_camera_host, grpc::InsecureChannelCredentials());
        this->stub_ = FaceProcessing::NewStub(channel);
        this->stream = this->stub_->recognize_face_js(new ClientContext);
    }

    void SendRequests() {
        cv::VideoCapture cap(camera_source);
        cv::Mat origin_image, display_image, cropedImage;
        vector<FaceDetectInfo> faceInfo;
        vector<LabeledFaceIn> facesOut;
        vector<TrackingBox> tmp_det;
        int max_age = 2, min_hits = 3;
        SORTtracker tracker(max_age, min_hits, 0.1);
        bool success, send_success = false, first_detections = true;
        int new_left, new_top, is_send = 0;
        float scale;
        double delay = 0, timer = 0;
        while (cap.read(origin_image)) {
            delay = ((double) getTickCount() - timer) * 1000.0 / cv::getTickFrequency();
            if (delay < 10) {
                    printf("IGNORE A frame\n");
                timer = (double) getTickCount();
                continue;
            }
            display_image = origin_image.clone();
            JSReq jsReq;
            is_send = is_send + 1;
            if (first_detections) {
                tracker.init(tmp_det);
                first_detections = false;
                float sw = 1.0 * origin_image.cols / 640;
                float sh = 1.0 * origin_image.rows / 640;
                scale = sw > sh ? sw : sh;
                scale = scale > 1.0 ? scale : 1.0;
            }
            if (is_send == 2) {
                is_send = 0;
            }
            if (is_send == 0) {
                tmp_det.clear();
                rf->detect(origin_image.clone(), 0.7, faceInfo, 640);
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
            }
            facesOut = this->work_queue.pop();
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
            timer = (double) getTickCount();
        }
    }

    void ReceiveResponses() {
        JSResp jSResp;
        vector<LabeledFaceIn> faces;
        LabeledFaceIn labeledFaceIn;
        LabeledFace labeledFace;
        bool receive_success;
        while (true) {
            receive_success = stream->Read(&jSResp);
            if (!receive_success){
                printf("CLIENT RECONNECT TO SERVER ...\n");
                this->CreateConnectionStream();
            }
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
    }

    std::thread SendRequestsThread() {
        return std::thread([this] {SendRequests();});
    }
    std::thread ReceiveResponsesThread() {
        return std::thread([this] {ReceiveResponses();});
    }

private:
    shared_ptr<Channel> channel;
    std::unique_ptr<FaceProcessing::Stub> stub_{};
    shared_ptr<ClientReaderWriter<JSReq, JSResp>> stream;
    RetinaFace *rf;
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
    CameraClient cameraClient(camera_source, multiple_camera_host, model_path);
    std::thread t1 = cameraClient.ReceiveResponsesThread();
    std::thread t2 = cameraClient.SendRequestsThread();
    t1.join();
    t2.join();
}