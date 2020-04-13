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
#include <unistd.h>
#include "DrawText.h"
#include "gstCamera.h"


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

    std::string camera_source;
    std::string multiple_camera_host;
    cv::Size screenSize;
    int numberLanes;
    int detectionFrequency;
    int recognitionFrequency;
    int maxAge;
    int minHits;
    float iouThreash;
    float faceDetectThreash;
    int fontScale;
    int delayToIgnore;

    CameraClient(std::string camera_source, std::string multiple_camera_host, std::string model_path, int numberLanes, int detectionFrequency,
                 int recognitionFrequency, int maxAge, int minHits, float iouThreash,float faceDetectThreash, int fontScale, int delayToIgnore, cv::Size screenSize) {
        this->camera_source = camera_source;
        this->multiple_camera_host = multiple_camera_host;
        this->rf = new RetinaFace(model_path, "net3");
        this->screenSize = screenSize;
        this->numberLanes = numberLanes;
        this->detectionFrequency = detectionFrequency;
        this->recognitionFrequency = recognitionFrequency;
        this->faceDetectThreash = faceDetectThreash;
        this->maxAge = maxAge;
        this->minHits = minHits;
        this->iouThreash = iouThreash;
        this->fontScale = fontScale;
        this->delayToIgnore = delayToIgnore;
        CreateConnectionStream();
    }

    void CreateConnectionStream() {
        this->channel = grpc::CreateChannel(this->multiple_camera_host, grpc::InsecureChannelCredentials());
        this->stub_ = FaceProcessing::NewStub(channel);
        this->stream = this->stub_->recognize_face_js(new ClientContext);
    }

    void SendRequests() {
        cv::Mat origin_image, display_image, cropedImage;
        std::vector<FaceDetectInfo> faceInfo;
        std::vector<LabeledFaceIn> facesOut;
        std::vector<TrackingBox> tmp_det;
        SORTtracker tracker(this->maxAge, this->minHits, this->iouThreash);
        DrawText drawer(30) ;
        bool success, send_success, first_detections = true, capSuccess;
        int new_left, new_top, detectionCount = 0, recognitionCount = 0;
        float scale;
        double delay = 0, timer = 0;
        while (1) {
            capSuccess = this->imagesQueue.pop(origin_image);
            if(!capSuccess){
                continue;
            }
            display_image = origin_image.clone();
            JSReq jsReq;
            detectionCount = detectionCount + 1;
            if (detectionCount == this->detectionFrequency) {
                detectionCount = 0;
            }
            if (first_detections) {
                tracker.init(tmp_det);
                first_detections = false;
                float sw = 1.0 * origin_image.cols / 640;
                float sh = 1.0 * origin_image.rows / 640;
                scale = sw > sh ? sw : sh;
                scale = scale > 1.0 ? scale : 1.0;
            }
            if (detectionCount == 0) {
                tmp_det.clear();
                rf->detect(origin_image.clone(), this->faceDetectThreash, faceInfo, 640);
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
            this->facesQueue.pop(facesOut);
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
                        std::string displayName;
                        cv::Scalar color;
                        if (it->name.empty()){
                            displayName = "vô danh";
                            color = CV_RGB(255, 0, 0);
                        } else{
                            displayName = it->name;
                            color = CV_RGB(0, 255, 0);
                        }
                        WriteText(display_image, displayName, cv::Point(pBox.x, pBox.y), pBox.width, drawer);
                        cv::Rect rect = cv::Rect(pBox.x, pBox.y, pBox.width, pBox.height);
                        DrawRectangle(display_image, rect, 3, 3, color);
                        // end put text and draw rectangle
                        if (it->name.empty()){
                            // get face image and landmarks to make request
                            std::tie(cropedImage, new_left, new_top) = CropFaceImageWithMargin(origin_image,
                                    pBox.x, pBox.y,pBox.x + pBox.width,pBox.y + pBox.height, 1.3);
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
                    }
                    it++;
                }
                recognitionCount = recognitionCount + 1;
                if (recognitionCount == this->recognitionFrequency) {
                    recognitionCount = 0;
                }
                if (recognitionCount == 0) {
                    send_success = stream->Write(jsReq);
                    if (!send_success){
                        printf("failed to send grpc\n");
                        throw std::exception();
                    }
                }
            }
            resize(display_image, display_image, screenSize);
            namedWindow("camera_client", WND_PROP_FULLSCREEN);
            setWindowProperty("camera_client", WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);
            imshow("camera_client", display_image);
            waitKey(1);
        }
    }

    void ReceiveResponses() {
        JSResp jSResp;
        std::vector<LabeledFaceIn> faces;
        LabeledFaceIn labeledFaceIn;
        LabeledFace labeledFace;
        bool receive_success;
        while (true) {
            receive_success = stream->Read(&jSResp);
            if (!receive_success){
                printf("failed to receive grpc\n");
                throw std::exception();
            }
            if (!jSResp.faces().empty() && receive_success) {
                for (int i = 0; i < jSResp.faces().size(); ++i) {
                    labeledFace = jSResp.faces(i);
                    labeledFaceIn.track_id = labeledFace.track_id();
                    labeledFaceIn.registration_id = labeledFace.registration_id();
                    labeledFaceIn.person_name = labeledFace.person_name();
                    labeledFaceIn.confidence = labeledFace.confidence();
                    faces.emplace_back(labeledFaceIn);
                }
                this->facesQueue.push(faces);
            }
        }
    }
    void ReadImages(){
        gstCamera* camera = gstCamera::Create(1920, 1080, "rtspsrc location=rtsp://172.16.10.108/101 user-id=admin user-pw=123456a@ latency=0 ! rtph264depay !  h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw, format=BGRx, width=1920, height=1080");
        if( !camera )
        {
            printf("failed to initialize camera device\n");
            throw std::exception();
        }
        if( !camera->Open() )
        {
            printf("failed to open camera for streaming\n");
            throw std::exception();
        }
        double delay = 0, timer = 0;
        bool capSuccess;
        cv::Mat origin_image;
        while (1){
            float *imgRGBA = NULL;
            void *cpu = NULL;
            void *gpu = NULL;
            timer = (double) cv::getTickCount();
            capSuccess = camera->Capture(&cpu, &gpu, 1000);
            if (!capSuccess){
                printf("failed to capture frame\n");
            }
            capSuccess = camera->ConvertRGBA(gpu, &imgRGBA, false);
            if (!capSuccess) {
                printf("failed to convert from NV12 to RGBA\n");
            }
            cudaDeviceSynchronize();
            origin_image = cv::Mat(1080, 1920, CV_8UC3, (void *) cpu);
            delay = ((double) cv::getTickCount() - timer) * 1000.0 / cv::getTickFrequency();
            printf("%f\n", delay);
            if (!capSuccess){
                camera = gstCamera::Create(1920, 1080, "rtspsrc location=rtsp://172.16.10.108/101 user-id=admin user-pw=123456a@ latency=0 ! rtph264depay !  h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw, format=BGRx, width=1920, height=1080");
                if( !camera )
                {
                    printf("failed to initialize camera device\n");
                    throw std::exception();
                }
                if( !camera->Open() )
                {
                    printf("failed to open camera for streaming\n");
                    throw std::exception();
                }
                auto time = std::time(nullptr);
                printf("cap not success ");
                std::cout << "at: " << std::put_time(std::gmtime(&time), "%c") << '\n';
                continue;
            }
            this->imagesQueue.push(origin_image);
        }
    }
    std::thread ReadImagesThread() {
        return std::thread([this] {ReadImages();});
    }

    std::thread SendRequestsThread() {
        return std::thread([this] {SendRequests();});
    }
    std::thread ReceiveResponsesThread() {
        return std::thread([this] {ReceiveResponses();});
    }

private:
    std::shared_ptr<Channel> channel;
    std::unique_ptr<FaceProcessing::Stub> stub_{};
    std::shared_ptr<ClientReaderWriter<JSReq, JSResp>> stream;
    RetinaFace *rf;
    CConcurrentQueue<std::vector<LabeledFaceIn>> facesQueue;
    CConcurrentQueue<cv::Mat> imagesQueue;
};

int main(int argc, char **argv) {
    std::ifstream file_input("../config/config.json");
    Json::Reader reader;
    Json::Value configs;
    reader.parse(file_input, configs);
    std::string multiple_camera_host = configs["multiple_camera_host"].asString() + ":50052";
    int numberLanes = configs["strLane"].asInt();
    int detectionFrequency = configs["detection_frequency"].asInt();
    int recognitionFrequency = configs["recognition_frequency"].asInt();
    int maxAge = configs["max_age"].asInt();
    int minHits = configs["min_hits"].asInt();
    float faceDetectThreash = configs["face_detect_threash"].asFloat();
    float iouThreash = configs["iou_threash"].asFloat();
    int fontScale = configs["font_scale"].asInt();
    int delayToIgnore = configs["delay_to_ignore"].asInt();


    std::string camera_source;
    if (configs["hikvision"].asBool()){
        if (configs["use_gstreamer"].asBool()){
            camera_source = "rtspsrc location=rtsp://" + configs["camera_source"].asString() +
                            "/101 user-id=admin user-pw=123456a@ latency=0 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink";
        } else{
            camera_source = "rtsp://admin:123456a@@" + configs["camera_source"].asString() + ":554/Streaming/Channels/101";
        }
    } else{
        if (configs["use_gstreamer"].asBool()){
            camera_source = "rtspsrc location=rtsp://" + configs["camera_source"].asString() +
                            ":554/axis-media/media.amp user-id=root user-pw=abcd1234 latency=0 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink";
        } else{
            camera_source = "rtsp://root:abcd1234@" + configs["camera_source"].asString() + ":554/axis-media/media.amp";
        }
    }
    std::string model_path = configs["model_path"].asString();
    Screen *screen = DefaultScreenOfDisplay(XOpenDisplay(NULL));
    CameraClient cameraClient(camera_source, multiple_camera_host, model_path, numberLanes, detectionFrequency, recognitionFrequency,
                              maxAge, minHits, iouThreash, faceDetectThreash, fontScale, delayToIgnore, cv::Size(screen->width, screen->height));
    try {
        std::thread t0 = cameraClient.ReadImagesThread();
        std::thread t1 = cameraClient.ReceiveResponsesThread();
        std::thread t2 = cameraClient.SendRequestsThread();
        t0.join();
        t1.join();
        t2.join();
    } catch (const std::exception&){
        return 2;
    }
}