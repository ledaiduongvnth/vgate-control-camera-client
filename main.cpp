#include <postProcessRetina.h>
#include <opencv2/videoio.hpp>
#include <memory>
#include <string>
#include <thread>
#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>
#include "multiple_camera_server.grpc.pb.h"
#include "image_proc.h"
#include "queue.h"
#include "SORTtracker.h"
#include <jsoncpp/json/value.h>
#include "jsoncpp/json/json.h"
#include "X11/Xlib.h"
#include <unistd.h>
#include <videoOutput.h>
#include "DrawText.h"
#include "gstCamera.h"
#include "retinaNet.h"
#include "fstream"
#include "base64.h"
#include "cudaColorspace.h"

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


struct Data {
    float3* imgRGB32;
    cv::Mat* OpenCVimg;
};

class CameraClient {
public:
    std::string camera_source;
    std::string multiple_camera_host;
    cv::Size screenSize;
    int detectionFrequency;
    int recognitionFrequency;
    int maxAge;
    int minHits;
    float iouThreash;
    float faceDetectThreash;
    int fontScale;
    int cameraWidth;
    int cameraHeight;
    int areaId;
    std::string direction;
    retinaNet* net;
    bool rotateImage;

    CameraClient(std::string camera_source, std::string multiple_camera_host, int areaId,
                 int detectionFrequency, int recognitionFrequency, int maxAge, int minHits, float iouThreash,
                 float faceDetectThreash, int fontScale, cv::Size screenSize, int cameraWidth, int cameraHeight,
                 std::string direction, bool rotateImage) {
        this->camera_source = camera_source;
        this->multiple_camera_host = multiple_camera_host;
        this->screenSize = screenSize;
        this->detectionFrequency = detectionFrequency;
        this->recognitionFrequency = recognitionFrequency;
        this->faceDetectThreash = faceDetectThreash;
        this->maxAge = maxAge;
        this->minHits = minHits;
        this->iouThreash = iouThreash;
        this->fontScale = fontScale;
        this->cameraWidth = cameraWidth;
        this->cameraHeight = cameraHeight;
        this->areaId = areaId;
        this->direction = direction;
        this->channel = grpc::CreateChannel(this->multiple_camera_host, grpc::InsecureChannelCredentials());
        this->stub_ = FaceProcessing::NewStub(channel);
        ClientContext *context = new ClientContext;
        context->AddMetadata("area_id", grpc::string(std::to_string(this->areaId)));
        context->AddMetadata("direction", grpc::string(this->direction));
        this->stream = this->stub_->recognize_face_js(context);
        this->net = new retinaNet();
        this->rotateImage = rotateImage;
    }

    void SendRequests() {
        cv::Mat cropedImage;
        std::vector<FaceDetectInfo> faceInfo;
        std::vector<LabeledFaceIn> facesOut;
        std::vector<TrackingBox> tmp_det;
        SORTtracker sortTrackers(this->maxAge, this->minHits, this->iouThreash);
        DrawText drawer(this->fontScale);
        bool success, send_success, first_detections = true, capSuccess2;
        int new_left, new_top, detectionCount = 0, recognitionCount = 0;
        float scale;
        postProcessRetina *rf = new postProcessRetina((std::string &) "model_path", "net3");
        while (1) {
//            Data* data;
            cv::Mat displayImage;
            float3* imgRGB32;
            capSuccess2 = this->imagesQueue.pop(displayImage);
            if (!capSuccess2) {
                continue;
            }
            capSuccess2 = this->imagesQueue2.pop(imgRGB32);
            if (!capSuccess2) {
                continue;
            }
            printf("this is originImage width %d, height %d\n", displayImage.cols, displayImage.rows);
            cv::cvtColor(displayImage, displayImage, cv::COLOR_RGB2BGR);
            JSReq jsReq;
            /* Detect faces in an image */
            detectionCount = detectionCount + 1;
            if (detectionCount == this->detectionFrequency) {
                detectionCount = 0;
            }
            if (first_detections) {
                sortTrackers.init(tmp_det);
                first_detections = false;
                float sw = 1.0 * displayImage.cols / 640;
                float sh = 1.0 * displayImage.rows / 640;
                scale = sw > sh ? sw : sh;
                scale = scale > 1.0 ? scale : 1.0;
            }
            if (detectionCount == 0) {
                tmp_det.clear();
                faceInfo.clear();
                this->net->Detect(imgRGB32, this->cameraWidth, this->cameraHeight, rf, faceInfo, this->faceDetectThreash);
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
            /* Detect faces in an image */
            this->facesQueue.pop(facesOut);
            // update tracking
            sortTrackers.step(tmp_det, displayImage.size(), stream);
            if (!faceInfo.empty()) {
                /* Make Grpc request and get face faces label from queue */
                for (auto it = sortTrackers.trackers.begin(); it != sortTrackers.trackers.end();) {
                    cv::Rect_<float> pBox = (*it).box;
                    if (pBox.x > 0 && pBox.y > 0 && pBox.x + pBox.width < displayImage.size().width &&
                        pBox.y + pBox.height < displayImage.size().height) {
                        if (!facesOut.empty()) {
                            for (auto &k : facesOut) {
                                if (k.track_id == it->source_track_id) {
                                    it->name = k.person_name;
                                }
                            }
                        }
                        if (it->name.empty()) {
                            std::tie(cropedImage, new_left, new_top) = CropFaceImageWithMargin(displayImage.clone(),
                                                                                               pBox.x, pBox.y,
                                                                                               pBox.x + pBox.width,
                                                                                               pBox.y + pBox.height,
                                                                                               1.4);
                            UnlabeledFace *face = jsReq.add_faces();
                            std::vector<uchar> buf;
                            success = cv::imencode(".jpg", cropedImage, buf);
                            if (success) {
                                auto *enc_msg = reinterpret_cast<unsigned char *>(buf.data());
                                std::string encoded = Base64Encode(enc_msg, buf.size());
                                face->set_track_id(it->source_track_id);
                                face->set_image_bytes(encoded);
                                face->set_is_saving_history(false);
                                for (size_t j = 0; j < 5; j++) {
                                    face->add_landmarks(it->landmarks[j] - (float) new_top);
                                }
                                for (size_t j = 5; j < 10; j++) {
                                    face->add_landmarks(it->landmarks[j] - (float) new_left);
                                }
                            }
                        }
                    }
                    it++;
                }
                /* Make Grpc request and get face faces label from queue */

                /* Send Grpc request */
                recognitionCount = recognitionCount + 1;
                if (recognitionCount == this->recognitionFrequency) {
                    recognitionCount = 0;
                }
                if (recognitionCount == 0) {
                    send_success = stream->Write(jsReq);
                    if (!send_success) {
                        printf("failed to send grpc\n");
                        throw std::exception();
                    }
                }
                /* Send Grpc request */

                /* Draw box and face label */
                WriteTextAndBox(displayImage, drawer, sortTrackers);
                /* Draw box and face label */
            }
            if(this->rotateImage){
                cv::Mat dst;
                cv::flip(displayImage, dst, -1);
                displayImage = dst;
            }
            cv::resize(displayImage, displayImage, screenSize);
            printf("this is screen width %d, height %d\n", screenSize.width, screenSize.height);
            printf("this is displayImage width %d, height %d\n", displayImage.cols, displayImage.rows);

            cv::namedWindow("camera_client", cv::WND_PROP_FULLSCREEN);
            cv::setWindowProperty("camera_client", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
            cv::imshow("camera_client", displayImage);
            cv::waitKey(1);
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
            if (!receive_success) {
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

    [[noreturn]] void ReadImages() {
        videoOptions vo;
        vo.width = this->cameraWidth;
        vo.height = this->cameraHeight;
        vo.zeroCopy = true;
        vo.codec = videoOptions::CODEC_H264;
        videoSource* inputStream = videoSource::Create(this->camera_source.c_str(), vo);
        if (!inputStream) {
            printf("failed to initialize camera device\n");
            throw std::exception();
        }
        if (!inputStream->Open()) {
            printf("failed to open camera for streaming\n");
            throw std::exception();
        }
        uchar3 * imgRGB8 = NULL;
        const size_t ImageSizeRGB8 = imageFormatSize(IMAGE_RGB8, this->cameraWidth, this->cameraHeight);
        if( !cudaAllocMapped((void**)&imgRGB8, ImageSizeRGB8)){
            printf("failed to allocate bytes for image\n");
        }
        while (true) {
            float3 *imgRGB32 = NULL;
            bool capSuccess = inputStream->Capture((void**)&imgRGB32, IMAGE_RGB32F, 1000);
            if (!capSuccess) {
                printf("failed to capture frame\n");
            } else {
                if( CUDA_FAILED(cudaConvertColor(imgRGB32, IMAGE_RGB32F, imgRGB8, IMAGE_RGB8, this->cameraWidth, this->cameraHeight))){
                    printf("failed to convert color");
                }
                CUDA(cudaDeviceSynchronize());
                cv::Mat originImage = cv::Mat(this->cameraHeight, this->cameraWidth, CV_8UC3, imgRGB8);
//                printf("this is originImage width %d, height %d\n", originImage.cols, originImage.rows);
//                Data data{imgRGB32, &originImage};
                this->imagesQueue.push(originImage.clone());
                this->imagesQueue2.push(imgRGB32);
            }
        }
    }

    std::thread ReadImagesThread() {
        return std::thread([this] { ReadImages(); });
    }

    std::thread SendRequestsThread() {
        return std::thread([this] { SendRequests(); });
    }

    std::thread ReceiveResponsesThread() {
        return std::thread([this] { ReceiveResponses(); });
    }

private:
    std::shared_ptr<Channel> channel;
    std::unique_ptr<FaceProcessing::Stub> stub_{};
    std::shared_ptr<ClientReaderWriter<JSReq, JSResp>> stream;
    CConcurrentQueue<std::vector<LabeledFaceIn>> facesQueue;
    CConcurrentQueue<cv::Mat> imagesQueue;
    CConcurrentQueue<float3*> imagesQueue2;
};

int main() {
    std::ifstream file_input("../config/config.json");
    Json::Reader reader;
    Json::Value configs;
    reader.parse(file_input, configs);
    std::string camera_source = configs["camera_source"].asString();
    int cameraWidth = configs["camera_width"].asInt();
    int cameraHeight = configs["camera_height"].asInt();
    bool rotateImage = configs["rotate_image"].asBool();
    std::string multiple_camera_host = configs["multiple_camera_host"].asString();
    int detectionFrequency = configs["detection_frequency"].asInt();
    int recognitionFrequency = configs["recognition_frequency"].asInt();
    int maxAge = configs["max_age"].asInt();
    int minHits = configs["min_hits"].asInt();
    float faceDetectThreash = configs["face_detect_threash"].asFloat();
    float iouThreash = configs["iou_threash"].asFloat();
    int fontScale = configs["font_scale"].asInt();
    int areaId = configs["area_id"].asInt();
    std::string direction = configs["direction"].asString();

    Screen *screen = DefaultScreenOfDisplay(XOpenDisplay(NULL));
    CameraClient cameraClient(camera_source, multiple_camera_host, areaId, detectionFrequency,
                              recognitionFrequency, maxAge, minHits, iouThreash, faceDetectThreash, fontScale,
                              cv::Size(screen->width, screen->height), cameraWidth, cameraHeight, direction, rotateImage);
    try {
        std::thread t0 = cameraClient.ReadImagesThread();
        std::thread t1 = cameraClient.ReceiveResponsesThread();
        std::thread t2 = cameraClient.SendRequestsThread();
        t0.join();
        t1.join();
        t2.join();
    } catch (const std::exception &) {
        return 2;
    }
}