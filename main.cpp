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
#include <cudaResize.h>
#include <cudaFont.h>
#include "DrawText.h"
#include "gstCamera.h"
#include "retinaNet.h"
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
        this->net = new retinaNet();
        this->rotateImage = rotateImage;
    }

    void SendRequests() {
        cv::Mat cropedImage;
        std::vector<FaceDetectInfo> faceInfo;
        std::vector<LabeledFaceIn> facesOut;
        std::vector<TrackingBox> tmp_det;
        SORTtracker sortTrackers(this->maxAge, this->minHits, this->iouThreash);
        bool success, send_success, first_detections = true, capSuccess2;
        int new_left, new_top, detectionCount = 0, recognitionCount = 0;
        float scale;
        postProcessRetina *rf = new postProcessRetina((std::string &) "model_path", "net3");
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
        float3 * imgRGB32size1920x1080 = NULL;

        float3 *imgRGB32size640x640 = NULL;
        const size_t ImageSizeRGB32size640x640 = imageFormatSize(IMAGE_RGB32F, 640, 640);
        if( !cudaAllocMapped((void**)&imgRGB32size640x640, ImageSizeRGB32size640x640)){
            printf("failed to allocate bytes for image\n");
        }
        videoOutput* outputStream = videoOutput::Create("display://0");
        cudaFont* font = cudaFont::Create(adaptFontSize(20));

        while (1) {
            bool capSuccess = inputStream->Capture((void**)&imgRGB32size1920x1080, IMAGE_RGB32F, 1000);
            if (!capSuccess) {
                printf("failed to capture frame\n");
                continue;
            }
            if(CUDA_FAILED(cudaResize(imgRGB32size1920x1080, 1920, 1080, imgRGB32size640x640, 640, 360))){
                printf(LOG_TRT "imageNet::PreProcess() -- cudaResize failed\n");
                throw std::exception();
            }
            CUDA(cudaDeviceSynchronize());
            JSReq jsReq;
            /* Detect faces in an image */
            detectionCount = detectionCount + 1;
            if (detectionCount == this->detectionFrequency) {
                detectionCount = 0;
            }
            if (first_detections) {
                sortTrackers.init(tmp_det);
                first_detections = false;
                float sw = this->cameraWidth / 640;
                float sh = this->cameraHeight / 640;
                scale = sw > sh ? sw : sh;
            }
            std::vector<std::pair<std::string, int2>> labels;
            if (detectionCount == 0) {
                tmp_det.clear();
                faceInfo.clear();
                this->net->Detect(imgRGB32size640x640, this->cameraWidth, this->cameraHeight, rf, faceInfo, this->faceDetectThreash);
                for (auto &t : faceInfo) {
                    const int2  position = make_int2(t.rect.x1 * scale , t.rect.y1 * scale);
                    labels.push_back(std::pair<std::string, int2>("unknown", position));
                }
            }
            font->OverlayText(imgRGB32size1920x1080, IMAGE_RGB32F, 1920, 1080, labels, make_float4(255, 0, 0, 255));
            if( outputStream != NULL )
            {
                outputStream->Render(imgRGB32size1920x1080, this->cameraWidth, this->cameraHeight);

                // update status bar
                char str[256];
                sprintf(str, "Video Viewer (%ux%u) | %.1f FPS", this->cameraWidth, this->cameraHeight, outputStream->GetFrameRate());
                outputStream->SetStatus(str);

                // check if the user quit
                if( !outputStream->IsStreaming() )
                    break;
            }
        }
    }


    std::thread SendRequestsThread() {
        return std::thread([this] { SendRequests(); });
    }


private:
    CConcurrentQueue<std::vector<LabeledFaceIn>> facesQueue;
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
        std::thread t2 = cameraClient.SendRequestsThread();
        t2.join();
    } catch (const std::exception &) {
        return 2;
    }
}