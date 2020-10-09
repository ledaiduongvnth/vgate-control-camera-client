#include <postProcessRetina.h>
#include <opencv2/videoio.hpp>
#include <string>
#include <jsoncpp/json/value.h>
#include <unistd.h>
#include <videoOutput.h>
#include <cudaResize.h>
#include <cudaFont.h>
#include "DrawText.h"
#include "gstCamera.h"
#include "retinaNet.h"
#include "cudaColorspace.h"

int main() {
    cv::Mat cropedImage;
    std::vector<FaceDetectInfo> faceInfo;
    float scale;
    postProcessRetina *rf = new postProcessRetina((std::string &) "model_path", "net3");
    videoOptions vo;
    vo.width = 1920;
    vo.height = 1080;
    vo.zeroCopy = true;
    vo.codec = videoOptions::CODEC_H264;
    videoSource* inputStream = videoSource::Create("rtsp://admin:abcd1234@172.16.10.84:554/Streaming/Channels/101", vo);
    if (!inputStream) {
        printf("failed to initialize camera device\n");
        throw std::exception();
    }
    if (!inputStream->Open()) {
        printf("failed to open camera for streaming\n");
        throw std::exception();
    }
    uchar3 * imgRGB8size1920x1080 = NULL;
    uchar3 * imgRGB8size640x360 = NULL;
    const size_t ImageSizeRGB8size640x360 = imageFormatSize(IMAGE_RGB8, 640, 360);
    if( !cudaAllocMapped((void**)&imgRGB8size640x360, ImageSizeRGB8size640x360)){
        printf("failed to allocate bytes for image\n");
    }
    float3 *imgRGB32size640x640 = NULL;
    const size_t ImageSizeRGB32size640x640 = imageFormatSize(IMAGE_RGB32F, 640, 640);
    if( !cudaAllocMapped((void**)&imgRGB32size640x640, ImageSizeRGB32size640x640)){
        printf("failed to allocate bytes for image\n");
    }
    videoOutput* outputStream = videoOutput::Create("display://0");
    cudaFont* font = cudaFont::Create(adaptFontSize(10));
    float sw = 1920 / 640;
    float sh = 1080 / 640;
    scale = sw > sh ? sw : sh;
    retinaNet net = retinaNet();

    while (1) {
        bool capSuccess = inputStream->Capture((void**)&imgRGB8size1920x1080, IMAGE_RGB8, 1000);
        if (!capSuccess) {
            printf("failed to capture frame\n");
            continue;
        }
        if(CUDA_FAILED(cudaResize(imgRGB8size1920x1080, 1920, 1080, imgRGB8size640x360, 640, 360))){
            printf(LOG_TRT "cudaResize failed\n");
            throw std::exception();
        }
        if( CUDA_FAILED(cudaConvertColor(imgRGB8size640x360, IMAGE_RGB8, imgRGB32size640x640, IMAGE_RGB32F, 640, 360))){
            printf("failed to convert color\n");
            throw std::exception();
        }
        CUDA(cudaDeviceSynchronize());
        std::vector<std::pair<std::string, int2>> labels;
        faceInfo.clear();
        net.Detect(imgRGB32size640x640, 1920, 1080, rf, faceInfo, 0.5);
        for (auto &t : faceInfo) {
            const int2  position = make_int2(t.rect.x1 * scale , t.rect.y1 * scale - 20);
            labels.push_back(std::pair<std::string, int2>("unknown", position));
        }
        font->OverlayText(imgRGB8size1920x1080, IMAGE_RGB8, 1920, 1080, labels, make_float4(255,0,0,255));
        if( outputStream != NULL ){
            outputStream->Render(imgRGB8size1920x1080, 1920, 1080);
            char str[256];
            sprintf(str, "Video Viewer (%ux%u) | %.1f FPS", 1920, 1080, outputStream->GetFrameRate());
            outputStream->SetStatus(str);
            if( !outputStream->IsStreaming() )
                break;
        }
    }
}