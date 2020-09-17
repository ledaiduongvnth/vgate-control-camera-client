#ifndef RETINAFACE_H
#define RETINAFACE_H

#include <iostream>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>


struct anchor_win
{
    float x_ctr;
    float y_ctr;
    float w;
    float h;
};

struct anchor_box
{
    float x1;
    float y1;
    float x2;
    float y2;
};

struct FacePts
{
    float x[5];
    float y[5];
};

struct FaceDetectInfo
{
    float score;
    anchor_box rect;
    FacePts pts;
};

struct anchor_cfg
{
public:
    int STRIDE;
    std::vector<int> SCALES;
    int BASE_SIZE;
    std::vector<float> RATIOS;
    int ALLOWED_BORDER;

    anchor_cfg()
    {
        STRIDE = 0;
        SCALES.clear();
        BASE_SIZE = 0;
        RATIOS.clear();
        ALLOWED_BORDER = 0;
    }
};

class postProcessRetina
{
public:
    postProcessRetina(std::string &model, std::string network = "net3", float nms = 0.4);
    ~postProcessRetina();

    void detect(std::vector<std::vector<float>> results, float threshold, std::vector<FaceDetectInfo> &faceInfo, int model_size);
private:
    anchor_box bbox_pred(anchor_box anchor, cv::Vec4f regress);
    std::vector<anchor_box> bbox_pred(std::vector<anchor_box> anchors, std::vector<cv::Vec4f> regress);
    std::vector<FacePts> landmark_pred(std::vector<anchor_box> anchors, std::vector<FacePts> facePts);
    FacePts landmark_pred(anchor_box anchor, FacePts facePt);
    static bool CompareBBox(const FaceDetectInfo &a, const FaceDetectInfo &b);
    std::vector<FaceDetectInfo> nms(std::vector<FaceDetectInfo> &bboxes, float threshold);
private:


    float pixel_means[3] = {0.0, 0.0, 0.0};
    float pixel_stds[3] = {1.0, 1.0, 1.0};
    float pixel_scale = 1.0;

    std::string network;
    float nms_threshold;
    std::vector<float> _ratio;
    std::vector<anchor_cfg> cfg;

    std::vector<int> _feat_stride_fpn;
    std::map<std::string, std::vector<anchor_box>> _anchors_fpn;
    std::map<std::string, std::vector<anchor_box>> _anchors;

    std::map<std::string, int> _num_anchors;

};

#endif // RETINAFACE_H
