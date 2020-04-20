//
// Created by d on 05/02/2020.
//

#ifndef CAMERA_CLIENT_UTIL_H
#define CAMERA_CLIENT_UTIL_H

#endif //CAMERA_CLIENT_UTIL_H

struct LabeledFaceIn {
    std::string track_id;
    std::string registration_id;
    std::string person_name;
    float confidence;
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

