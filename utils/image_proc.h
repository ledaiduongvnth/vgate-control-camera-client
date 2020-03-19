//
// Created by d on 05/02/2020.
//

#ifndef CAMERA_CLIENT_IMAGE_PROC_H
#define CAMERA_CLIENT_IMAGE_PROC_H

#endif //CAMERA_CLIENT_IMAGE_PROC_H
#include <string>
#include <opencv2/opencv.hpp>

std::tuple<cv::Mat, int, int> CropFaceImageWithMargin(cv::Mat srcImg, int x1, int y1, int x2, int y2, float expanded_face_scale);

void DrawRectangle(cv::Mat img, cv::Rect rect, int r, int thickness, cv::Scalar color);

