//
// Created by d on 09/03/2020.
//

#include "drawavatar2lanes.h"

void drawavatar2lanes::DrawBoxes(cv::Mat &frame) {
    cv::rectangle(frame, cv::Point(leftWindow.x, leftWindow.y), cv::Point(leftWindow.x + leftWindow.width, leftWindow.y + leftWindow.height), cv::Scalar(0, 255,0));
    cv::rectangle(frame, cv::Point(rightWindow.x, rightWindow.y), cv::Point(rightWindow.x + rightWindow.width, rightWindow.y + rightWindow.height), cv::Scalar(0, 255,0));
}


