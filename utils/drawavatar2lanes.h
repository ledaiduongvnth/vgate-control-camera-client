//
// Created by d on 09/03/2020.
//

#include <opencv2/opencv.hpp>


#ifndef CAMERA_CLIENT_DRAWAVATAR2LANES_H
#define CAMERA_CLIENT_DRAWAVATAR2LANES_H


class drawavatar2lanes {
public:
    drawavatar2lanes(int verticalMargin, int horizontalMargin, int distanceBetweenWindows, int screenWidth, int screenHeight,
    int windowWidth, int windowHeight){
        this-> verticalMargin = verticalMargin;
        this-> horizontalMargin = horizontalMargin;
        this-> distanceBetweenWindows = distanceBetweenWindows;
        this-> screenWidth = screenWidth;
        this-> screenHeight = screenHeight;
        this->windowWidth = windowWidth;
        this->screenHeight = windowHeight;
        this->leftWindow.x = horizontalMargin;
        this->leftWindow.y = verticalMargin;
        this->leftWindow.width = windowWidth;
        this->leftWindow.height = windowHeight;
        this->rightWindow.x = horizontalMargin + windowWidth + distanceBetweenWindows;
        this->rightWindow.y = verticalMargin;
        this->rightWindow.height = windowHeight;
        this->rightWindow.width = windowWidth;
    }
    int verticalMargin;
    int horizontalMargin;
    int distanceBetweenWindows;
    int screenWidth;
    int screenHeight;
    int windowWidth;
    int windowHeight;
    cv::Rect leftWindow;
    cv::Rect rightWindow;

    void DrawBoxes(cv::Mat& frame);

};


#endif //CAMERA_CLIENT_DRAWAVATAR2LANES_H
