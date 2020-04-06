#include <opencv2/opencv.hpp>
#include "image_proc.h"


std::tuple<cv::Mat, int, int> CropFaceImageWithMargin(cv::Mat srcImg, int x1, int y1, int x2, int y2, float expanded_face_scale){
    int new_width = (int)((float)(x2 - x1) / 2 * expanded_face_scale);
    int new_height = (int)((float)(y2 - y1) / 2 * expanded_face_scale);
    int x_center = (x1 + x2) / 2;
    int y_center = (y1 + y2) / 2;
    int new_top = y_center - new_height > 0 ? (y_center - new_height) : 0;
    int new_bottom = y_center + new_height < srcImg.size().height ? (y_center + new_height) : srcImg.size().height;
    int new_left = x_center - new_width > 0 ? (x_center - new_width) : 0;
    int new_right = x_center + new_width < srcImg.size().width ? (x_center + new_width) : srcImg.size().width;
    cv::Mat face_image = srcImg(cv::Rect(cv::Point(new_left, new_top), cv::Point(new_right, new_bottom)));
    return std::make_tuple(face_image, new_left, new_top);
}

void DrawRectangle(cv::Mat img, cv::Rect rect, int r, int thickness, cv::Scalar color) {
    int x1 = rect.x;
    int x2 = rect.x + rect.width;
    int y1 = rect.y;
    int y2 = rect.y + rect.height;
    int d = std::min(abs(x1 - x2), abs(y1 - y2)) / 10;


    cv::line(img, cv::Point(x1 + r, y1), cv::Point(x1 + r + d, y1), color, thickness);
    cv::line(img, cv::Point(x1, y1 + r), cv::Point(x1, y1 + r + d), color, thickness);
    cv::ellipse(img, cv::Point(x1 + r, y1 + r), cv::Point(r, r), 180, 0, 90, color, thickness);

    cv::line(img, cv::Point(x2 - r, y1), cv::Point(x2 - r - d, y1), color, thickness);
    cv::line(img, cv::Point(x2, y1 + r), cv::Point(x2, y1 + r + d), color, thickness);
    cv::ellipse(img, cv::Point(x2 - r, y1 + r), cv::Point(r, r), 270, 0, 90, color, thickness);

    cv::line(img, cv::Point(x1 + r, y2), cv::Point(x1 + r + d, y2), color, thickness);
    cv::line(img, cv::Point(x1, y2 - r), cv::Point(x1, y2 - r - d), color, thickness);
    cv::ellipse(img, cv::Point(x1 + r, y2 - r), cv::Point(r, r), 90, 0, 90, color, thickness);

    cv::line(img, cv::Point(x2 - r, y2), cv::Point(x2 - r - d, y2), color, thickness);
    cv::line(img, cv::Point(x2, y2 - r), cv::Point(x2, y2 - r - d), color, thickness);
    cv::ellipse(img, cv::Point(x2 - r, y2 - r), cv::Point(r, r), 0, 0, 90, color, thickness);
}
