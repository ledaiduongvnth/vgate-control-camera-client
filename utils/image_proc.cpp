#include <opencv2/opencv.hpp>

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