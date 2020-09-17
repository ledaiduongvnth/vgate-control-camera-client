//
// Created by d on 05/02/2020.
//

#ifndef CAMERA_CLIENT_IMAGE_PROC_H
#define CAMERA_CLIENT_IMAGE_PROC_H

#include <string>
#include <opencv2/opencv.hpp>
#include <codecvt>
#include <KalmanTracker.h>
#include <SORTtracker.h>
#include "DrawText.h"

std::tuple<cv::Mat, int, int>
CropFaceImageWithMargin(cv::Mat srcImg, int x1, int y1, int x2, int y2, float expanded_face_scale) {
    int new_width = (int) ((float) (x2 - x1) / 2 * expanded_face_scale);
    int new_height = (int) ((float) (y2 - y1) / 2 * expanded_face_scale);
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


void WriteTextAndBox(cv::Mat &displayImage, DrawText &drawer, SORTtracker sortTrackers) {
    for (auto it = sortTrackers.trackers.begin(); it != sortTrackers.trackers.end();) {
        cv::Rect_<float> pBox = (*it).box;
        if (pBox.x > 0 && pBox.y > 0 && pBox.x + pBox.width < displayImage.size().width &&
            pBox.y + pBox.height < displayImage.size().height && sortTrackers.frame_count - it->init_frame_count>3) {
            std::string displayName;
            cv::Scalar color;
            if (it->name.empty()) {
                displayName = "unknown";
                color = CV_RGB(255, 0, 0);
            } else {
                displayName = it->name;
                color = CV_RGB(0, 255, 0);
            }

            cv::Rect rect = cv::Rect(pBox.x, pBox.y, pBox.width, pBox.height);
            DrawRectangle(displayImage, rect, 3, 3, color);

            cv::Mat overlay = displayImage.clone();
            cv::Mat output = displayImage.clone();
            int baseline = 0;
            int text_height = drawer.pixel_width;
            int text_width = drawer.pixel_width * displayName.size() / 2;

            pBox.y = pBox.y - text_height / 2;
            pBox.x = pBox.x - text_width / 2 + pBox.width / 2;
            cv::rectangle(overlay, cv::Point(pBox.x, pBox.y) + cv::Point(0, baseline),
                          cv::Point(pBox.x, pBox.y) + cv::Point(text_width, -text_height), cv::Scalar(0), cv::FILLED);
            float alpha = 0.5;
            cv::addWeighted(overlay, alpha, output, 1 - alpha, 0, output);
            displayImage = output;

            std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
            std::wstring ws(displayName.size(), L' ');
            ws.resize(std::mbstowcs(&ws[0], displayName.c_str(), displayName.size()));
            drawer.PrintText(displayImage, ws, pBox.x, pBox.y, cv::Scalar(255, 255, 255));
        }
        it++;
    }
}

#endif //CAMERA_CLIENT_IMAGE_PROC_H
