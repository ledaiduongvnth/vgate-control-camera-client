//
// Created by d on 05/02/2020.
//

#ifndef CAMERA_CLIENT_SORT_TRACKING_H
#define CAMERA_CLIENT_SORT_TRACKING_H

#include <iostream>
#include <fstream>
#include <iomanip>
#include <set>

#include "Hungarian.h"
#include "KalmanTracker.h"

#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"


typedef struct TrackingBox
{
    int frame;
    int id;
    Rect_<float> box;
    vector<float> landmarks;
};

double GetIOU(Rect_<float> bb_test, Rect_<float> bb_gt);

std::tuple<vector<KalmanTracker>, int> update_trackers(vector<KalmanTracker> trackers, int max_age, int min_hits, int frame_count, Size img_size, vector<TrackingBox> detFrameData, double iouThreshold);

#endif //CAMERA_CLIENT_SORT_TRACKING_H
