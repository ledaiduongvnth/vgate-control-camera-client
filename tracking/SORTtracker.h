///////////////////////////////////////////////////////////////////////////////
//  SORT: A Simple, Online and Realtime Tracker
//  
//  This is a C++ reimplementation of the open source tracker in
//  https://github.com/abewley/sort
//  Based on the work of Alex Bewley, alex@dynamicdetection.com, 2016
//
//  Cong Ma, mcximing@sina.cn, 2016
//  Rewritten by Beloborodov Dmitri (BeloborodovDS), 2017
//  
//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//  
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//  
//  You should have received a copy of the GNU General Public License
//  along with this program.  If not, see <http://www.gnu.org/licenses/>.
///////////////////////////////////////////////////////////////////////////////

#ifndef SORT_TRACKER
#define SORT_TRACKER

#include "Hungarian.h"
#include "KalmanTracker.h"

#include "opencv2/core/core.hpp"
#include <set>


// Computes IOU between two bounding boxes
double GetIOU(cv::Rect_<float> bb_test, cv::Rect_<float> bb_gt);

/* Bounding box of the SORT tracker
 * frame: current frame number
 * id: object/trajectory id 
 * age: frames since this object was detected last time
 * box: bounding box
 */
struct TrackingBox {
    int frame;
    int id;
    int age;
    cv::Rect_<float> box;
    std::vector<float> landmarks;

};


class SORTtracker {

public:
    //how many frames processed
    int frame_count;

    //SORT parameters
    int max_age;
    int min_hits;
    double iouThreshold;

    //Kalman filter trackers
    std::vector<KalmanTracker> trackers;

    //internal
    std::vector<cv::Rect_<float> > predictedBoxes;
    std::vector<std::vector<double> > iouMatrix;
    std::vector<int> assignment;
    std::set<int> unmatchedDetections;
    std::set<int> unmatchedTrajectories;
    std::set<int> allItems;
    std::set<int> matchedItems;
    std::vector<cv::Point> matchedPairs;
    unsigned int trkNum;
    unsigned int detNum;

    /* instantiate SORT tracker with parameters
     * maxage: maximum allowed object "age" (frames since it was detected last time)
     * minhits: minimum detection in a row to start tracking
     * iou_thresh: IOU threshold to match detected and tracked objects (if lower, not matched)
     */
    SORTtracker(int maxage = 1, int minhits = 3, float iou_thresh = 0.3);

    /* cleanup
     */
    ~SORTtracker();

    /* initialize tracker with fresh detections
     * detections: bounding boxes of objects
     */
    void init(std::vector<TrackingBox> detections);

    /* update tracker and get tracking results
     * detections: new bounding boxes of objects
     * results: tracked boxes with id and other properties
     */
    void step(std::vector<TrackingBox> detections, const cv::Size &img_size);

};

#endif