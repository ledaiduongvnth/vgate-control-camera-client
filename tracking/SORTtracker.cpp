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


#include "Hungarian.h"
#include "KalmanTracker.h"
#include "opencv2/core/core.hpp"
#include <cfloat>
#include <map>

#include "SORTtracker.h"

using namespace std;

double GetIOU(Rect_<float> bb_test, Rect_<float> bb_gt) {
    float in = (bb_test & bb_gt).area();
    float un = bb_test.area() + bb_gt.area() - in;

    if (un < DBL_EPSILON)
        return 0;

    return (double) (in / un);
}

SORTtracker::SORTtracker(int maxage, int minhits, float iou_thresh) {
    frame_count = 0;
    max_age = maxage;
    min_hits = minhits;
    iouThreshold = iou_thresh;

    KalmanTracker::kf_count = 0; // tracking id relies on this, so we have to reset it in each seq.

    trkNum = 0;
    detNum = 0;
}

SORTtracker::~SORTtracker() {
    KalmanTracker::kf_count = 0; // tracking id relies on this, so we have to reset it in each seq.
}

void SORTtracker::init(vector<TrackingBox> detections) {
    //clear state
    KalmanTracker::kf_count = 0;

    trackers.clear();
    predictedBoxes.clear();
    iouMatrix.clear();
    assignment.clear();
    unmatchedDetections.clear();
    unmatchedTrajectories.clear();
    allItems.clear();
    matchedItems.clear();
    matchedPairs.clear();
    frame_count = 0;
    trkNum = 0;
    detNum = 0;
    //create a new Kalman filter tracker for each detection box
    for (unsigned int i = 0; i < detections.size(); i++) {
        KalmanTracker trk = KalmanTracker(detections[i].box, min_hits);
        trk.landmarks = detections[i].landmarks;
        trk.box = detections[i].box;
        trackers.push_back(trk);
    }
}

void SORTtracker::step(vector<TrackingBox> detections, const Size &img_size) {
    frame_count++;

    //for each Kalman tracker: try to predict next box; if failed, erase tracker
    predictedBoxes.clear();
    for (vector<KalmanTracker>::iterator it = trackers.begin(); it != trackers.end();) {
        Rect_<float> pBox = (*it).predict();
        if (pBox.x > 0 && pBox.y > 0 && pBox.x + pBox.width < img_size.width &&
            pBox.y + pBox.height < img_size.height) {
//            it->box = pBox;
            predictedBoxes.push_back(pBox);
            it++;
        } else {
            it = trackers.erase(it);
        }
    }

    //tracked boxes / detected boxes number
    trkNum = predictedBoxes.size();
    detNum = detections.size();

    //resize IOU matrix to be (trkNum x detNum)
    iouMatrix.clear();
    iouMatrix.resize(trkNum, vector<double>(detNum, 0));

    //compute 1-IOU for each pair of tracked/detected objects
    for (unsigned int i = 0; i < trkNum; i++) // compute iou matrix as a distance matrix
    {
        for (unsigned int j = 0; j < detNum; j++) {
            // use 1-iou because the hungarian algorithm computes a minimum-cost assignment.
            iouMatrix[i][j] = 1 - GetIOU(predictedBoxes[i], detections[j].box);
        }
    }

    // solve the assignment problem using hungarian algorithm.
    // the resulting assignment is [track(prediction) : detection], with len=preNum
    HungarianAlgorithm HungAlgo;
    assignment.clear();
    HungAlgo.Solve(iouMatrix, assignment);

    // find matches, unmatched_detections and unmatched_predictions
    unmatchedTrajectories.clear();
    unmatchedDetections.clear();
    allItems.clear();
    matchedItems.clear();

    if (detNum > trkNum) //there are unmatched detections
    {
        //allItems = {all detections}
        for (unsigned int n = 0; n < detNum; n++)
            allItems.insert(n);

        //matchedItems = {all assigned detections (best matches for each trajectory)}
        for (unsigned int i = 0; i < trkNum; ++i)
            matchedItems.insert(assignment[i]);

        //umnatchedItems = allItems - matchedItems
        set_difference(allItems.begin(), allItems.end(),
                       matchedItems.begin(), matchedItems.end(),
                       insert_iterator<set<int> >(unmatchedDetections, unmatchedDetections.begin()));
    } else if (detNum < trkNum) //there are unmatched trajectory/predictions
    {
        //unmatchedTrajectories = {trajectories that were not matched to any detection}
        for (unsigned int i = 0; i < trkNum; ++i)
            if (assignment[i] == -1) // unassigned label will be set as -1 in the assignment algorithm
                unmatchedTrajectories.insert(i);
    } else; //perfect match, nothing to do

    //collect matched pairs and filter out matches with low IOU
    matchedPairs.clear();
    for (unsigned int i = 0; i < trkNum; ++i) //for each trajectory
    {
        //skip not matched trajectories
        if (assignment[i] == -1)
            continue;
        //if low IOU: mark trajectory and detection as unmatched
        if (1 - iouMatrix[i][assignment[i]] < iouThreshold) {
            unmatchedTrajectories.insert(i);
            unmatchedDetections.insert(assignment[i]);
        } else //collect matched pairs
            matchedPairs.push_back(cv::Point(i, assignment[i]));
    }

    // update matched trackers with assigned detections.
    // each prediction is corresponding to a tracker
    int detIdx, trkIdx;
    for (unsigned int i = 0; i < matchedPairs.size(); i++) {
        trkIdx = matchedPairs[i].x;
        detIdx = matchedPairs[i].y;
        trackers[trkIdx].update(detections[detIdx].box);
        trackers[trkIdx].landmarks = detections[detIdx].landmarks;
        trackers[trkIdx].box = detections[detIdx].box;
    }

    // create and initialise new trackers for unmatched detections
    for (set<int>::iterator umd = unmatchedDetections.begin(); umd != unmatchedDetections.end(); umd++) {
        KalmanTracker tracker = KalmanTracker(detections[*umd].box, min_hits);
        tracker.landmarks = detections[*umd].landmarks;
        tracker.box = detections[*umd].box;
        trackers.push_back(tracker);
    }

    // get trackers' output
    for (vector<KalmanTracker>::iterator it = trackers.begin(); it != trackers.end();) {
        //if (time since update < 1) and (hits >= min_hits or not enough frames passed)
        //push Kalman filter's state (box) to result

        if ((*it).m_is_tracking && ((*it).m_time_since_update <= max_age)) {
            TrackingBox res;
            res.box = (*it).get_state();
            res.id = (*it).m_id + 1;
            res.frame = frame_count;
            res.age = (*it).m_time_since_update;
        }


        // create names vector
        vector<string> names;
        for (auto &it : trackers) {
            names.push_back(it.name);
        }

        //create vector of duplicate names
        std::map<std::string, int> countMap;
        // Iterate over the vector and store the frequency of each element in map
        for (auto & name : names)
        {
            auto result = countMap.insert(std::pair<std::string, int>(name, 1));
            if (result.second == false)
                result.first->second++;
        }
        vector<string> duplicateNames;
        for (auto & elem : countMap)
        {
            if (elem.second > 1)
            {
                duplicateNames.push_back(elem.first);
            }
        }

        // filter duplicated names
        if(std::find(duplicateNames.begin(), duplicateNames.end(), it->name) != duplicateNames.end()) {
            it->name = "";
        }

        // remove dead tracker (if time since update > max_age)
        if ((*it).m_time_since_update > max_age)
            it = trackers.erase(it);
        else
            it++;
    }

    return;
}

