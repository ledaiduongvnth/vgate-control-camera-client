//
// Created by d on 05/02/2020.
//

#ifndef CAMERA_CLIENT_QUEUE_H
#define CAMERA_CLIENT_QUEUE_H


#include <condition_variable>
#include <vector>
#include <queue>
#include "util.h"

class WorkQueue
{
    std::condition_variable work_available;
    std::mutex work_mutex;
    std::queue<std::vector<LabeledFaceIn>> work;

public:
    void push_work(const std::vector<LabeledFaceIn>& item);
    std::vector<LabeledFaceIn> pop();
};

#endif //CAMERA_CLIENT_QUEUE_H
