//
// Created by d on 05/02/2020.
//

#include "queue.h"

void WorkQueue::push_work(const std::vector <LabeledFaceIn> &item) {
    std::unique_lock<std::mutex> lock(work_mutex);
    bool was_empty = work.empty();
    work.push(item);
    lock.unlock();
    if (was_empty)
    {
        work_available.notify_one();
    }
}

std::vector <LabeledFaceIn> WorkQueue::pop() {
    std::vector<LabeledFaceIn> tmp = std::vector<LabeledFaceIn>();
    if (!work.empty()){
        tmp = work.front();
        work.pop();

    }
    return tmp;
}
