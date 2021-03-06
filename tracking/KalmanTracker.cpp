///////////////////////////////////////////////////////////////////////////////
// KalmanTracker.cpp: KalmanTracker Class Implementation Declaration

#include "KalmanTracker.h"


int KalmanTracker::kf_count = 0;


// initialize Kalman filter
void KalmanTracker::init_kf(StateType stateMat) {
    int stateNum = 7;
    int measureNum = 4;
    kf = cv::KalmanFilter(stateNum, measureNum, 0);

    measurement = cv::Mat::zeros(measureNum, 1, CV_32F);

    kf.transitionMatrix = (cv::Mat_<float>(stateNum, stateNum) <<
                                                           1, 0, 0, 0, 1, 0, 0,
            0, 1, 0, 0, 0, 1, 0,
            0, 0, 1, 0, 0, 0, 1,
            0, 0, 0, 1, 0, 0, 0,
            0, 0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 0, 1);

    setIdentity(kf.measurementMatrix);
    setIdentity(kf.processNoiseCov, cv::Scalar::all(1e-2));
    setIdentity(kf.measurementNoiseCov, cv::Scalar::all(1e-1));
    setIdentity(kf.errorCovPost, cv::Scalar::all(1));

    // initialize state vector with bounding box in [cx,cy,s,r] style
    kf.statePost.at<float>(0, 0) = stateMat.x + stateMat.width / 2;
    kf.statePost.at<float>(1, 0) = stateMat.y + stateMat.height / 2;
    kf.statePost.at<float>(2, 0) = stateMat.area();
    kf.statePost.at<float>(3, 0) = stateMat.width / stateMat.height;
}


// Predict the estimated bounding box.
StateType KalmanTracker::predict() {
    // predict
    cv::Mat p = kf.predict();
    m_age += 1;

    if (m_time_since_update > 0)
        m_hit_streak = 0;
    m_time_since_update += 1;

    StateType predictBox = get_rect_xysr(p.at<float>(0, 0), p.at<float>(1, 0), p.at<float>(2, 0), p.at<float>(3, 0));

    m_history.push_back(predictBox);
    return m_history.back();
}


// Update the state vector with observed bounding box.
void KalmanTracker::update(StateType stateMat) {
    m_time_since_update = 0;
    m_history.clear();
    m_hits += 1;
    m_hit_streak += 1;

    if (m_hit_streak > m_hits_to_start)
        m_is_tracking = true;

    // measurement
    measurement.at<float>(0, 0) = stateMat.x + stateMat.width / 2;
    measurement.at<float>(1, 0) = stateMat.y + stateMat.height / 2;
    measurement.at<float>(2, 0) = stateMat.area();
    measurement.at<float>(3, 0) = stateMat.width / stateMat.height;

    // update
    kf.correct(measurement);
}


// Return the current state vector
StateType KalmanTracker::get_state() {
    cv::Mat s = kf.statePost;
    return get_rect_xysr(s.at<float>(0, 0), s.at<float>(1, 0), s.at<float>(2, 0), s.at<float>(3, 0));
}


// Convert bounding box from [cx,cy,s,r] to [x,y,w,h] style.
StateType KalmanTracker::get_rect_xysr(float cx, float cy, float s, float r) {
    float w = sqrt(s * r);
    float h = s / w;
    float x = (cx - w / 2);
    float y = (cy - h / 2);

    if (x < 0 && cx > 0)
        x = 0;
    if (y < 0 && cy > 0)
        y = 0;

    return StateType(x, y, w, h);
}

std::string KalmanTracker::randomString() {
    std::string str = "AAAAAA";

    // string sequence
    str[0] = rand() % 26 + 65;
    str[1] = rand() % 26 + 65;
    str[2] = rand() % 26 + 65;

    // number sequence
    str[3] = rand() % 10 + 48;
    str[4] = rand() % 10 + 48;
    str[5] = rand() % 10 + 48;

    return str;
}

void KalmanTracker::save(std::shared_ptr<ClientReaderWriter<JSReq, JSResp>> stream, bool is_save) {
    JSReq jsReq;
    UnlabeledFace *face = jsReq.add_faces();
    face->set_track_id(source_track_id);
    face->set_is_saving_history(is_save);
    bool send_success = stream->Write(jsReq);
    if (!send_success){
        printf("failed to send grpc\n");
        throw std::exception();
    }
}