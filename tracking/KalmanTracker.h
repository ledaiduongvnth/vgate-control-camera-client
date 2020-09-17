///////////////////////////////////////////////////////////////////////////////
// KalmanTracker.h: KalmanTracker Class Declaration

#ifndef KALMAN_H
#define KALMAN_H 2

#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "multiple_camera_server.grpc.pb.h"

using multiple_camera_server::JSReq;
using multiple_camera_server::JSResp;
using grpc::ClientReaderWriter;


using multiple_camera_server::UnlabeledFace;

#define StateType cv::Rect_<float>


// This class represents the internel state of individual tracked objects observed as bounding box.
class KalmanTracker {
public:
    KalmanTracker(int hits_to_start = 3) {
        init_kf(StateType());
        m_time_since_update = 0;
        m_hits = 0;
        m_hit_streak = 0;
        m_age = 0;
        m_id = kf_count;
        //kf_count++;

        m_is_tracking = false;
        m_hits_to_start = hits_to_start;

    }

    std::string randomString();

    KalmanTracker(StateType initRect, int hits_to_start = 3) {
        init_kf(initRect);
        m_time_since_update = 0;
        m_hits = 0;
        m_hit_streak = 0;
        m_age = 0;
        m_id = kf_count;
        kf_count++;

        m_is_tracking = false;
        m_hits_to_start = hits_to_start;
        source_track_id = randomString();
        name = "";
    }

    ~KalmanTracker() {
        m_history.clear();
    }

    StateType predict();

    void update(StateType stateMat);

    void save(std::shared_ptr<ClientReaderWriter<JSReq, JSResp>> stream, bool is_save);

    StateType get_state();

    StateType get_rect_xysr(float cx, float cy, float s, float r);

    static int kf_count;

    int m_time_since_update;
    int m_hits;
    int m_hit_streak;
    int m_age;
    int m_id;

    int m_hits_to_start;
    bool m_is_tracking;
    std::string source_track_id;
    std::string name;
    cv::Rect_<float> box;
    std::vector<float> landmarks;
    int init_frame_count;

private:
    void init_kf(StateType stateMat);

    cv::KalmanFilter kf;
    cv::Mat measurement;

    std::vector<StateType > m_history;
};


#endif