//
// Created by d on 05/02/2020.
//

#ifndef CAMERA_CLIENT_UTIL_H
#define CAMERA_CLIENT_UTIL_H

#endif //CAMERA_CLIENT_UTIL_H

struct LabeledFaceIn {
    std::string track_id;
    std::string registration_id;
    std::string person_name;
    float confidence;
};

std::string MakeCameraSource(std::string host, std::string cameraType, std::string userName, std::string passWord) {
    std::string cameraSource;
    if (cameraType == "hikvision") {
        cameraSource = "rtspsrc location=rtsp://" + host + "/101 user-id=" + userName + " user-pw=" + passWord +
                       " latency=0 ! rtph264depay !  h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw, format=BGRx";
    }
    if (cameraType == "axis") {
        cameraSource =
                "rtspsrc location=rtsp://" + host + ":554/axis-media/media.amp user-id=" + userName + " user-pw=" +
                passWord +
                " latency=0 ! rtph264depay !  h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw, format=BGRx";
    }
    if (cameraType == "webcam") {
        cameraSource = "dev/video0";
    }

    return cameraSource;
}

