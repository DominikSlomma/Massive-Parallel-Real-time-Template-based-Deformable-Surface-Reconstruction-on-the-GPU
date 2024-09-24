#include "Extractor.h"

#include <iostream>
#include <chrono>

namespace stbr {
Extractor::Extractor(cv::Mat &frame, std::vector<cv::Point2f> &pixel_reference, const YAML::Node &config) : pixel_reference_(pixel_reference) {
    
    lk_iteration_ = config["Kanade"]["iteration"].as<int>();
    lk_width_ = config["Kanade"]["width"].as<int>();
    lk_height_ = config["Kanade"]["height"].as<int>();
    cv::cvtColor(frame, pre_frame_gray_, cv::COLOR_BGR2GRAY);
    
}


void Extractor::upload(const std::vector<cv::Point2f>& vec, cv::cuda::GpuMat& d_mat)
{
    cv::Mat mat(vec.size(), 1, CV_32FC2, (void*)vec.data());
    
    d_mat.create(1,vec.size(), CV_32FC2);
    d_mat.upload(vec);
}


void Extractor::download(const cv::cuda::GpuMat& d_mat, std::vector<cv::Point2f>& vec)
{
    vec.resize(d_mat.cols);
    cv::Mat mat(1, d_mat.cols, CV_32FC2, (void*)&vec[0]);
    d_mat.download(mat);
}

void Extractor::download(const cv::cuda::GpuMat& d_mat, std::vector<uchar>& vec)
{
    vec.resize(d_mat.cols);
    cv::Mat mat(1, d_mat.cols, CV_8UC1, (void*)&vec[0]);
    d_mat.download(mat);
}


void Extractor::extract(cv::Mat &frame, std::vector<cv::Point2f> &pixel_correspondence) {
        cv::Mat cur_frame_gray;
        cv::cvtColor(frame, cur_frame_gray, cv::COLOR_BGR2GRAY);
        cv::calcOpticalFlowPyrLK(pre_frame_gray_, cur_frame_gray, pixel_reference_, pixel_correspondence, status, err, cv::Size(lk_width_,lk_height_),lk_iteration_, criteria); // also 21,21 window would be good
}


} //namespace