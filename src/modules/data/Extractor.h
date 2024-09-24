#ifndef EXTRACTOR_H
#define EXTRACTOR_H

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>

#include <vector>
#include <yaml-cpp/yaml.h>


// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <cuda_runtime_api.h>
// #include "opencv2/core.hpp"
// #include <opencv2/core/utility.hpp>
// #include "opencv2/imgproc.hpp"
// #include "opencv2/highgui.hpp"
// #include "opencv2/video.hpp"
// #include "opencv2/cudaoptflow.hpp"
// #include "opencv2/cudaimgproc.hpp"


namespace stbr {
class Extractor {
    public:
        Extractor(cv::Mat &frame, std::vector<cv::Point2f> &pixel_reference, const YAML::Node &config);

        void extract(cv::Mat &frame, std::vector<cv::Point2f> &pixel_correspondence);
        std::vector<uchar> status;

    private:

        void download(const cv::cuda::GpuMat& d_mat, std::vector<cv::Point2f>& vec);
        void download(const cv::cuda::GpuMat& d_mat, std::vector<uchar>& vec);
        void upload(const std::vector<cv::Point2f>& vec, cv::cuda::GpuMat& d_mat);

        std::vector<Eigen::Vector3i> triangles_;

        cv::Mat pre_frame_gray_;
        std::vector<cv::Point2f> pixel_reference_;
        std::vector<float> err;
        cv::TermCriteria criteria = cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 10, 0.03);
        
        int lk_iteration_ = 10;
        int lk_width_ = 100;
        int lk_height_ = 100;


};
} // namespace

#endif

