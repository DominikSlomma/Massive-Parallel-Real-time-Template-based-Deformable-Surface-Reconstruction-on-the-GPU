#ifndef COLONGT_H
#define COLONGT_H

#include "GroundTruth_compare.h"
#include <yaml-cpp/yaml.h>
#include <string>

#include <opencv2/opencv.hpp>
#include <Eigen/Core>

namespace stbr {
class GroundTruth;

class ColonGT : public GroundTruth_compare {
public:
    ColonGT(const YAML::Node & config);

    void compareWithGroundTruth(std::vector<Eigen::Vector3d> vertices, std::vector<Eigen::Vector3i> triangles, std::vector<Eigen::Vector3d> &gt_pc) override;
private:
    const YAML::Node config_;
    int FrameNo_;
    int max_number_ = 0;
    std::string path_;

    Eigen::Matrix3d K;
    double fx;
    double fy;
    double cx;
    double cy;
    int width;
    int height;
};
}
#endif