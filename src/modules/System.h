#ifndef SYSTEM_H
#define SYSTEM_H

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <vector>
#include <yaml-cpp/yaml.h>
#include <thread>

namespace Viewer {
    class MeshViewer;
}

namespace stbr {

class Tracking;
class MeshMap;
class GroundTruth_compare;
class database;

class System {
public:
    System();
    System(std::vector<Eigen::Vector3i> ref_triangles, std::vector<Eigen::Vector3d> ref_vertices, cv::Mat ref_img, const YAML::Node &config, GroundTruth_compare *gt);

    bool monocular_feed(cv::Mat &img); 

    Viewer::MeshViewer *viewer_ = nullptr;


private:
    bool terminate_requested();

    bool isTerminated_ = false;
    std::vector<Eigen::Vector3i> ref_triangles_;
    std::vector<Eigen::Vector3d> ref_vertices_;
    std::vector<Eigen::Vector3d> gt_pc_;
    cv::Mat ref_img_;
    const YAML::Node config_;

    Tracking *tracking_ = nullptr;
    MeshMap *map_ = nullptr;
    GroundTruth_compare *gt_ = nullptr;
    database *db_ = nullptr;

    std::unique_ptr<std::thread> viewing_thread_ = nullptr;

};
} // Namespace
#endif