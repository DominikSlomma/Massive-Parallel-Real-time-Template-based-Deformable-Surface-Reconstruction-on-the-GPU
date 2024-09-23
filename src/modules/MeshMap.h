#ifndef MESHMAP_H
#define MESHMAP_H

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <vector>
#include <unordered_map>
#include <yaml-cpp/yaml.h>

namespace stbr {

class Tracking;
// class Optimizer;
class OptimizerDistanceOnly;
class OptimizerWithoutMiddlePoint;
class database;

class optGPU;
// class OptimizerVB;

class MeshMap {
    public:
        MeshMap(std::vector<Eigen::Vector3d> &vertices, std::vector<Eigen::Vector3i> &triangles, Eigen::Matrix3d K, int max_iteration, int optimization_algorithm, bool verbose);
        MeshMap(std::vector<Eigen::Vector3d> vertices, std::vector<Eigen::Vector3i>ref_triangles, const YAML::Node &config);
        
        ~MeshMap();
        void setTracking(Tracking *tracking, database *db);
        std::vector<Eigen::Vector3d>& getVertices();
        std::vector<Eigen::Vector3i>& getTriangles();

        void set_Observation(std::vector<double> &obs);

        void unordered_map();

    private:
        Tracking *tracking_ = nullptr;
        // Optimizer *optimizer_ = nullptr;
        OptimizerDistanceOnly *optimizerDistance_ = nullptr;
        OptimizerWithoutMiddlePoint *optimizeWithout_ = nullptr;
        optGPU *optGpu_ = nullptr;
        // OptimizerVB *optimizerVB_ = nullptr;

        std::vector<double> obs_;
        std::vector<Eigen::Vector3d> vertices_;
        std::vector<Eigen::Vector3d> new_vertices_;
        std::vector<Eigen::Vector3i> triangles_;
        Eigen::Matrix3d K_;
        const YAML::Node config_;
        double fx_ = 0;
        double fy_ = 0;
        double cx_ = 0;
        double cy_ = 0;
        int number_vertices_= 0;
        int number_triangles_ = 0;
        int optimization_algorithm_=0;

        std::unordered_map<int, int> triangle_unordered_mapping_;
        std::unordered_map<int, int> vertices_unordered_mapping_;

        database *db_ = nullptr;
        
};
} // namespace
#endif