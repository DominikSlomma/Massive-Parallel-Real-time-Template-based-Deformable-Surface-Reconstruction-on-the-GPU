#include "GT_compare/SfTGT.h"

#include <open3d/Open3D.h>
#include <open3d/t/geometry/RaycastingScene.h>
#include <sstream>
#include <iostream>
#include <iomanip>

namespace stbr {
SfTGT::SfTGT(const YAML::Node &config) : config_(config) {
    FrameNo_ = 0;
    max_number_ = config["Phi_SfT"]["max_number_frames"].as<int>();
    path_ = config["Phi_SfT"]["gt_path"].as<std::string>();
}

// open3d::io::ReadPointCloud("/home/anonym/Schreibtisch/PhD/code/Sparse Template based Reconstruction/data/phi_SfT/real/S1/point_clouds/point_cloud_000.ply", *test);

void SfTGT::compareWithGroundTruth(std::vector<Eigen::Vector3d> vertices, std::vector<Eigen::Vector3i> triangles, std::vector<Eigen::Vector3d> &gt_pc) {
    open3d::geometry::TriangleMesh tmp;
    std::shared_ptr<open3d::geometry::TriangleMesh> mesh = std::make_shared<open3d::geometry::TriangleMesh>(tmp);
    mesh->vertices_ = vertices;
    mesh->triangles_ = triangles;
    
    std::stringstream ss;
    ss << std::setw(3) << std::setfill('0') << FrameNo_;
    std::string result = ss.str();

    auto pc = std::make_shared<open3d::geometry::PointCloud>(); 
    // auto pc = open3d::io::CreatePointCloudFromFile(path_ + std::to_string(gt_id) + ".txt", "xyz");
    // open3d::io::ReadPointCloud("/home/anonym/Schreibtisch/PhD/code/Sparse Template based Reconstruction/data/phi_SfT/real/S1/point_clouds/point_cloud_" + result + ".ply", *pc);
    open3d::io::ReadPointCloud(path_ + result + ".ply", *pc);
    std::vector<Eigen::Vector3d> tmp1;
    gt_pc.clear();
    for (int i=0;i<pc->points_.size();i++) {
        Eigen::Vector3d point = pc->points_[i];
        double x = point.x();
        double y = point.y();
        double z = point.z();
        if ((x == 0) && (y==0) && (z==0))
            continue;
        tmp1.push_back(pc->points_[i]);
        gt_pc.push_back(pc->points_[i]);
    }
    pc->points_ = tmp1;
    open3d::t::geometry::TriangleMesh t_mesh = open3d::t::geometry::TriangleMesh::FromLegacy(*mesh);
    open3d::t::geometry::PointCloud t_pc = open3d::t::geometry::PointCloud::FromLegacy(*pc);
    // t_mesh.Clear();

    auto scene = open3d::t::geometry::RaycastingScene() ;
    scene.AddTriangles(t_mesh);
    open3d::core::Tensor distances = scene.ComputeDistance(t_pc.GetPointPositions(),0);
    std::vector<float> values = distances.ToFlatVector<float>();

    float mean =0.0f;
    for (const auto& vector : values) {
        mean += vector*vector;
    }
    mean /= values.size();
    mean = sqrt(mean);
    all_mean_.push_back(mean);
    std::cout << mean << std::endl;
    FrameNo_++;
    if(FrameNo_ == max_number_)
        FrameNo_ = 0;
}
}