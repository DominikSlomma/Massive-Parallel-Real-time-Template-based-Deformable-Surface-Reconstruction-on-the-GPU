#include "GT_compare/HamlynGT.h"

#include <open3d/Open3D.h>
#include <open3d/t/geometry/RaycastingScene.h>
namespace stbr {

HamlynGT::HamlynGT(const YAML::Node &config) : config_(config) {
    FrameNo_ = 0;;
    FPS_ = config["Hamlyn"]["FPS"].as<int>();
    mutliplier_ = config["Hamlyn"]["multiplier"].as<double>();
    modulo_  = config["Hamlyn"]["modulo"].as<int>();
    addition_ = config["Hamlyn"]["addition"].as<double>(); 
    path_ =  config["Hamlyn"]["gt_path"].as<std::string>();
}

void HamlynGT::compareWithGroundTruth(std::vector<Eigen::Vector3d> vertices, std::vector<Eigen::Vector3i> triangles, std::vector<Eigen::Vector3d> &gt_pc) {

    

    open3d::geometry::TriangleMesh tmp;
    std::shared_ptr<open3d::geometry::TriangleMesh> mesh = std::make_shared<open3d::geometry::TriangleMesh>(tmp);
    mesh->vertices_ = vertices;//mesh123->vertices_;
    mesh->triangles_ = triangles;//mesh123->triangles_;
    uint gt_id = int(round((double(FrameNo_) / FPS_ + addition_)* mutliplier_)) % modulo_;
    auto pc = open3d::io::CreatePointCloudFromFile(path_ + std::to_string(gt_id) + ".txt", "xyz");
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
    std::cout << mean << " " << values.size() << std::endl;
    FrameNo_++;
}
}