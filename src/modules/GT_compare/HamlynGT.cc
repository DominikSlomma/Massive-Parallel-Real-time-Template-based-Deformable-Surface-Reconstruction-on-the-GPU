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
    // Erstellen eines TriangleMesh-Objekts
    // static int cc = 0;
    // std::string filename = "/home/anonym/Schreibtisch/PhD/code/Sparse Template based Reconstruction/data/f7_defslam_gt/" + std::to_string(cc) + ".obj";
    // cc++;
    // std::shared_ptr<open3d::geometry::TriangleMesh> mesh123;
    //     // mesh = createRefMesh(config, gt_id);
    //     mesh123 = open3d::io::CreateMeshFromFile(filename);
    //     // int num = mesh123->vertices_.size();
    //     mesh123->RemoveUnreferencedVertices();
        // std::cout << num << " " << mesh123->vertices_.size() << std::endl;
        // if(num != mesh123->vertices_.size())
        //     std::cout << "asdas\n";
    // exit(1);
    

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
    std::cout << mean << " " << values.size() << std::endl;
    FrameNo_++;
    
    // cv::Mat mat(288, 360, CV_32F, values.data());
    // cv::Mat scaled_mat;
    // cv::normalize(mat, scaled_mat, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    // cv::Mat heatmap;
    // cv::applyColorMap(scaled_mat, heatmap, cv::COLORMAP_JET);

    // cv::Mat in[] = {cv::Mat::zeros(scaled_mat.rows, scaled_mat.cols, CV_8UC1), cv::Mat::zeros(scaled_mat.rows, scaled_mat.cols, CV_8UC1), scaled_mat};
    // cv::Mat in[] = {scaled_mat, scaled_mat, scaled_mat};
    // int from_to[] = {0,0, 1,1, 2,2};
    // cv::mixChannels(in, 3, &heatmap, 1, from_to, 3);
    
    // double mean=0;
    // std::vector<Eigen::Vector3d> pce = pc->points_;
    // int counter = 0;
    // for(int i=0;i<values.size();i++) {
    //     float val = values[i];
    //     if((pce[i].x() == 0) && (pce[i].y() == 0) && (pce[i].z() == 0))
    //         continue;
    //     counter++;
    //     mean += val;
    //     // std::cout << val << std::endl;
    // }
    // std::cout << mean/counter << " " << counter << std::endl;
    // cv::imshow("Heatmap", heatmap);
    // output = heatmap;
}
}