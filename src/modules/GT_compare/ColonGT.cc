#include "ColonGT.h"

#include <open3d/Open3D.h>
#include <open3d/t/geometry/RaycastingScene.h>
#include <memory>

#include <sstream>
#include <iostream>
#include <iomanip>

namespace stbr {
    ColonGT::ColonGT(const YAML::Node &config) : config_(config) {
        FrameNo_ = config["colonoscopy"]["start_id"].as<int>();
        max_number_ = config["colonoscopy"]["max_number_frames"].as<int>();
        path_ = config["colonoscopy"]["gt_path"].as<std::string>();
        
        K <<    config["Image"]["fx"].as<double>(), 0.000000, config["Image"]["cx"].as<double>(),
                0.000000, config["Image"]["fy"].as<double>(), config["Image"]["cy"].as<double>(),
                0.000000, 0.000000, 1.000000;
        fx = config["Image"]["fx"].as<double>();
        fy = config["Image"]["fy"].as<double>();
        cx = config["Image"]["cx"].as<double>();
        cy = config["Image"]["cy"].as<double>();
        width = config["Image"]["width"].as<double>();
        height = config["Image"]["height"].as<double>();
    }

    void ColonGT::compareWithGroundTruth(std::vector<Eigen::Vector3d> vertices, std::vector<Eigen::Vector3i> triangles, std::vector<Eigen::Vector3d> &gt_pc) {
        open3d::geometry::TriangleMesh tmp;
        std::shared_ptr<open3d::geometry::TriangleMesh> mesh = std::make_shared<open3d::geometry::TriangleMesh>(tmp);
        mesh->vertices_ = vertices;
        mesh->triangles_ = triangles;
        
        std::stringstream ss;
        ss << std::setw(5) << std::setfill('0') << FrameNo_;
        std::string result = ss.str();

        // auto pc = std::make_shared<open3d::geometry::PointCloud>(); 
        // auto pc = open3d::io::CreatePointCloudFromFile(path_ + std::to_string(gt_id) + ".txt", "xyz");
        // open3d::io::ReadPointCloud("/home/anonym/Schreibtisch/PhD/code/Sparse Template based Reconstruction/data/phi_SfT/real/S1/point_clouds/point_cloud_" + result + ".ply", *pc);
        // open3d::io::ReadPointCloud(path_ + result + ".ply", *pc);
        cv::Mat frame = cv::imread(path_ + result + ".exr", cv::IMREAD_ANYDEPTH);
        // std::cout << frame.size() << std::endl; exit(1);
        // Iteriere über alle Bildpunkte
        std::vector<Eigen::Vector3d> pts;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                // Pixelwert am aktuellen Punkt abrufen (für ein Graustufenbild)
                float d = frame.at<float>(y, x);
                double depth = (double)d;
                if (depth == 0)
                    continue;
                depth *= 5;
                // std::cout << depth << std::endl;
                Eigen::Vector3d tmp, uvt;
                uvt << x,y,1;
                // tmp = K_inv*uvt*(depth);
                tmp <<  (x-cx)*depth/fx, 
                        (y-cy)*depth/fy, 
                        depth; 
                pts.push_back(tmp);
                // std::cout <<  fx << " " << (x-cx)*depth/fx << " " << (y-cy)*depth/fy << " " << depth << std::endl;
                // Hier kannst du mit dem Pixelwert arbeiten (z.B. bearbeiten, anzeigen usw.)
                // Zum Beispiel: std::cout << "Pixelwert an (" << x << ", " << y << "): " << (int)pixel_value << std::endl;
            }
        }
        open3d::geometry::PointCloud pc_;
        pc_.points_ = pts;
        std::shared_ptr<open3d::geometry::PointCloud> pc = std::make_shared<open3d::geometry::PointCloud>(pc_);

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
            FrameNo_ = config_["colonoscopy"]["start_id"].as<int>();
    }
}