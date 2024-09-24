#include "System.h"
#include "Viewer.h"
#include "Tracking.h"
#include "MeshMap.h"
#include "GroundTruth_compare.h"
#include "HamlynGT.h"
// #include "viewer/MeshViewer.h"
// #include "MeshViewer.h"
#include "database.h"

#include "optGPU.cuh"

#include <chrono>

namespace stbr {
System::System() {

}

System::System(std::vector<Eigen::Vector3i> ref_triangles, std::vector<Eigen::Vector3d> ref_vertices, cv::Mat ref_img, const YAML::Node &config, GroundTruth_compare *gt)
 : ref_triangles_(ref_triangles), ref_vertices_(ref_vertices), ref_img_(ref_img), config_(config), gt_(gt) {
    Eigen::Matrix3d K;
    K <<    config["Image"]["fx"].as<double>(), 0.000000, config["Image"]["cx"].as<double>(),
            0.000000, config["Image"]["fy"].as<double>(), config["Image"]["cy"].as<double>(),
            0.000000, 0.000000, 1.000000;
    
    tracking_ = new Tracking(ref_img_, ref_vertices_, ref_triangles_, config_);
    map_ = new MeshMap(ref_vertices_, ref_triangles_, config_);
    db_ = new database();

    tracking_->set_MeshMap(map_);
    map_->setTracking(tracking_, db_);

    viewer_ = new Viewer::MeshViewer(db_, ref_img_, ref_vertices_, ref_triangles_, config_);


    viewing_thread_ = std::unique_ptr<std::thread>(new std::thread(&Viewer::MeshViewer::run, viewer_));
}


bool System::monocular_feed(cv::Mat &img) {
    static int ii = 0;
    db_->setTexture(img);

    tracking_->track(img);
    map_->unordered_map();
    gt_->compareWithGroundTruth(map_->getVertices(), map_->getTriangles(), gt_pc_);
    db_->setVerticesAndTriangles(map_->getVertices(), map_->getTriangles());
    // db_->setVertices(map_->getVertices());
    db_->setGT(gt_pc_);

    // static int iii=0;
    // if (iii <= 10)
    //         std::this_thread::sleep_for(std::chrono::milliseconds(500));
    // else
    //         std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // iii++;

    // optimiserGPU a;
    // isTerminated();

    if(ii == 0){
        cv::waitKey(0);
        ii++;}

    while(db_->isPause()) {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
        if(db_->isTerminated())
            return true;
    }
    // exit(1);
// std::vector<Eigen::Vector3d> vert = map_->getVertices();
//     for (int i= 0; i< ref_vertices_.size(); i++) {
//         double d1 = vert[i].norm();
//         double d2 = ref_vertices_[i].norm();
//         double diff = d1-d2;
//         if(diff >)
//     }
// if(ii ==35){
//         cv::imwrite("test.png", img);
//         cv::waitKey(0);}
//     ii++;
    // if(ii == 0){
    //     cv::waitKey(0);
    //     ii++;
    // }
    
    return db_->isTerminated();
    // viewer_->UpdateMesh(img, map_->getVertices(), map_->getTriangles());

}

} // namespace