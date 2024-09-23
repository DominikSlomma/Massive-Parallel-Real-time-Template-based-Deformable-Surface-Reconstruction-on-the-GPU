// #include "MeshMap.h"
// #include "Tracking.h"
// #include "viewer/Mesh_Visualizer.h"
#include "System.h"
#include "GT_compare/HamlynGT.h"
#include "utils.h"

// #include "viewer/MeshViewer.h"

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <yaml-cpp/yaml.h>
#include <cmath>


#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace stbr;

int main() {
    // frame_id
    int FrameNo = 0;
    // MeshViewer::MeshViewer *a;
    // a = new MeshViewer::MeshViewer();
    // a->run();

    // Config
    const YAML::Node config = YAML::LoadFile("../app/Phantom7/config.yaml");
    uint gt_id = int(round((double(FrameNo) / config["Hamlyn"]["FPS"].as<double>() + config["Hamlyn"]["addition"].as<double>()) * config["Hamlyn"]["multiplier"].as<double>())) % config["Hamlyn"]["modulo"].as<int>();
    HamlynGT* gt = new HamlynGT(config);
    // Creation of a mesh
    std::string video_file = config["System"]["video_file_path"].as<std::string>();
    // std::string obj_file_path = config["System"]["reference_file_path"].as<std::string>();


    cv::Mat frame;
    cv::VideoCapture cap(video_file);
    cap >> frame;
     std::vector<Eigen::Vector3d> vertices;
    std::vector<Eigen::Vector3i> triangles;
    utils::createRefMesh(config, gt_id, vertices, triangles);


    
    System *sys = new System(triangles, vertices, frame, config, gt); 
    
    int i=0;
    bool isTerminated = false;
    int num_img = 0;
    while(!isTerminated) {
        
        std::cout <<"img num: " <<  num_img << std::endl;
        num_img++;
        if(frame.empty())
            break;
        isTerminated = sys->monocular_feed(frame);

        int key = cv::waitKey(1);
        if (key == 'q')
        {
            std::cout << "q key is pressed by the user. Stopping the video" << std::endl;
            isTerminated = true;
        }
        cap >> frame;
        // if(100==i++)
        //     break;
        // isTerminated = true;
    
    }        

    double sum = 0;
    for (double element : gt->all_mean_) {
        sum += element;
    }
    double average = static_cast<double>(sum) / gt->all_mean_.size();
    std::cout << "RMS: " << average << std::endl;

    gt->all_mean_.push_back(-1);
    gt->all_mean_.push_back(average);

    utils::saveToCSV(gt->all_mean_, "hamlyn_f7_10K.csv");

    cap.release();
    return 0;
}

