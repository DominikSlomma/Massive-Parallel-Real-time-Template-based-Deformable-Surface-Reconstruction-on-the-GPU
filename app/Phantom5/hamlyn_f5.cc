#include "System.h"
#include "GT_compare/HamlynGT.h"
#include "utils.h"
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

    // Config
    const YAML::Node config = YAML::LoadFile("../app/Phantom5/config.yaml");
    uint gt_id = int(round((double(FrameNo) / config["Hamlyn"]["FPS"].as<double>() + config["Hamlyn"]["addition"].as<double>()) * config["Hamlyn"]["multiplier"].as<double>())) % config["Hamlyn"]["modulo"].as<int>();
    HamlynGT* gt = new HamlynGT(config);
    // Creation of a mesh
    std::string video_file = config["System"]["video_file_path"].as<std::string>();


    cv::Mat frame;
    cv::VideoCapture cap(video_file);
    cap >> frame;
    std::vector<Eigen::Vector3d> vertices;
    std::vector<Eigen::Vector3i> triangles;
    utils::createRefMesh(config, gt_id, vertices, triangles);

    

    
    System *sys = new System(triangles, vertices, frame, config, gt); 
    
    
    bool isTerminated = false;
    int ii=0;
    while(!isTerminated) {
        if(frame.empty())
            break;
        isTerminated = sys->monocular_feed(frame);
        std::cout << "Frame Num: " << ii << std::endl; ii++;
        int key = cv::waitKey(1);
        if (key == 'q')
        {
            std::cout << "q key is pressed by the user. Stopping the video" << std::endl;
            isTerminated = true;
        }
        cap >> frame;

    
    }        

    double sum = 0;
    for (double element : gt->all_mean_) {
        sum += element;
    }
    double average = static_cast<double>(sum) / gt->all_mean_.size();
    std::cout << "average of RMSE: " << average << " start value: " << gt->all_mean_[0] << std::endl;

    cap.release();
    return 0;
}

