
#include "System.h"
#include "GT_compare/SfTGT.h"
#include "utils.h"

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
// #include <open3d/Open3D.h>
// #include <open3d/t/geometry/RaycastingScene.h>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <yaml-cpp/yaml.h>
#include <cmath>
#include <sstream>

#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace stbr;

int main() {
    // frame_id
   

    // Config
    const YAML::Node config = YAML::LoadFile("../app/p_sft_S1_real/config.yaml");
    
    // Todo
    SfTGT* gt = new SfTGT(config);
    // Creation of a mesh
    std::string img_file_path = config["System"]["video_file_path"].as<std::string>();
    // std::string obj_file_path = config["System"]["reference_file_path"].as<std::string>();


    cv::Mat frame;

    std::vector<Eigen::Vector3d> vertices;
    std::vector<Eigen::Vector3i> triangles;
    utils::getMesh(config["System"]["reference_file_path"].as<std::string>(), vertices, triangles);
    


    frame = cv::imread(img_file_path + "000.png", cv::IMREAD_COLOR);

    System *sys = new System(triangles, vertices, frame, config, gt); 
    
    int max_number = config["Phi_SfT"]["max_number_frames"].as<int>();

    bool end = false;
    bool only_once = config["Phi_SfT"]["only_once"].as<bool>();
    bool isTerminated = false;
    while(!end && !isTerminated){
        frame = cv::imread(img_file_path + "000.png", cv::IMREAD_COLOR);
        for (int num_img=1;(num_img < max_number) && (!isTerminated); num_img++) {
            
            
            isTerminated = sys->monocular_feed(frame);
            std::stringstream ss;
            ss << std::setw(3) << std::setfill('0') << num_img;
            std::string result = ss.str();
            frame = cv::imread(img_file_path + result + ".png", cv::IMREAD_COLOR);

            int key = cv::waitKey(1);
            if (key == 'q')
            {
                std::cout << "q key is pressed by the user. Stopping the video" << std::endl;
                end = true;
                isTerminated = true;
            }


            
        }

        if(only_once)
            isTerminated = true;
        
    }
    double sum = 0;
    for (double element : gt->all_mean_) {
        sum += element;
    }
    double average = static_cast<double>(sum) / gt->all_mean_.size();

    std::cout << "average of RMSE: " << average << " start value: " << gt->all_mean_[0] << std::endl;



    return 0;
}

