#include "System.h"
#include "GT_compare/ColonGT.h"

#include "utils.h"

#include <opencv2/opencv.hpp>
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
    const YAML::Node config = YAML::LoadFile("../app/colon_2/config.yaml");
     
    // Todo
    ColonGT* gt = new ColonGT(config);
    // Creation of a mesh                   

    std::string img_file_path = config["System"]["video_file_path"].as<std::string>();
    std::string reference_path = config["System"]["reference_file_path"].as<std::string>();
    
    std::vector<Eigen::Vector3d> vertices;
    std::vector<Eigen::Vector3i> triangles;
    utils::getMeshColon(config, reference_path, vertices, triangles);

    cv::Mat frame;

    int start_id = config["colonoscopy"]["start_id"].as<int>();
    std::stringstream ss;
    ss << std::setw(5) << std::setfill('0') << start_id;
    std::string s_start_id = ss.str();
    frame = cv::imread(img_file_path + s_start_id + ".png");
    System *sys = new System(triangles, vertices, frame, config, gt); 

    int max_number = config["colonoscopy"]["max_number_frames"].as<int>();

    bool end = false;
    bool only_once = config["colonoscopy"]["only_once"].as<bool>();
    bool isTerminated = false;

    while(!end && !isTerminated){

        frame = cv::imread(img_file_path + ss.str() + ".png", cv::IMREAD_COLOR);
        
        for (int num_img=1;num_img < max_number && !isTerminated; num_img++) {
            
            std::cout << "frame Num: " << num_img << std::endl;            

            isTerminated = sys->monocular_feed(frame);
            std::stringstream ss;
            ss << std::setw(5) << std::setfill('0') << num_img;
            std::string result = ss.str();
            frame = cv::imread(img_file_path + result + ".png");

            int key = cv::waitKey(10);
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

