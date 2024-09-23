#include "Tracking.h"
#include "Extractor.cuh"
#include "MeshMap.h"
#include <iostream>
#include <algorithm>
#include <fstream>
namespace stbr {
Tracking::Tracking(cv::Mat &frame, Eigen::Matrix3d K, std::vector<Eigen::Vector3d> &vertices, std::vector<Eigen::Vector3i> &triangles, int thresholdValue) :  K_(K), fx_(K(0,0)), fy_(K(1,1)), cx_(K(0,2)), cy_(K(1,2)), pre_frame_(frame),
    vertices_(vertices), triangles_(triangles), number_triangles_(triangles_.size()), number_vertices_(vertices_.size())
{   
    
    // std::cout << "das hier\n"; exit(1);
    createMask(thresholdValue);

    // findUsableVerticies();
    
    // findUsableTriangles();
    
    createInitialObseration();
    extraction = new Extractor(frame, pixel_reference_, config_);
}

Tracking::Tracking(cv::Mat ref_img, std::vector<Eigen::Vector3d> ref_vertices, std::vector<Eigen::Vector3i> ref_triangles, const YAML::Node &config)
: ref_img_(ref_img), vertices_(ref_vertices), triangles_(ref_triangles), config_(config), number_triangles_(ref_triangles.size()), number_vertices_(ref_vertices.size()) {
    K_ <<    config["Image"]["fx"].as<double>(), 0.000000, config["Image"]["cx"].as<double>(),
            0.000000, config["Image"]["fy"].as<double>(), config["Image"]["cy"].as<double>(),
            0.000000, 0.000000, 1.000000;
    fx_= K_(0,0);
    fy_= K_(1,1);
    cx_= K_(0,2);
    cy_= K_(1,2);
    pre_frame_ = ref_img;
    // vertices_ = ref_vertices_;
    // triangles_ = ref_triangles_;
    
    createMask(config["Preprocessing"]["brightness_threshold"].as<int>());

    // findUsableVerticies();
    
    // findUsableTriangles();
    
    createInitialObseration();
    extraction = new Extractor(ref_img, pixel_reference_, config_);
    // createInitialObserationGPU();
    // extraction = new Extractor(ref_triangles, ref_img, pixel_reference_, config_);
}
        

void Tracking::set_MeshMap(MeshMap *unordered_map) {
    unordered_map_ = unordered_map;
    unordered_map_->set_Observation(obs);
}

std::vector<double> Tracking::getObservation() {
    return obs;
}

void Tracking::createInitialObserationGPU() {
    
   

    for(int vert_id=0; vert_id < vertices_.size(); vert_id++) {
        Eigen::Vector3d vertex = vertices_[vert_id];

        double x = vertex.x();
        double y = vertex.y();
        double z = vertex.z();

        double u = fx_*x/z + cx_;
        double v = fy_*y/z + cy_;

        obs.push_back(u);
        obs.push_back(v);
        h_pixel_reference_.push_back(u);
        h_pixel_reference_.push_back(v);
        h_pixel_correspondence_.push_back(0);
        h_pixel_correspondence_.push_back(0);
    }
    
}


void Tracking::createInitialObseration() {
    
   

    for(int vert_id=0; vert_id < vertices_.size(); vert_id++) {
        Eigen::Vector3d vertex = vertices_[vert_id];

        double x = vertex.x();
        double y = vertex.y();
        double z = vertex.z();

        double u = fx_*x/z + cx_;
        double v = fy_*y/z + cy_;

        obs.push_back(u);
        obs.push_back(v);
        pixel_reference_.push_back(cv::Point2f((u),(v)));

        // printf("%0.4f, %0.4f, %0.4f\n", x, y, z);


    }
    // exit(1);
    // for (int i=0; i< obs.size(); i++) {
    //         std::cout << obs[i] << std::endl;
    //     }
    //     exit(1);
    
}

void Tracking::getObs(std::vector<double> &observation) {
    observation = obs;
}


void Tracking::findUsableVerticies() {
    for(int i=0; i < number_vertices_; i++) {
        bool isPointUsable = false;
        double x = vertices_[i].x();
        double y = vertices_[i].y();
        double z = vertices_[i].z();
        
        double u = fx_*x/z + cx_;
        double v = fy_*y/z + cy_;
        if((u >= config_["Preprocessing"]["width_min"].as<int>()) && 
        (v >= config_["Preprocessing"]["height_min"].as<int>()) && 
        (u <= config_["Preprocessing"]["width_max"].as<int>()) && 
        (v <= config_["Preprocessing"]["height_max"].as<int>()) && (mask_.at<uchar>(int(v),int(u)) == 255)) {
            isPointUsable = true;
        }
        if(!config_["Preprocessing"]["create_mask"].as<bool>())
            isPointUsable = true;
        usable_vertices_.push_back(isPointUsable);
    }
}

void Tracking::findUsableTriangles() {

    for(int i=0;i < triangles_.size();i++) {
        bool isTriangleUsabel = false;
        int f1 = triangles_[i].x();
        int f2 = triangles_[i].y();
        int f3 = triangles_[i].z();

        if(usable_vertices_[f1] && usable_vertices_[f2] && usable_vertices_[f3]) {
            isTriangleUsabel = true;
        }
        usable_triangles_.push_back(isTriangleUsabel);
    }
}


void Tracking::createMask(int thresholdValue) {
    cv::Mat hsvImage;
    cv::cvtColor(pre_frame_, hsvImage, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> channels;
    cv::split(hsvImage, channels);
    cv::threshold(channels[2], mask_, thresholdValue, 255, cv::THRESH_BINARY);
}

void Tracking::draw_correspondence(cv::Mat &frame) {

//         static int  inumber = 0;
//         std::ostringstream oss;
//         oss << std::setw(3) << std::setfill('0') << inumber;
//         cv::Mat whiteImage(frame.rows, frame.cols, CV_8UC3, cv::Scalar(255, 255, 255));
//         cv::Mat binaryMask;
//         std::string filename = "../data/phi_SfT/real/S1/masks/mask_" + oss.str() + ".png";
//         // std::cout << filename << std::endl;
//         cv::Mat mask = cv::imread(filename, cv::IMREAD_GRAYSCALE);

// // // std::cout << mask << std::endl;
// //     static int ii = 0;  
// //     if ((ii == 21)) {
//         cv::threshold(mask, binaryMask, 200, 255, cv::THRESH_BINARY);
//         cv::Mat result;
//         cv::bitwise_and(frame, frame, whiteImage, binaryMask);
//         inumber++;
//         // cv::imshow("mask", mask);
//         if(inumber > config_["Phi_SfT"]["max_number_frames"].as<int>()-1) {
//             inumber = 0;
//         }
// //     // cv::imshow("Result", whiteImage);
// //     //     cv::waitKey(0); 
// //     }
//     cv::Mat tmp(frame.size(), CV_8UC3, cv::Scalar(255, 255, 255));
//     for (int y = 0; y < frame.rows; y++) {
//         for (int x = 0; x < frame.cols; x++) {
//             if (mask.at<uchar>(y, x) == 255) {
//                 tmp.at<cv::Vec3b>(y, x) = frame.at<cv::Vec3b>(y, x);
//             }
//         }
//     }

for(uint i = 0; i < pixel_reference_.size(); i++)
    {   
        // if (extraction->status[i] == 1) {
            cv::line(frame, pixel_correspondence_[i], pixel_reference_[i], cv::Scalar(0, 255, 0), 2);
            cv::circle(frame, pixel_correspondence_[i], 2, cv::Scalar(0, 0, 255), -1);
        // }
    }

    // for(uint i = 0; i < h_pixel_reference_.size() / 2; i++)
    // {   
    //     float u1 = h_pixel_correspondence_[i*2];
    //     float v1 = h_pixel_correspondence_[i*2+1];
    //     cv::Point2f pt_correspond(u1,v1);
        
    //     float u2 = h_pixel_reference_[i*2];
    //     float v2 = h_pixel_reference_[i*2+1];
    //     cv::Point2f pt_ref(u2,v2);
        
    //     // if (extraction->status[i] == 1) {
    //         cv::line(frame, pt_correspond, pt_ref, cv::Scalar(0, 255, 0), 2);
    //         cv::circle(frame, pt_correspond, 2, cv::Scalar(0, 0, 255), -1);
    //     // }
    // }

    // frame = tmp.clone();

    // if((ii == 21)) {
    //     cv::imshow("as", whiteImage);
    //     cv::imwrite(std::to_string(ii) + ".jpg", whiteImage);
    //     // cv::imshow("asss", mask);
    //     cv::waitKey(0);
    // }
    // ii++;
}

void Tracking::updateObservation() {
    for(int obs_id=0; obs_id < obs.size()/2; obs_id++) {

        double u = pixel_correspondence_[obs_id].x;
        double v = pixel_correspondence_[obs_id].y;

        obs[obs_id*2]   = u;
        obs[obs_id*2+1] = v;
    

    }

    // for (int i=0; i< obs.size()/2; i++) {
    //         std::cout << obs[i] << std::endl;
    //     }
    //     exit(1);
    // exit(1);

}

void Tracking::track(cv::Mat &frame) {
    cv::Mat modifiedFrame = frame.clone();

    extraction->extract(frame, pixel_correspondence_);
    updateObservation();

    this->draw_correspondence(modifiedFrame);
    // static int ii=0;
    // // cv::imwrite("/home/anonym/Schreibtisch/PhD/code/Sparse Template based Reconstruction/data/tmp/" + std::to_string(ii) + ".png", modifiedFrame);
    
    // if (ii == 1)
        // cv::waitKey(0);

    // ii++;
    cv::resize(modifiedFrame, modifiedFrame, cv::Size(), config_["Image"]["scale"].as<double>(), config_["Image"]["scale"].as<double>());
    cv::imshow("Frame", modifiedFrame);

}


void Tracking::track(cv::Mat &frame, std::vector<cv::Point2f> &pixel) {
    cv::Mat modifiedFrame = frame.clone();

    extraction->extract(frame, pixel_correspondence_);
    updateObservation();

    // this->draw_correspondence(modifiedFrame);

    cv::imshow("Frame", modifiedFrame);
    pixel = pixel_correspondence_;
    
    
    
    // static int FrameNo = 0;
    // std::ofstream file("obs_test3/obs_" + std::to_string(FrameNo) + ".txt");
    // FrameNo++;
    // for(int i=0; i < obs.size()/6; i++) {
    //     file <<  std::setprecision(20) << std::fixed <<  obs[i*6] << " " << obs[i*6+1] << " " << obs[i*6+2] << " " << obs[i*6+3] << " " << obs[i*6+4] << " " << obs[i*6+5] << std::endl;
    // }
    // file.close();

    // std::this_thread::sleep_for(std::chrono::milliseconds(100));
    // if(FrameNo == 39)
    //     exit(1);
    // cv::waitKey(0);

}
}