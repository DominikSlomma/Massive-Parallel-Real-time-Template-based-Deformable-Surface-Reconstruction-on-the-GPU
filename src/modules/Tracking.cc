#include "Tracking.h"
#include "Extractor.h"
#include "MeshMap.h"
#include <iostream>
#include <algorithm>
#include <fstream>
namespace stbr {
Tracking::Tracking(cv::Mat &frame, Eigen::Matrix3d K, std::vector<Eigen::Vector3d> &vertices, std::vector<Eigen::Vector3i> &triangles, int thresholdValue) :  K_(K), fx_(K(0,0)), fy_(K(1,1)), cx_(K(0,2)), cy_(K(1,2)), pre_frame_(frame),
    vertices_(vertices), triangles_(triangles), number_triangles_(triangles_.size()), number_vertices_(vertices_.size())
{   
    
    createMask(thresholdValue);
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

    
    createMask(config["Preprocessing"]["brightness_threshold"].as<int>());    
    createInitialObseration();
    extraction = new Extractor(ref_img, pixel_reference_, config_);
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

    }
   
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

    for(uint i = 0; i < pixel_reference_.size(); i++)
    {   
        // if (extraction->status[i] == 1) {
            cv::line(frame, pixel_correspondence_[i], pixel_reference_[i], cv::Scalar(0, 255, 0), 2);
            cv::circle(frame, pixel_correspondence_[i], 2, cv::Scalar(0, 0, 255), -1);
        // }
    }
   
}

void Tracking::updateObservation() {
    for(int obs_id=0; obs_id < obs.size()/2; obs_id++) {

        double u = pixel_correspondence_[obs_id].x;
        double v = pixel_correspondence_[obs_id].y;

        obs[obs_id*2]   = u;
        obs[obs_id*2+1] = v;
    

    }

}

void Tracking::track(cv::Mat &frame) {
    cv::Mat modifiedFrame = frame.clone();

    extraction->extract(frame, pixel_correspondence_);
    updateObservation();

    this->draw_correspondence(modifiedFrame);

    cv::resize(modifiedFrame, modifiedFrame, cv::Size(), config_["Image"]["scale"].as<double>(), config_["Image"]["scale"].as<double>());
    cv::imshow("Frame", modifiedFrame);

}


void Tracking::track(cv::Mat &frame, std::vector<cv::Point2f> &pixel) {
    cv::Mat modifiedFrame = frame.clone();

    extraction->extract(frame, pixel_correspondence_);
    updateObservation();


    cv::imshow("Frame", modifiedFrame);
    pixel = pixel_correspondence_;
    
    
    


}
}