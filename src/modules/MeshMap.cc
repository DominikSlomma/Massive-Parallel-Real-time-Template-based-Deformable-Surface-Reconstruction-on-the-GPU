#include "MeshMap.h"
#include "Tracking.h"
// #include "Optimizer.h"
#include "OptimizerWithoutMiddlePoint.h"
#include "OptimizerDistanceOnly.h"

#include "optGPU.cuh"
// #include "OptimizerVB.h"


namespace stbr {

MeshMap::MeshMap(std::vector<Eigen::Vector3d> &vertices, std::vector<Eigen::Vector3i> &triangles, Eigen::Matrix3d K, int max_iteration, int optimization_algorithm, bool verbose): K_(K), fx_(K(0,0)), fy_(K(1,1)), cx_(K(0,2)), cy_(K(1,2)),
        vertices_(vertices), triangles_(triangles), number_triangles_(triangles.size()), number_vertices_(vertices.size()), optimization_algorithm_(optimization_algorithm) {
            
            
            // if (optimization_algorithm_ == 0)
            //     optimizerDistance_ = new OptimizerDistanceOnly(max_iteration, vertices, triangles_, K, verbose);
            // else if (optimization_algorithm_ == 1)
            //     optimizeWithout_ = new OptimizerWithoutMiddlePoint(max_iteration, vertices_, triangles_, verbose, K_);
            
    
                
        }

MeshMap::MeshMap(std::vector<Eigen::Vector3d> vertices, std::vector<Eigen::Vector3i>triangles, const YAML::Node &config) 
: vertices_(vertices), triangles_(triangles), config_(config) {
    K_ <<    config["Image"]["fx"].as<double>(), 0.000000, config["Image"]["cx"].as<double>(),
            0.000000, config["Image"]["fy"].as<double>(), config["Image"]["cy"].as<double>(),
            0.000000, 0.000000, 1.000000;
    fx_= K_(0,0);
    fy_= K_(1,1);
    cx_= K_(0,2);
    cy_= K_(1,2);

    number_triangles_ = triangles.size();
    number_vertices_ = vertices.size();
    optimization_algorithm_ = config["System"]["optimization_algorithm"].as<int>();

    int max_iteration = config["Optimizer"]["max_iteration"].as<int>();
    int optimization_algorithm = config["System"]["optimization_algorithm"].as<int>();
    bool verbose = config["System"]["verbose"].as<bool>();

    optGpu_ = new optGPU(max_iteration, vertices, triangles, K_, verbose);

    for (int i=0; i < vertices.size(); i++) {
        new_vertices_.push_back(vertices[i]);
    }

    // std::cout << optimization_algorithm_ << std::endl; exit(1);
    // if (optimization_algorithm_ == 0)
    //     optimizerDistance_ = new OptimizerDistanceOnly(max_iteration, vertices_, triangles_, K_, verbose);
    // else if (optimization_algorithm_ == 1)
    //     optimizeWithout_ = new OptimizerWithoutMiddlePoint(max_iteration, vertices_, triangles_, verbose, K_);

    
}

MeshMap::~MeshMap() {
    // delete everything
}

void MeshMap::set_Observation(std::vector<double> &obs) {
    obs_ = obs;
}


void MeshMap::setTracking(Tracking *tracking, database *db) {

    optGpu_->setDB(db);
    db_ = db;
    tracking_ = tracking;
}

std::vector<Eigen::Vector3d>& MeshMap::getVertices() {
    return new_vertices_;
}

std::vector<Eigen::Vector3i>& MeshMap::getTriangles() {
    return triangles_;
}

void MeshMap::unordered_map() {
    // tracking_->getObs(obs_);
    // int pre_id =-1;
    // for(int i=0; i<obs_.size()/6; i++) {
    //     int id = obs_[i*6];
        
    //     if(id == pre_id)
    //         continue;
    //     pre_id = id;
        
    //     triangle_unordered_mapping_[id] = 0;
    //     Eigen::Vector3i triangle = triangles_[id];
    //     vertices_unordered_mapping_[triangle.x()] = 0;
    //     vertices_unordered_mapping_[triangle.y()] = 0;
    //     vertices_unordered_mapping_[triangle.z()] = 0;
    // }
    // int num_vertices = vertices_unordered_mapping_.size();
    // int num_triangles = triangle_unordered_mapping_.size();

    // // double optimize_vertices[num_vertices*3];
    // // double optimize_triangles[num_triangles*3];

    // int counter = 0;
    // for(auto& unordered_map: triangle_unordered_mapping_) {

    //     unordered_map.second = counter; 
    //     counter++;
    // }
    // counter = 0;
    // for(auto& unordered_map: vertices_unordered_mapping_) {

    //     unordered_map.second = counter;
    //     counter++;
    // }

    // double vertices[num_vertices*3];
    //  for(auto& unordered_map: vertices_unordered_mapping_) {
    //     Eigen::Vector3d vertex = vertices_[unordered_map.first];
    //     // std::cout << num_vertices << std::endl;
    //     int id = unordered_map.second;
    //     vertices[id * 3] = vertex.x();
    //     vertices[id * 3 + 1] = vertex.y();
    //     vertices[id * 3 + 2] = vertex.z();
    // }
    
    // std::vector<int> triangles;
    // for(auto& unordered_map: triangle_unordered_mapping_) {
    //     Eigen::Vector3i triangle = triangles_[unordered_map.first];
    //     triangles.push_back(vertices_unordered_mapping_[triangle.x()]);
    //     triangles.push_back(vertices_unordered_mapping_[triangle.y()]);
    //     triangles.push_back(vertices_unordered_mapping_[triangle.z()]);
    // }


    // std::cout << obs_[0] << " " << obs_[1] << " " << number_vertices_<< std::endl; exit(1);
    tracking_->getObs(obs_);
    optGpu_->setParamater(obs_, obs_.size());
    optGpu_->run();
    optGpu_->getVertices(new_vertices_);

   


    // std::cout << vertices_.size() << " " << triangles_.size() << std::endl;
    // std::cout << vertices_unordered_mapping_.size() << " " << triangle_unordered_mapping_.size() << std::endl;
    
    // exit(1);
    
    // if (optimization_algorithm_ == 0) {
    //     optimizerDistance_->setParamater(&obs_[0], vertices_unordered_mapping_, triangle_unordered_mapping_, vertices_unordered_mapping_.size(), triangle_unordered_mapping_.size(), obs_.size() / 6);
    //     optimizerDistance_->initialize();
    //     optimizerDistance_->run();
    //     optimizerDistance_->getVertices(vertices_);
    // } else if (optimization_algorithm_ == 1) {
    //     optimizeWithout_->setParamater(&obs_[0], vertices_unordered_mapping_, triangle_unordered_mapping_, vertices_unordered_mapping_.size(), triangle_unordered_mapping_.size(), obs_.size() / 6);
    //     optimizeWithout_->initialize();
    //     optimizeWithout_->run();
    //     optimizeWithout_->getVertices(vertices_);
    // }
}
}