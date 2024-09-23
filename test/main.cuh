#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <unordered_map>

#include <cuda.h>
#include <cuda_runtime.h>
#include <math_functions.h>
#include "suitesparse/cholmod.h"
#include "suitesparse/amd.h"
#include "suitesparse/colamd.h"


namespace test_opt {
    class optGPU {
        public:
        __host__ optGPU(int max_iteration, std::vector<Eigen::Vector3d> &vertices, std::vector<Eigen::Vector3i> &triangles, Eigen::Matrix3d K, bool verbose);

        __host__ void setParamater(double *observation, std::unordered_map<int,int> &unordered_mapping_vertices, std::unordered_map<int,int> &unordered_mapping_triangles, int number_vertices, int number_triangles, int number_observation);
        __host__ void initialize();
        __host__ void getVertices(std::vector<Eigen::Vector3d> &vertices);

        __host__ void run();
        
        private:
            // host parameter
            Eigen::Matrix3d K_, invK_;
            double *observation_ = nullptr;
            int number_vertices_ = 0;
            int number_triangles_ = 0;
            int number_observation_ = 0;

            std::unordered_map<int, int> triangle_unordered_mapping_;
            std::unordered_map<int, int> vertices_unordered_mapping_;

            std::vector<Eigen::Vector3d> e_vertices_;
            std::vector<Eigen::Vector3d> e_reference_;
            std::vector<Eigen::Vector3i> e_triangles_;

            std::vector<int> triangles_;

            double *vertices_;
            double *reference_;
            double* sub_vert_;
            double* cost_;
            double* dx_;

            // device parameter
            double *d_reference_;
            double *d_vertices_;
            int *d_triangles_;
            double* d_sub_vert_;
            double* d_cost_;
            double* d_dx_;
            

            // device Parameter

    };
};

__device__ void compute_error(Eigen::Vector3d &error, double* reference, double* vertices, double* test);
__global__ void optimise_single_triangles(int* triangles, double* reference, double* vertices, double* sub_vert, double* cost, double* dx);
__device__ void distance_only_gn(int* triangles, double* reference, double* vertices, double* sub_vert, double* cost, double* dx) ;