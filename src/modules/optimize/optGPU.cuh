#ifndef OPTIMISERGPU_H
#define OPTIMISERGPU_H

#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <unordered_map>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cusolverSp.h>
// #include <math_functions.h>
#include <cuda_runtime_api.h>

#include "suitesparse/cholmod.h"
#include "suitesparse/amd.h"


namespace stbr
{

    class database;
    class optGPU {
        public:
            __host__ optGPU(int max_iteration, std::vector<Eigen::Vector3d> &reference, std::vector<Eigen::Vector3i> &triangles, Eigen::Matrix3d K, bool verbose);
            __host__ ~optGPU();
            // __host__ void setParamater(double *observation, std::unordered_map<int,int> &unordered_mapping_vertices, std::unordered_map<int,int> &unordered_mapping_triangles, int number_vertices, int number_triangles, int number_observation);
            __host__ void setParamater(std::vector<double> observation, int number_observation);
            
            __host__ void getVertices(std::vector<Eigen::Vector3d> &vertices);

            __host__ void run();

            __host__ void setDB(database *db);

        private:
        
            std::vector<Eigen::Vector3i> triangles_;
            std::vector<Eigen::Vector3d> ver;

            database *db_ = nullptr;

            // just for the paper
            std::vector<double> total_time_;
            std::vector<double> cholmod_timer_;
            std::vector<double> preparation_;
            std::vector<double> subtriangle_;



            int max_iteration_=10;

            void checkCusolverStatus(cusolverStatus_t status);

            cusparseMatDescr_t descrA_;
            cusolverSpHandle_t cusolverH_ = nullptr;
            cusolverStatus_t status_;

            int singularity_ = 0;
            int number_vertices_;
            int number_triangles_;
            int nnz_;


            double *b_;
            double *h_vertices_;
            double *h_d;

            char *W_aux;
            // int* nnz;
            std::vector<int> colPtrCsr;
            std::vector<int> rowPtrCsr;
            std::vector<double> valPtrCsr;

            int *Ap_;
            int *Aii_;
            double *h_AWA;
            double *h_AWb;


            // device parameter
            double* d_vertices_;
            double* d_reference_;
            double* d_bar_;
            double* d_K_;
            double* d_obs_;
            double* d_cost_;
            double* d_dx_;
            double* d_d;
            int* d_max_iteration_;
            int* d_triangles_;
            int* d_number_triangles_;
            int* d_number_vertices_;
            int* d_number_observation_;
            double *d_c_unit;
            
            int* d_nnz;
            int* d_colPtrCsr;
            int* d_rowPtrCsr;
            double* d_valPtrCsr;
            double* d_b;
            double* d_x;
            

            double* d_information_matrix;
            double *d_determinant;
            double *d_g;
            double *d_H;

            double *d_AWA;
            double *d_AWb;


            int blocksize_cost;
            int blocksize_G;
            int blocksize_H;
            int blocksize_det;
            int blocksize_adj;
            int blocksize_up;

            int threadsize_cost;
            int threadsize_G;
            int threadsize_H;
            int threadsize_det;
            int threadsize_adj;
            int threadsize_up;


            cudaStream_t stream_cost;
            cudaStream_t stream_G;
            cudaStream_t stream_H;
            cudaStream_t stream_det;
            cudaStream_t stream_adj;
            cudaStream_t stream_up;
            cudaStream_t stream_AWA;
            cudaStream_t stream_AWb;

            cudaEvent_t event_cost;
            cudaEvent_t event_G;
            cudaEvent_t event_H;
            cudaEvent_t event_det;
            cudaEvent_t event_adj;
            cudaEvent_t event_up;
            cudaEvent_t event_AWA;
            cudaEvent_t event_AWb;
            // cudaStream_t cost;
            // cudaStream_t cost;
            // cudaStream_t cost;


            cholmod_common m_cS; 
            cholmod_sparse *m_cholSparseS;				
            cholmod_factor *m_cholFactorS; 
            cholmod_dense  *m_cholSparseR, *m_cholSparseE;


 

    };
    
}; // namespace name

#endif // OPTIMISERGPU_H
