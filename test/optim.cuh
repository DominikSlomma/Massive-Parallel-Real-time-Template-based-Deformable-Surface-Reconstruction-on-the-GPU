
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



namespace optim
{
    class optGPU {
        public:
            __host__ optGPU(int max_iteration, std::vector<Eigen::Vector3d> &reference, std::vector<Eigen::Vector3i> &triangles, Eigen::Matrix3d K, bool verbose);
            __host__ ~optGPU();
            // __host__ void setParamater(double *observation, std::unordered_map<int,int> &unordered_mapping_vertices, std::unordered_map<int,int> &unordered_mapping_triangles, int number_vertices, int number_triangles, int number_observation);
            __host__ void setParamater(std::vector<double> observation, int number_observation);
            
            __host__ void getVertices();

            __host__ void run();

        private:

            void checkCusolverStatus(cusolverStatus_t status);

            cusparseMatDescr_t descrA_;
            cusolverSpHandle_t cusolverH_ = nullptr;
            cusolverStatus_t status_;

            int singularity_ = 0;
            int number_vertices_;
            int number_triangles_;
            int nnz_;

            char *W_aux;
            // int* nnz;
            std::vector<int> colPtrCsr;
            std::vector<int> rowPtrCsr;
            std::vector<double> valPtrCsr;


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

            
            int* d_nnz;
            int* d_colPtrCsr;
            int* d_rowPtrCsr;
            double* d_valPtrCsr;
            double* d_b;
            double* d_x;
            

            double* d_information_matrix;


 

    };
    
}; // namespace name


__device__ void merge_g(int tid, double *g, double * g_tmp);
__device__ void merge_H(int tid, double *H, double * H_tmp);
__device__ void compute_update(int tid, double *dx, double *d, double *H_tmp, double *g_tmp, double *g);


__device__ void compute_g(double* sub_u, double* d, double* d_bar, double* constant_unit, double* g_tmp);
__device__ void compute_H(double* sub_u, double* d, double* d_bar, double* constant_unit, double* H_tmp);

__device__ void compute_determinante(double* H, double* d_tmp);
__device__ void compute_adjugate(double* H, double* a_tmp);

__device__ double compute_cost(double* sub_u, double* d, double* d_bar);

__device__ void existIDInW(int tri1, int tri2, int tri3, int val, bool &isIn, int &tri);

// #ifdef __cplusplus
// extern "C" {
// #endif
__device__ void prepare_AWA(int* num_triangles, int* triangles, double* info_matrix, int* m, int* colPtrCsr, int* rowPtrCsr, double* valPtrCsr);
__device__ void prepare_AWb(int* triangles, int* num_triangles, double* info_matrix, double* d, double* b, int* nnz);

// #ifdef __cplusplus
// }
// #endif

// Kernel to compute \bar{d_{ij}}
__global__ void compute_d_bar(int* triangle, double* reference, double* d_bar);

// preprocess the observation!
__global__ void obs2unitvector(double* obs, double* vertices, double* K);

// Gauß Newton optimisation
__global__ void gauss_newton(int* triangles, double* d_bar_, double* reference, double* vertices, double* K,
int* num_triangles, int* num_vertices, double* observation, int* num_obs, int* max_iteration, double* cost, double* info_matrix, double* final_d);

// preprocess Cusolver
__global__ void prepareCuSolver(int*num_triangles, int* m, int* triangles, int* colPtrCsr, int* rowPtrCsr, double* valPtrCsr, double* b, double* info_matrix, double* d, int* nnz);
