
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
            int max_iteration_=1;

            void checkCusolverStatus(cusolverStatus_t status);

            cusparseMatDescr_t descrA_;
            cusolverSpHandle_t cusolverH_ = nullptr;
            cusolverStatus_t status_;

            int singularity_ = 0;
            int number_vertices_;
            int number_triangles_;
            int nnz_;


            double *b_;

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

// // preprocess the observation!
// __global__ void obs2unitvector(double* obs, double* vertices, double* K);

// Gauß Newton optimisation
__global__ void gauss_newton(int* triangles, double* d_bar_, double* reference, double* vertices, double* K,
int* num_triangles, int* num_vertices, double* observation, int* num_obs, int* max_iteration, double* cost, double* info_matrix, double* final_d);

// preprocess Cusolver
__global__ void prepareCuSolver(int*num_triangles, int* m, int* triangles, int* colPtrCsr, int* rowPtrCsr, double* valPtrCsr, double* b, double* info_matrix, double* d, int* nnz);


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



__global__ void compute_d_bar_x(int* numTriangles, int *triangles, double *reference, double *d_bar);
__global__ void compute_d_bar_y(int* numTriangles, int *triangles, double *reference, double *d_bar);
__global__ void compute_d_bar_z(int* numTriangles, int *triangles, double *reference, double *d_bar);

// preprocess the observation!
__global__ void obs2unitvector(double* obs, double* vertices, double* K, int *number_observation);

__global__ void computeH_new(double *H, int *triangles, double* vertices, double *d, double *constant_unit, int *num_triangles);

__global__ void computeG_new(double *g, int *triangles, double *vertices, double *d, double *d_bar, double *constant_unit, int *num_triangles);

__global__ void compute_constantUnits(double *c_unit, int *triangles, int *num_triangles, double *vertices);

__global__ void compute_determinante_new(double *H, int *num_triangles, double *determinante);

__global__ void compute_adjugate_new(double *H, int *num_triangles, double *adjugate_matrix); 

__global__ void update_dx(double *d, int *num_triangles, double *adjugate_matrix, double *determinante, double *g, double *cost);

__global__ void compute_d(double *d, double *ref, int *num_tria, int *triangles);

__global__ void compute_cost_new(double *d_bar, int *triangle, int *num_triangle, double *d, double *vertices, double *cost);


__global__ void compute_AWA(double *info_matrix, int *triangle, int *num_triangle, int *numVertices, double *AWA);

__global__ void compute_AWB(double *info_matrix, double *d, int *triangle, int *num_triangle, int *numVertices, double *AWB);
__device__ double atomicAdd_double(double* address, double val);



////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
