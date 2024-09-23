#include "optim.cuh"
#include <chrono>
#include <fstream>
#include <sstream>
#include <string>

namespace optim {
    __host__ optGPU::optGPU(int max_iteration, std::vector<Eigen::Vector3d> &reference, std::vector<Eigen::Vector3i> &triangles, Eigen::Matrix3d K, bool verbose) {
        number_vertices_ = reference.size();
        number_triangles_ = triangles.size();

            // std::cout << triangles[0] << " " << triangles[1] << " " << triangles[2] << std::endl;
            // exit(1);
        
        // W_aux is an auxallary matrix to identify the number of non zero elements and to create the compressed row sparse matrix (csr) for the lower triangle! 
        W_aux = (char*)malloc(number_vertices_ * number_vertices_ * sizeof(char));
        memset( W_aux, 0, number_vertices_ * number_vertices_ * sizeof(char));
        int nnz = 0;

        for(int i=0; i < number_triangles_; i++) {
            Eigen::Vector3i tri = triangles[i]; 
            for(int m=0; m<3;m++) {
                int v1 = tri[m];
                for (int n=m; n<3;n++) {
                    int v2 = tri[n];
                    if (v1 <= v2) {
                        if (W_aux[v2*number_vertices_+v1] == 0)
                            nnz++; 
                        W_aux[v2*number_vertices_+v1] = 1;
                    } else {

                        if (W_aux[v1*number_vertices_+v2] == 0)
                            nnz++;
                        W_aux[v1*number_vertices_+v2] = 1;
                    }           
                }
            }
        }


        int counter = 0;
        rowPtrCsr.push_back(counter);

        for (int i=0; i<number_vertices_; i++) {

            for(int j=0;j<number_vertices_;j++) {
                int idx = (int)W_aux[i*number_vertices_+j];
                if(idx == 1) {
                    colPtrCsr.push_back(j);
                    counter++;
                    valPtrCsr.push_back(0);
                }
                // std::cout << (int)W_aux[i*number_vertices_+j] << "\t";
            }
            rowPtrCsr.push_back(counter);

            // std::cout << std::endl;
        }
        nnz_ = nnz;

        // std::cout <<  nnz << std::endl;

        // for(int i=0; i<colPtrCsr.size();i++) {
        //     std::cout << colPtrCsr[i] << "\t";
        // }
        // std::cout << std::endl;

        // for(int i=0; i<rowPtrCsr.size();i++) {
        //     std::cout << rowPtrCsr[i] << "\t";
        // }
        // std::cout << std::endl;

        // for(int i=0; i<valPtrCsr.size();i++) {
        //     std::cout << valPtrCsr[i] << "\t";
        // }
        // std::cout << std::endl;
        // exit(1);
        
        // reference_ = (double*)malloc(number_vertices_*3*sizeof(double));
        // vertices_ = (double*)malloc(number_vertices_*3*sizeof(double)); // ist ok
        
        cudaError_t err1 = cudaMalloc((void**)&d_vertices_, number_vertices_*3*sizeof(double));
        cudaError_t err2 = cudaMalloc((void**)&d_reference_, number_vertices_*3*sizeof(double));
        cudaError_t err3 = cudaMalloc((void**)&d_triangles_, number_triangles_*3*sizeof(int));
        cudaError_t err4 = cudaMalloc((void**)&d_K_, 4*sizeof(double));
        cudaError_t err5 = cudaMalloc((void**)&d_max_iteration_, 1*sizeof(int));
        // cudaError_t err6 = cudaMalloc((void**)&d_break_criteria, 1*sizeof(double));
        cudaError_t err7 = cudaMalloc((void**)&d_obs_, number_vertices_*2*sizeof(double));

        // cudaError_t err4 = cudaMalloc((void**)&d_sub_vert_, number_triangles_*3*3*sizeof(double));
        cudaError_t err8 = cudaMalloc((void**)&d_cost_, number_triangles_*20*sizeof(double));
        cudaError_t err9 = cudaMalloc((void**)&d_dx_, number_triangles_*max_iteration*sizeof(double));

        cudaError_t err10 = cudaMalloc((void**)&d_number_triangles_, 1*sizeof(int));
        cudaError_t err11 = cudaMalloc((void**)&d_number_vertices_, 1*sizeof(int));
        cudaError_t err12 = cudaMalloc((void**)&d_number_observation_, 1*sizeof(int));
        cudaError_t err13 = cudaMalloc((void**)&d_bar_, number_triangles_*3*sizeof(double));


        cudaError_t err14 = cudaMalloc((void**)&d_colPtrCsr, colPtrCsr.size()*sizeof(int));
        cudaError_t err15 = cudaMalloc((void**)&d_rowPtrCsr, rowPtrCsr.size()*sizeof(int));
        cudaError_t err16 = cudaMalloc((void**)&d_valPtrCsr, valPtrCsr.size()*sizeof(double));
        cudaError_t err17 = cudaMalloc((void**)&d_b, number_vertices_*sizeof(double));
        cudaError_t err18 = cudaMalloc((void**)&d_x, number_vertices_*sizeof(double));
        cudaError_t err19 = cudaMalloc((void**)&d_information_matrix, number_triangles_*9*sizeof(double));
        cudaError_t err20 = cudaMalloc((void**)&d_nnz, 1*sizeof(int));
        cudaError_t err21 = cudaMalloc((void**)&d_d, number_triangles_*3*sizeof(double));


        if ((err1 != cudaSuccess) || (err2 != cudaSuccess) || (err3 != cudaSuccess) || (err4 != cudaSuccess)
        || (err5 != cudaSuccess) || (err7 != cudaSuccess) || (err8 != cudaSuccess) 
        || (err9 != cudaSuccess) || (err10 != cudaSuccess) || (err11 != cudaSuccess) || (err12 != cudaSuccess)
        || (err13 != cudaSuccess) || (err14 != cudaSuccess) || (err15 != cudaSuccess) || (err16 != cudaSuccess) 
        || (err17 != cudaSuccess) || (err18 != cudaSuccess) || (err19 != cudaSuccess) || (err20 != cudaSuccess)
        || (err21 != cudaSuccess)) {
            printf("CUDA Error1: %s\n", cudaGetErrorString(err1));
            printf("CUDA Error2: %s\n", cudaGetErrorString(err2));
            printf("CUDA Error3: %s\n", cudaGetErrorString(err3));
            printf("CUDA Error4: %s\n", cudaGetErrorString(err4));
            printf("CUDA Error5: %s\n", cudaGetErrorString(err5));
            printf("CUDA Error7: %s\n", cudaGetErrorString(err7));
            printf("CUDA Error8: %s\n", cudaGetErrorString(err8));
            printf("CUDA Error9: %s\n", cudaGetErrorString(err9));
            printf("CUDA Error10: %s\n", cudaGetErrorString(err10));
            printf("CUDA Error11: %s\n", cudaGetErrorString(err11));
            printf("CUDA Error12: %s\n", cudaGetErrorString(err12));
            printf("CUDA Error13: %s\n", cudaGetErrorString(err13));
            printf("CUDA Error14: %s\n", cudaGetErrorString(err14));
            printf("CUDA Error15: %s\n", cudaGetErrorString(err15));
            printf("CUDA Error16: %s\n", cudaGetErrorString(err16));
            printf("CUDA Error17: %s\n", cudaGetErrorString(err17));
            printf("CUDA Error18: %s\n", cudaGetErrorString(err18));
            printf("CUDA Error19: %s\n", cudaGetErrorString(err19));
            printf("CUDA Error20: %s\n", cudaGetErrorString(err20));
            printf("CUDA Error21: %s\n", cudaGetErrorString(err21));
            // Handle the error, e.g., by exiting the function
            exit(1);
        }
        
        

        double K_tmp[4];
        K_tmp[0] = K(0,0);
        K_tmp[1] = K(0,2);
        K_tmp[2] = K(1,1);
        K_tmp[3] = K(1,2);
        
        cudaMemcpy(d_triangles_, triangles.data(), number_triangles_*3*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_reference_, reference.data(), number_vertices_*3*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_K_, K_tmp, 4*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_max_iteration_, &max_iteration, 1*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_number_triangles_, &number_triangles_, 1*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_number_vertices_, &number_vertices_, 1*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_rowPtrCsr, rowPtrCsr.data(), rowPtrCsr.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_colPtrCsr, colPtrCsr.data(), colPtrCsr.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_valPtrCsr, valPtrCsr.data(), nnz * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_nnz, &nnz, 1 * sizeof(int), cudaMemcpyHostToDevice);

        // std::cout << rowPtrCsr.size() << std::endl; exit(1);

        // cudaMemcpy(d_number_observation_, &d_number_observation_, 1*sizeof(int), cudaMemcpyHostToDevice);
        // cudaMemcpy(d_break_criteria, break_criteria, 1*sizeof(double), cudaMemcpyHostToDevice);

        // Todo: Cuda informationen ausprinten wenn gefordert!

        // Todo: Compute \bar{d_{ij}} hier!
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));
        }


        for(int i = 0; i<triangles.size();i++) {
            Eigen::Vector3i tri = triangles[i];

            if ((tri.x() < 0 || tri.x() >= reference.size()) || (tri.y() < 0 || tri.y() >= reference.size()) || (tri.z() < 0 || tri.z() >= reference.size())) {
                printf("shit %d, \n", i);
            }
        }
        // std::cout << number_triangles_ << std::endl;
        compute_d_bar<<<number_triangles_,3>>>(d_triangles_, d_reference_, d_bar_);
        cudaDeviceSynchronize();

        

        // double depth[6];

        // cudaMemcpy(depth, d_bar_, 6*sizeof(double), cudaMemcpyDeviceToHost);

        // std::cout << depth[0] << " " << depth[1] << " " << depth[2] << std::endl;
        // std::cout << depth[3] << " " << depth[4] << " " << depth[5] << std::endl;
        // exit(1);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));
        }
        // exit(1);
        // printf("Con\n");
        // Todo: possible to release d_reference!


        status_ = cusolverSpCreate(&cusolverH_); 
        checkCusolverStatus(status_);


        cusparseCreateMatDescr(&descrA_);
        cusparseSetMatType(descrA_, CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseSetMatIndexBase(descrA_, CUSPARSE_INDEX_BASE_ZERO);
    }
    __host__ optGPU::~optGPU(){
        // cudaFree(d_vertices_);
        // cudaFree(d_reference_);
        // cudaFree(d_bar_);
        // cudaFree(d_K_);
        // cudaFree(d_obs_);
        // cudaFree(d_cost_);
        // cudaFree(d_dx_);
        // cudaFree(d_max_iteration_);
        // cudaFree(d_triangles_);
        // cudaFree(d_number_triangles_);
        // cudaFree(d_number_vertices_);
        // cudaFree(d_number_observation_);
    }


    void optGPU::checkCusolverStatus(cusolverStatus_t status) {
        switch (status) {
            case CUSOLVER_STATUS_SUCCESS:
                std::cout << "Operation completed successfully." << std::endl;
                break;
            case CUSOLVER_STATUS_NOT_INITIALIZED:
                std::cout << "CUSOLVER_STATUS_NOT_INITIALIZED: The library was not initialized." << std::endl;
                break;
            case CUSOLVER_STATUS_ALLOC_FAILED:
                std::cout << "CUSOLVER_STATUS_ALLOC_FAILED: The resources could not be allocated." << std::endl;
                break;
            case CUSOLVER_STATUS_INVALID_VALUE:
                std::cout << "CUSOLVER_STATUS_INVALID_VALUE: Invalid parameters were passed." << std::endl;
                break;
            case CUSOLVER_STATUS_ARCH_MISMATCH:
                std::cout << "CUSOLVER_STATUS_ARCH_MISMATCH: The device only supports compute capability 5.0 and above." << std::endl;
                break;
            case CUSOLVER_STATUS_INTERNAL_ERROR:
                std::cout << "CUSOLVER_STATUS_INTERNAL_ERROR: An internal operation failed." << std::endl;
                break;
            case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
                std::cout << "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED: The matrix type is not supported." << std::endl;
                break;
            default:
                std::cout << "Unknown error occurred." << std::endl;
        }
    }


    // __host__ void optGPU::setParamater(double *observation, std::unordered_map<int,int> &unordered_mapping_vertices, std::unordered_map<int,int> &unordered_mapping_triangles, int number_vertices, int number_triangles, int number_observation)

    __host__ void optGPU::setParamater(std::vector<double> observation, int number_observation) {
        
        // int *num_obs;
        // *num_obs = number_observation; 
        // std::cout << "hier\n";
        cudaMemcpy(d_obs_, observation.data(), number_vertices_*2*sizeof(double), cudaMemcpyHostToDevice);
        // cudaMemcpy(d_number_observation_, num_obs, 1*sizeof(int), cudaMemcpyHostToDevice); // Todo: at the moment not needed

        // auto start1 = std::chrono::high_resolution_clock::now();

        obs2unitvector<<<number_observation,1>>>(d_obs_, d_vertices_, d_K_);
        cudaDeviceSynchronize();
        // std::cout << number_triangles_ << " "<< number_vertices_ << std::endl; 
        // exit(1);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));
        }
        
        // auto end1 = std::chrono::high_resolution_clock::now();
        // // Berechnung der verstrichenen Zeit
        // auto elapsed1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);

        // // Ausgabe der verstrichenen Zeit in Sekunden
        // std::cout <<" Verstrichene Zeit: " << elapsed1.count() << " Mikrosekunden" << std::endl;


    }


    __host__ void optGPU::run() {

        // kernel call
        // Todo: -> use d_bar
        //<<<number_triangles_,18>>>

        int blocknumber = 1;
        int threadnumber = static_cast<int>(std::ceil(number_triangles_ * 18 ));
        
        // 1008 is the max number of threads which can be used to operate in a full block for a complete triangle!
        if (threadnumber > 1008) {
            blocknumber = static_cast<int>(std::ceil(threadnumber/1008)) ;
            threadnumber = 1008;
        }

        std::cout << threadnumber << " " << blocknumber << std::endl;

        size_t sharedMemSize = 68 * (threadnumber / 18) * sizeof(double);

        gauss_newton<<<blocknumber, threadnumber, sharedMemSize>>>(d_triangles_, d_bar_, d_reference_, d_vertices_, d_K_, d_number_triangles_,d_triangles_, d_obs_, d_triangles_, d_max_iteration_, d_cost_, d_information_matrix, d_d);
        cudaDeviceSynchronize();
        
        // std::cout << "I'm here: \n";
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));
        }
        // int num_threads_total = nnz_ + number_vertices_;
        // int blockNum;
        // int threadNum;
        // if(num_threads_total > 1024) {
        //     if (num_threads_total%2 == 1)
        //         num_threads_total++;
        //     blockNum = num_threads_total/1024+1;
        //     threadNum = num_threads_total/blockNum;
        // } else {
        //     blockNum = 1;
        //     threadNum = num_threads_total;
        // }


        // std::cout << "blocknum: " << blockNum << " threadnum: " << threadNum <<  " nnz: " << nnz_ << " number Vert: " << number_vertices_ << std::endl; 
        // prepareCuSolver<<<blockNum , threadNum>>>(d_number_triangles_, d_number_vertices_, d_triangles_, d_colPtrCsr, d_rowPtrCsr, d_valPtrCsr, d_b, d_information_matrix, d_d, d_nnz); 
        // err = cudaGetLastError();       //int* num_triangles, int* m, int* triangles, int* colPtrCsr, int* rowPtrCsr, double* valPtrCsr, double* b, double* info_matrix, double* d, int* nnz, int* number_vertices
        // if (err != cudaSuccess) {
        //     printf("CUDA Error: %s\n", cudaGetErrorString(err));
        // }

        // status_ = cusolverSpDcsrlsvchol(cusolverH_, number_vertices_, nnz_, descrA_, d_valPtrCsr, d_rowPtrCsr, d_colPtrCsr, d_b, 0, 0, d_x, &singularity_);
        // checkCusolverStatus(status_);


        // if (singularity_ > 0) {
        //     std::cout << "Warnung: Die Matrix ist singulär bei index " << singularity_ << std::endl;
        // }


    }

    __host__ void optGPU::getVertices() {
        std::cout << "\n Daten\n";
        // Kopiere Daten von der GPU zur CPU
        double* cost_ = (double*)malloc(number_triangles_*20*sizeof(double));

        cudaMemcpy(cost_, d_cost_, number_triangles_*20*sizeof(double), cudaMemcpyDeviceToHost);

        std::cout << "error norm" << std::endl;
        for(int i=0;i<20;i++) {
            std::cout <<"Iteration: " << i <<"\ttriangle1: " << cost_[i] <<"\ttriangle2: " << cost_[i+20] << std::endl; 
        }

        double d_new[4];
        cudaMemcpy(d_new, d_x, number_vertices_*sizeof(double), cudaMemcpyDeviceToHost);
        for(int i=0;i<4;i++) {
            std::cout << d_new[i] << std::endl;
        }
        
        std::cout << "\n";
        double in[9];

        cudaMemcpy(in, d_information_matrix, 2*9*sizeof(double), cudaMemcpyDeviceToHost);
        for (int i = 0; i< 9*2; i++) {

            if(i==9) {
                std::cout << std::endl;
            }
            std::cout << in[i] << std::endl;
        }
        std::cout << "\n";

        double in1[4];
        cudaMemcpy(in, d_b, 4*sizeof(double), cudaMemcpyDeviceToHost);
        for(int i=0;i<4;i++) {
            std::cout << in1[i] << std::endl;
        }
        // for(int i=0;i<number_triangles_;i++) {
        //     for(int m=0;m<3;m++) {
        //         for(int n=0;n<3;n++) {
        //             std::cout << in[i*9+m*3+n] << "\t";
        //         }
        //         std::cout << std::endl;
        //     }
        //         std::cout << std::endl;

        // }



        // double test[4];
        // cudaMemcpy(test, d_b, 4 * sizeof(double), cudaMemcpyDeviceToHost);

        // for(int i=0;i<4;i++) {
        //     std::cout << test[i] << std::endl;
        // }
        
        // double* vertices_ = (double*)malloc(number_vertices_*3*sizeof(double)); // ist ok

        // cudaMemcpy(vertices_, d_obs_,  number_vertices_*2*sizeof(double), cudaMemcpyDeviceToHost);
        // for(int i=0;i<number_vertices_;i++) {
        //     std::cout << vertices_[i*3] << " " << vertices_[i*3+1] << " " << vertices_[i*3+1] << std::endl;
        // }
    }



}

__device__ void existIDInW(int tri1, int tri2, int tri3, int val, bool &isIn, int &tri) {
    isIn = false;

    if(tri1 == val) {
        isIn = true;
        tri = 0;
    } else if(tri2 == val) {
        isIn = true;
        tri = 1;
    } else if(tri3 == val) {
        isIn = true;
        tri = 2;
    }
}

__global__ void prepareCuSolver(int* num_triangles, int* m, int* triangles, int* colPtrCsr, int* rowPtrCsr, double* valPtrCsr, double* b, double* info_matrix, double* d, int* nnz) {

    // m = num_vertices!

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < *nnz) {
        prepare_AWA(num_triangles, triangles, info_matrix, m, colPtrCsr, rowPtrCsr, valPtrCsr);
    } else if (tid < (*nnz + *m)) {
        prepare_AWb(triangles, num_triangles, info_matrix, d, b, nnz);
    }
}

__device__ void prepare_AWA(int* num_triangles, int* triangles, double* info_matrix, int* m, int* colPtrCsr, int* rowPtrCsr, double* valPtrCsr){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int colIdx = colPtrCsr[tid];
    int rowIdx = -1;
    valPtrCsr[tid] = 0;
    for(int i=0; i < *m; i++) {
        if((tid >= rowPtrCsr[i]) && (tid < rowPtrCsr[i+1])) {
            rowIdx = i;
            // colIdx = colPtrCsr[tid];
            break;
        }
    }

    for(int i=0; i < *num_triangles; i++) {
        int tri1, tri2, tri3;
        bool rowExist = false;
        bool colExist = false;

        int rIdx = -1;
        int cIdx = -1;

        tri1 = triangles[i*3];
        tri2 = triangles[i*3+1];
        tri3 = triangles[i*3+2];
        
        existIDInW(tri1, tri2, tri3, rowIdx, rowExist, rIdx);
        existIDInW(tri1, tri2, tri3, colIdx, colExist, cIdx);

        if (rowExist && colExist) {
            
            double* info_mat_tmp = info_matrix + (i*9);
            valPtrCsr[tid] += info_mat_tmp[rIdx*3+cIdx]; // row or col IDx kann größer als drei sein!
        }
    }
}


__device__ void prepare_AWb(int* triangles, int* num_triangles, double* info_matrix, double* d, double* b, int* nnz)  {
    int tid = blockDim.x * blockIdx.x + threadIdx.x - *nnz;
    b[tid] = 0;

    for(int i=0; i < *num_triangles; i++) {
        int tri1, tri2, tri3;
        double* info_mat_tmp;
        double* d_tmp;
        tri1 = triangles[i*3];
        tri2 = triangles[i*3+1];
        tri3 = triangles[i*3+2];

        if(tid==tri1) {
            info_mat_tmp = info_matrix + (i*9);
            d_tmp = d + (i*3);
            b[tid] +=  info_mat_tmp[0] * d_tmp[0];
            b[tid] +=  info_mat_tmp[3] * d_tmp[1];
            b[tid] +=  info_mat_tmp[6] * d_tmp[2];
        } else if(tid==tri2) {
            info_mat_tmp = info_matrix + (i*9);
            d_tmp = d + (i*3);
            b[tid] += info_mat_tmp[1] * d_tmp[0];
            b[tid] += info_mat_tmp[4] * d_tmp[1];
            b[tid] += info_mat_tmp[7] * d_tmp[2];
        } else if(tid==tri3) {
            info_mat_tmp = info_matrix + (i*9);
            d_tmp = d + (i*3);
            b[tid] += info_mat_tmp[2] * d_tmp[0];
            b[tid] += info_mat_tmp[5] * d_tmp[1];
            b[tid] += info_mat_tmp[8] * d_tmp[2];
        }
        
    }
}


__global__ void obs2unitvector(double* obs, double* vertices, double* K) {
    
    // K^-1 * [u;v;1]
    double uvt[3];
    uvt[0] = (obs[blockIdx.x*2] - K[1])/K[0];
    uvt[1] = (obs[blockIdx.x*2+1] - K[3])/K[2];

    double d = sqrt(uvt[0]*uvt[0]+uvt[1]*uvt[1]+1); // get the distance to compute the unit vector!

    uvt[0] /= d;
    uvt[1] /= d;
    uvt[2] = 1 / d;

    vertices[blockIdx.x*3]   = uvt[0];
    vertices[blockIdx.x*3+1] = uvt[1];
    vertices[blockIdx.x*3+2] = uvt[2];
}


/*
 * at the moment needed
 * triangles, reference, vertices, max_iterations 
 * 
 * Not needed at the moment
 * K, observation, int num_obs
 */
__global__ void gauss_newton(int* triangles, double* d_bar_, double* reference, double* vertices, double* K,
                       int* num_triangles, int* num_vertices, double* observation, int* num_obs, int* max_iteration, double* cost_output, double* info_matrix, double* final_d) {


// reinterpret_cast<double*>(sharedMem + 9 * maxTrianglePerBlock*  sizeof(double));

    // create sub triangles
    int tid = threadIdx.x % 18;
    int maxTrianglePerBlock = blockDim.x / 18; // example 17 per block 
    int triangleIdInBlock = threadIdx.x / 18; // triangle id in the actual block
    int totalTriangleId = blockIdx.x * maxTrianglePerBlock + triangleIdInBlock; // total id of the triangle

    if (totalTriangleId >= *num_triangles) {
        totalTriangleId = 0;
        tid = 255;
    }

    extern __shared__ char sharedMem[];

    double *sub_u = reinterpret_cast<double*>(sharedMem); 
    double *d_bar = sub_u + 9 * maxTrianglePerBlock; // \bar{d_{ij}}
    double *d = d_bar + 3 * maxTrianglePerBlock; // d <----- variable which will be optimised!
    double *constant_unit = d + 3 * maxTrianglePerBlock; // value of the 2 unit vector  u1^T * u2 and so on
    double *cost = constant_unit + 3 * maxTrianglePerBlock;

    double *dx = cost + 20 * maxTrianglePerBlock;
    double *g = dx + 3 * maxTrianglePerBlock;
    double *g_tmp = g + 3 * maxTrianglePerBlock;
    double *H = g_tmp + 6 * maxTrianglePerBlock;
    double *H_tmp = H + 9 * maxTrianglePerBlock;


    // allocate memory space to make the GPU programming more readable!
    // extern __shared__ char sharedMem[];
    // // double *sharedDouble2 = reinterpret_cast<double*>(sharedMem + 9 * maxTrianglePerBlock*  sizeof(double));

    // double *sub_u = reinterpret_cast<double*>(sharedMem); // is this necessary?
    // double *d_bar = reinterpret_cast<double*>(sub_u + 9 * maxTrianglePerBlock*  sizeof(double)); // \bar{d_{ij}}
    // double *d = reinterpret_cast<double*>(d_bar + 3 * maxTrianglePerBlock*  sizeof(double)); // d <----- variable which will be optimised!
    // double *constant_unit = reinterpret_cast<double*>(d + 3 * maxTrianglePerBlock*  sizeof(double)); // value of the 2 unit vector  u1^T * u2 and so on
    // double *cost = reinterpret_cast<double*>(constant_unit + 3 * maxTrianglePerBlock*  sizeof(double));

    // double *dx   = reinterpret_cast<double*>(cost + 20 * maxTrianglePerBlock*  sizeof(double));   
    // double *g     = reinterpret_cast<double*>(dx + 3 * maxTrianglePerBlock*  sizeof(double));
    // double *g_tmp = reinterpret_cast<double*>(g + 3 * maxTrianglePerBlock*  sizeof(double));
    // double *H    = reinterpret_cast<double*>(g_tmp + 6 * maxTrianglePerBlock*  sizeof(double));
    // double *H_tmp = reinterpret_cast<double*>(H + 9 * maxTrianglePerBlock*  sizeof(double));
    
    // __shared__ double sub_u[9]; // is this necessary?
    // __shared__ double d_bar[3]; // \bar{d_{ij}}
    // __shared__ double d[3]; // d <----- variable which will be optimised!
    // __shared__ double constant_unit[3]; // value of the 2 unit vector  u1^T * u2 and so on
    // __shared__ double cost[20];

    // __shared__ double dx[3];   

    // __shared__ double g[3];
    // __shared__ double g_tmp[6];

    // __shared__ double H[9];
    // __shared__ double H_tmp[9];
    // cost_output[0] = 1000;

    // vertices[0]= 20;

    int max_iter = *max_iteration;


    // todo!

    // passe den zugriff auf die triangles an beispiel mit 586 triangles! ->  für 18 threads
    // allokierung ist falsch!


    int triangle[3]; 
    triangle[0] = triangles[totalTriangleId*3];
    triangle[1] = triangles[totalTriangleId*3+1];
    triangle[2] = triangles[totalTriangleId*3+2];

    // only the first thread of each block shall do that!
    

    // sub_ver[0] = vertices[triangle1*3+0];
    // sub_ver[1] = vertices[triangle1*3+1];
    // sub_ver[2] = vertices[triangle1*3+2];

    // sub_ver[3] = verticesg_tmp[triangle3*3+0];
    // sub_ver[7] = vertices[triangle3*3+1];
    // sub_ver[8] = vertices[triangle3*3+2];

    // d_bar[0] = reference[triangle1*3];
    // d_bar[1] = reference[triangle2*3];
    // d_bar[2] = reference[triangle3*3];

    

    if(tid < 3) {
        d_bar[triangleIdInBlock * 3 + tid] = d_bar_[totalTriangleId*3+tid];
    } else if(tid < 6) {
        double ref_tmp[3];
        ref_tmp[0] = reference[triangle[tid-3]*3];
        ref_tmp[1] = reference[triangle[tid-3]*3+1];
        ref_tmp[2] = reference[triangle[tid-3]*3+2];
        d[triangleIdInBlock * 3 + tid-3] = sqrt(ref_tmp[0]*ref_tmp[0] + ref_tmp[1]*ref_tmp[1] + ref_tmp[2]*ref_tmp[2]) + 1; //sqrt() -> get distance!
    } else if(tid < 9) {
        sub_u[triangleIdInBlock * 9 + tid-6] = vertices[triangle[0]*3+tid-6];
    } else if(tid < 12) {
        sub_u[triangleIdInBlock * 9 + tid-6] = vertices[triangle[1]*3+tid-9];
    } else if(tid < 15) {
        sub_u[triangleIdInBlock * 9 + tid-6] = vertices[triangle[2]*3+tid-12];
    } 
    else {
        // d[2] = 1;
        double v1[3], v2[3];
        switch (tid)
        {
        case 15:
            v1[0] = vertices[triangle[0]*3];
            v1[1] = vertices[triangle[0]*3+1];
            v1[2] = vertices[triangle[0]*3+2];

            v2[0] = vertices[triangle[1]*3];
            v2[1] = vertices[triangle[1]*3+1];
            v2[2] = vertices[triangle[1]*3+2];

            constant_unit[triangleIdInBlock * 3 + 0] = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]; 
            break;

        case 16:
            v1[0] = vertices[triangle[0]*3];
            v1[1] = vertices[triangle[0]*3+1];
            v1[2] = vertices[triangle[0]*3+2];

            v2[0] = vertices[triangle[2]*3];
            v2[1] = vertices[triangle[2]*3+1];
            v2[2] = vertices[triangle[2]*3+2];

            constant_unit[triangleIdInBlock * 3 + 1] = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]; 
            break;

        case 17:
            v1[0] = vertices[triangle[1]*3];
            v1[1] = vertices[triangle[1]*3+1];
            v1[2] = vertices[triangle[1]*3+2];

            v2[0] = vertices[triangle[2]*3];
            v2[1] = vertices[triangle[2]*3+1];
            v2[2] = vertices[triangle[2]*3+2];

            constant_unit[triangleIdInBlock * 3 + 2] = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]; 
            break;
        }
    }


    // make sure sub triangles are in shared memory
    


    // optimisation
    for(int iter=0; iter < 1; iter++) {
        __syncthreads();

        // cost_output[blockIdx.x*20+iter] = 10;

        // compute cost
        // Todo: set a number of threads
        
        if(tid < 1) {
            cost[triangleIdInBlock*20+iter] = compute_cost(sub_u + triangleIdInBlock* 9, d + triangleIdInBlock* 3, d_bar + triangleIdInBlock * 3);
            // cost_output[blockIdx.x*20+iter] = vertices[2];
            cost_output[totalTriangleId*20+iter] = cost[triangleIdInBlock*20+iter];
        }
        // compute g // Todo:
        else if(tid < 7) {
            compute_g(sub_u + triangleIdInBlock*9, d + triangleIdInBlock*3, d_bar + triangleIdInBlock*3, constant_unit + triangleIdInBlock*3, g_tmp + triangleIdInBlock*6);
        }
        // compute H // Todo:
        else if(tid < 16) {
            compute_H(sub_u + triangleIdInBlock*9, d + triangleIdInBlock*3, d_bar + triangleIdInBlock*3, constant_unit + triangleIdInBlock*3, H_tmp + triangleIdInBlock*9);
        }

        __syncthreads();
        
        

        // // // bring g_tmp and H_tmp together!
        if (tid < 3) {
            merge_g(tid, g + triangleIdInBlock*3, g_tmp + triangleIdInBlock*6);

        } else if(tid < 9) {
            
            merge_H(tid, H + triangleIdInBlock*9, H_tmp + triangleIdInBlock*9);
        }

        __syncthreads();

        // if(tid < 9) {
        //     info_matrix[totalTriangleId*9+tid] = H[triangleIdInBlock * 9 + tid];
        // }
        
        // __syncthreads();
    
        if (tid < 6){
            compute_determinante(H + triangleIdInBlock*9, g_tmp + triangleIdInBlock*6);
        } else if (tid < 15) {
            compute_adjugate(H + triangleIdInBlock*9, H_tmp + triangleIdInBlock*9);
        }
        __syncthreads();
        // if(tid < 9) {
        //     info_matrix[totalTriangleId*9+tid] = H_tmp[triangleIdInBlock * 9 + tid];
        // }
        // //     double dd = (g_tmp[0]+g_tmp[1]+g_tmp[2]+g_tmp[3]+g_tmp[4]+g_tmp[5]);
        // //     cost_output[tid] = H_tmp[tid];
        // // } else if(tid<10) {
        // //     cost_output[tid] = (g_tmp[0]+g_tmp[1]+g_tmp[2]+g_tmp[3]+g_tmp[4]+g_tmp[5]);
        // // } 
        // // return;

        compute_update(tid, dx + triangleIdInBlock*3, d + triangleIdInBlock*3, H_tmp + triangleIdInBlock*9, g_tmp + triangleIdInBlock*6, g + triangleIdInBlock*3);

        // // switch (tid)
        // // {
        // //     case 0:
        // //         dx[0] = (H_tmp[0]*g[0] + H_tmp[3]*g[1] + H_tmp[6]*g[2]) / (g_tmp[0]+g_tmp[1]+g_tmp[2]+g_tmp[3]+g_tmp[4]+g_tmp[5]);
        // //         d[0] += dx[0];
        // //         break;
        // //     case 1:
        // //         dx[1] = (H_tmp[1]*g[0] + H_tmp[4]*g[1] + H_tmp[7]*g[2]) / (g_tmp[0]+g_tmp[1]+g_tmp[2]+g_tmp[3]+g_tmp[4]+g_tmp[5]);
        // //         d[1] += dx[1];
        // //         break;
        // //     case 2:
        // //         dx[2] = (H_tmp[2]*g[0] + H_tmp[5]*g[1] + H_tmp[8]*g[2]) / (g_tmp[0]+g_tmp[1]+g_tmp[2]+g_tmp[3]+g_tmp[4]+g_tmp[5]);
        // //         d[2] += dx[2];
        // //         break;
            
        // //     default:
        // //         break;
        // // }


    }



    // save information matrix!
    if(tid < 9) {
        info_matrix[totalTriangleId*9+tid] = H[triangleIdInBlock * 9 + tid];
    } else if(tid < 12) {
        final_d[totalTriangleId*3+tid-9] = d[triangleIdInBlock * 3 + tid-9];
    }

}

__device__ void compute_adjugate(double* H, double* a_tmp) {
    int tid = (threadIdx.x%18)-6;

    switch (tid)
    {
    case 0:
        a_tmp[0] = H[4] * H[8] - H[7] * H[5];
        break;
    case 1:
        a_tmp[1] = H[7] * H[2] - H[1] * H[8];
        break;
    case 2:
        a_tmp[2] = H[1] * H[5] - H[4] * H[2];
        break;
    case 3:
        a_tmp[3] = H[6] * H[5] - H[3] * H[8];
        break;
    case 4:
        a_tmp[4] = H[0] * H[8] - H[2] * H[6];
        break;
    case 5:
        a_tmp[5] = H[3] * H[2] - H[0] * H[5];
        break;
    case 6:
        a_tmp[6] = H[3] * H[7] - H[6] * H[4];
        break;
    case 7:
        a_tmp[7] = H[6] * H[1] - H[0] * H[7];
        break;
    case 8:
        a_tmp[8] = H[0] * H[4] - H[3] * H[1];
        break;
    
    default:
        break;
    }
}

// computes aonly a part of the determinant
__device__ void compute_determinante(double* H, double* d_tmp) {
    int tid = threadIdx.x%18;

    switch (tid)
    {
    case 0:
        d_tmp[0] = H[0] * H[4] * H[8];
        break;
    case 1:
        d_tmp[1] = H[3] * H[7] * H[2];
        break;
    case 2:
        d_tmp[2] = H[6] * H[1] * H[5];
        break;
    case 3:
        d_tmp[3] = -H[2] * H[4] * H[6];
        break;
    case 4:
        d_tmp[4] = -H[5] * H[7] * H[0];
        break;
    case 5:
        d_tmp[5] = -H[8] * H[1] * H[3];
        break;
    
    default:
        break;
    }
}

__device__ void compute_update(int tid, double *dx, double * d, double *H_tmp, double *g_tmp, double *g) {
    switch (tid)
    {
        case 0:
            dx[0] = (H_tmp[0]*g[0] + H_tmp[3]*g[1] + H_tmp[6]*g[2]) / (g_tmp[0]+g_tmp[1]+g_tmp[2]+g_tmp[3]+g_tmp[4]+g_tmp[5]);
            d[0] += dx[0];
            break;
        case 1:
            dx[1] = (H_tmp[1]*g[0] + H_tmp[4]*g[1] + H_tmp[7]*g[2]) / (g_tmp[0]+g_tmp[1]+g_tmp[2]+g_tmp[3]+g_tmp[4]+g_tmp[5]);
            d[1] += dx[1];
            break;
        case 2:
            dx[2] = (H_tmp[2]*g[0] + H_tmp[5]*g[1] + H_tmp[8]*g[2]) / (g_tmp[0]+g_tmp[1]+g_tmp[2]+g_tmp[3]+g_tmp[4]+g_tmp[5]);
            d[2] += dx[2];
            break;
        
        default:
            break;
    }
}


__device__ void compute_H(double* sub_u, double* d, double* d_bar, double* constant_unit, double* H_tmp){
    double v1v2[3];
    double tmp;
    int tid = (threadIdx.x%18) - 7;

    switch (tid)
    {
    case 0:
        v1v2[0] = sub_u[0] * d[0] - sub_u[3] * d[1]; 
        v1v2[1] = sub_u[1] * d[0] - sub_u[4] * d[1];
        v1v2[2] = sub_u[2] * d[0] - sub_u[5] * d[1];

        tmp = d[0] - constant_unit[0]*d[1];
        H_tmp[0] = tmp*tmp / (v1v2[0]*v1v2[0]+v1v2[1]*v1v2[1]+v1v2[2]*v1v2[2]);
        break;
    case 1:
        v1v2[0] = sub_u[0] * d[0] - sub_u[3] * d[1]; 
        v1v2[1] = sub_u[1] * d[0] - sub_u[4] * d[1];
        v1v2[2] = sub_u[2] * d[0] - sub_u[5] * d[1];

        tmp = -constant_unit[0] * d[0] + d[1];
        H_tmp[1] = tmp*tmp / (v1v2[0]*v1v2[0]+v1v2[1]*v1v2[1]+v1v2[2]*v1v2[2]);
        break;
    case 2:
        v1v2[0] = sub_u[0] * d[0] - sub_u[3] * d[1]; 
        v1v2[1] = sub_u[1] * d[0] - sub_u[4] * d[1];
        v1v2[2] = sub_u[2] * d[0] - sub_u[5] * d[1];

        tmp = (d[0] - constant_unit[0]*d[1]) * (-constant_unit[0] * d[0] + d[1]);
        H_tmp[2] = tmp / (v1v2[0]*v1v2[0]+v1v2[1]*v1v2[1]+v1v2[2]*v1v2[2]);
        break;
    case 3:
        v1v2[0] = sub_u[0] * d[0] - sub_u[6] * d[2]; 
        v1v2[1] = sub_u[1] * d[0] - sub_u[7] * d[2];
        v1v2[2] = sub_u[2] * d[0] - sub_u[8] * d[2];

        tmp = d[0] - constant_unit[1]*d[2];
        H_tmp[3] = tmp*tmp / (v1v2[0]*v1v2[0]+v1v2[1]*v1v2[1]+v1v2[2]*v1v2[2]);
        break;
    case 4:
        v1v2[0] = sub_u[0] * d[0] - sub_u[6] * d[2]; 
        v1v2[1] = sub_u[1] * d[0] - sub_u[7] * d[2];
        v1v2[2] = sub_u[2] * d[0] - sub_u[8] * d[2];

        tmp = -constant_unit[1] * d[0] + d[2];
        H_tmp[4] = tmp*tmp / (v1v2[0]*v1v2[0]+v1v2[1]*v1v2[1]+v1v2[2]*v1v2[2]);
        break;
    case 5:
        v1v2[0] = sub_u[0] * d[0] - sub_u[6] * d[2]; 
        v1v2[1] = sub_u[1] * d[0] - sub_u[7] * d[2];
        v1v2[2] = sub_u[2] * d[0] - sub_u[8] * d[2];

        tmp = (d[0] - constant_unit[1]*d[2]) * (-constant_unit[1] * d[0] + d[2]);
        H_tmp[5] = tmp / (v1v2[0]*v1v2[0]+v1v2[1]*v1v2[1]+v1v2[2]*v1v2[2]);
        break;
    case 6:
        v1v2[0] = sub_u[3] * d[1] - sub_u[6] * d[2]; 
        v1v2[1] = sub_u[4] * d[1] - sub_u[7] * d[2];
        v1v2[2] = sub_u[5] * d[1] - sub_u[8] * d[2];

        tmp = d[1] - constant_unit[2]*d[2];
        H_tmp[6] = tmp*tmp / (v1v2[0]*v1v2[0]+v1v2[1]*v1v2[1]+v1v2[2]*v1v2[2]);
        break;
    case 7:
        v1v2[0] = sub_u[3] * d[1] - sub_u[6] * d[2]; 
        v1v2[1] = sub_u[4] * d[1] - sub_u[7] * d[2];
        v1v2[2] = sub_u[5] * d[1] - sub_u[8] * d[2];

        tmp = -constant_unit[2] * d[1] + d[2];
        H_tmp[7] = tmp*tmp / (v1v2[0]*v1v2[0]+v1v2[1]*v1v2[1]+v1v2[2]*v1v2[2]);
        break;
    case 8:
        v1v2[0] = sub_u[3] * d[1] - sub_u[6] * d[2]; 
        v1v2[1] = sub_u[4] * d[1] - sub_u[7] * d[2];
        v1v2[2] = sub_u[5] * d[1] - sub_u[8] * d[2];

        tmp = (d[1] - constant_unit[2]*d[2]) * (-constant_unit[2] * d[1] + d[2]);
        H_tmp[8] = tmp / (v1v2[0]*v1v2[0]+v1v2[1]*v1v2[1]+v1v2[2]*v1v2[2]);
        break;
    }
}

__device__ void merge_g(int tid, double *g, double * g_tmp) {
     switch (tid)
    {
    case 0:
        g[0] = g_tmp[0] + g_tmp[2];
        break;
    case 1:
        g[1] = g_tmp[1] + g_tmp[4];
        break;
    case 2:
        g[2] = g_tmp[3] + g_tmp[5];
        break;
    default:
        break;
    }
}

__device__ void merge_H(int tid, double *H, double * H_tmp) {
    switch (tid-3)
    {
    case 0:
        H[0] = H_tmp[0]+H_tmp[3];
        break;
    
    case 1:
        H[4] = H_tmp[1]+H_tmp[6];
        break;
    
    case 2:
        H[8] = H_tmp[4]+H_tmp[7];
        break;
    
    case 3:
        H[1] = H_tmp[2];
        H[3] = H_tmp[2];
        break;
    
    case 4:
        H[2] = H_tmp[5];
        H[6] = H_tmp[5];
        break;
    
    case 5:
        H[5] = H_tmp[8];
        H[7] = H_tmp[8];
        break;
    default:
        break;
    }
}

__device__ void compute_g(double* sub_u, double* d, double* d_bar, double* constant_unit, double* g_tmp){
    double v1v2[3];
    double normv1v2;
    
    
    int tid = (threadIdx.x % 18) - 1; // thread for each computed triangle in this block!

    switch (tid)
    {
    case 0:
        v1v2[0] = sub_u[0]*d[0]-sub_u[3]*d[1];
        v1v2[1] = sub_u[1]*d[0]-sub_u[4]*d[1];
        v1v2[2] = sub_u[2]*d[0]-sub_u[5]*d[1];
        normv1v2 = sqrt(v1v2[0]*v1v2[0] + v1v2[1]*v1v2[1] + v1v2[2]*v1v2[2]);
        g_tmp[0] = (-d[0] + constant_unit[0] * d[1])*(1 - d_bar[0] / normv1v2); 
        break;
    case 1:
        v1v2[0] = sub_u[0]*d[0]-sub_u[3]*d[1];
        v1v2[1] = sub_u[1]*d[0]-sub_u[4]*d[1];
        v1v2[2] = sub_u[2]*d[0]-sub_u[5]*d[1];
        normv1v2 = sqrt(v1v2[0]*v1v2[0] + v1v2[1]*v1v2[1] + v1v2[2]*v1v2[2]);
        g_tmp[1] = (constant_unit[0] * d[0] - d[1])*(1 - d_bar[0] / normv1v2); 
        break;
    case 2:
        v1v2[0] = sub_u[0]*d[0]-sub_u[6]*d[2];
        v1v2[1] = sub_u[1]*d[0]-sub_u[7]*d[2];
        v1v2[2] = sub_u[2]*d[0]-sub_u[8]*d[2];
        normv1v2 = sqrt(v1v2[0]*v1v2[0] + v1v2[1]*v1v2[1] + v1v2[2]*v1v2[2]);
        g_tmp[2] = (-d[0] + constant_unit[1] * d[2])*(1 - d_bar[1] / normv1v2); 
        break;
    case 3:
        v1v2[0] = sub_u[0]*d[0]-sub_u[6]*d[2];
        v1v2[1] = sub_u[1]*d[0]-sub_u[7]*d[2];
        v1v2[2] = sub_u[2]*d[0]-sub_u[8]*d[2];
        normv1v2 = sqrt(v1v2[0]*v1v2[0] + v1v2[1]*v1v2[1] + v1v2[2]*v1v2[2]);
        g_tmp[3] = (constant_unit[1] * d[0] - d[2])*(1 - d_bar[1] / normv1v2); 
        break;
    case 4:
        v1v2[0] = sub_u[3]*d[1]-sub_u[6]*d[2];
        v1v2[1] = sub_u[4]*d[1]-sub_u[7]*d[2];
        v1v2[2] = sub_u[5]*d[1]-sub_u[8]*d[2];
        normv1v2 = sqrt(v1v2[0]*v1v2[0] + v1v2[1]*v1v2[1] + v1v2[2]*v1v2[2]);
        g_tmp[4] = (-d[1] + constant_unit[2] * d[2])*(1 - d_bar[2] / normv1v2); 
        break;
    case 5:
        v1v2[0] = sub_u[3]*d[1]-sub_u[6]*d[2];
        v1v2[1] = sub_u[4]*d[1]-sub_u[7]*d[2];
        v1v2[2] = sub_u[5]*d[1]-sub_u[8]*d[2];
        normv1v2 = sqrt(v1v2[0]*v1v2[0] + v1v2[1]*v1v2[1] + v1v2[2]*v1v2[2]);
        g_tmp[5] = (constant_unit[2] * d[1] - d[2])*(1 - d_bar[2] / normv1v2); 
        break;
    }
}


__device__ double compute_cost(double* sub_u, double* d, double* d_bar) {
    double error[3];
    double v1[3], v2[3], v3[3];

    v1[0]= sub_u[0]*d[0];
    v1[1]= sub_u[1]*d[0];
    v1[2]= sub_u[2]*d[0];

    v2[0]= sub_u[3]*d[1];
    v2[1]= sub_u[4]*d[1];
    v2[2]= sub_u[5]*d[1];

    v3[0]= sub_u[6]*d[2];
    v3[1]= sub_u[7]*d[2];
    v3[2]= sub_u[8]*d[2];

    error[0] = sqrt((v1[0]-v2[0])*(v1[0]-v2[0])+(v1[1]-v2[1])*(v1[1]-v2[1])+(v1[2]-v2[2])*(v1[2]-v2[2])) - d_bar[0];
    error[1] = sqrt((v1[0]-v3[0])*(v1[0]-v3[0])+(v1[1]-v3[1])*(v1[1]-v3[1])+(v1[2]-v3[2])*(v1[2]-v3[2])) - d_bar[1];
    error[2] = sqrt((v2[0]-v3[0])*(v2[0]-v3[0])+(v2[1]-v3[1])*(v2[1]-v3[1])+(v2[2]-v3[2])*(v2[2]-v3[2])) - d_bar[2];

    return sqrt(error[0]*error[0]+error[1]*error[1]+error[2]*error[2]);
}

__global__ void compute_d_bar(int* triangles, double* reference, double* d_bar) {
    int triangle[3]; 
    triangle[0] = triangles[blockIdx.x*3];
    triangle[1] = triangles[blockIdx.x*3+1];
    triangle[2] = triangles[blockIdx.x*3+2];
    double v1[3];
    double v2[3];
    double tmp[3];

    // Error in here!
    // Todo: use fixed size arrays! so klappt das nicht! so wirft das nur ein cuda error und dann habe ich das problem!

    int tid = threadIdx.x;
    switch (tid)
    {
    case 0: // norm(v1-v2)
        v1[0] = reference[triangle[0]*3];
        v1[1] = reference[triangle[0]*3+1];
        v1[2] = reference[triangle[0]*3+2];

        v2[0] = reference[triangle[1]*3];
        v2[1] = reference[triangle[1]*3+1];
        v2[2] = reference[triangle[1]*3+2];
        
        tmp[0] = v1[0] - v2[0];
        tmp[1] = v1[1] - v2[1];
        tmp[2] = v1[2] - v2[2];
        d_bar[blockIdx.x*3] = sqrt(tmp[0]*tmp[0] + tmp[1]*tmp[1] + tmp[2]*tmp[2]); 
        break;
    case 1: // norm(v1-v3)
        v1[0] = reference[triangle[0]*3];
        v1[1] = reference[triangle[0]*3+1];
        v1[2] = reference[triangle[0]*3+2];

        v2[0] = reference[triangle[2]*3];
        v2[1] = reference[triangle[2]*3+1];
        v2[2] = reference[triangle[2]*3+2];
        
        tmp[0] = v1[0] - v2[0];
        tmp[1] = v1[1] - v2[1];
        tmp[2] = v1[2] - v2[2];
        d_bar[blockIdx.x*3+1] = sqrt(tmp[0]*tmp[0]+tmp[1]*tmp[1]+tmp[2]*tmp[2]);
        break;

    case 2: // norm(v2-v3)
        v1[0] = reference[triangle[1]*3];
        v1[1] = reference[triangle[1]*3+1];
        v1[2] = reference[triangle[1]*3+2];

        v2[0] = reference[triangle[2]*3];
        v2[1] = reference[triangle[2]*3+1];
        v2[2] = reference[triangle[2]*3+2];
        
        tmp[0] = v1[0] - v2[0];
        tmp[1] = v1[1] - v2[1];
        tmp[2] = v1[2] - v2[2];
        d_bar[blockIdx.x*3+2] = sqrt(tmp[0]*tmp[0]+tmp[1]*tmp[1]+tmp[2]*tmp[2]);
        break;
    }

}

// __global__ void tester(double* in1, double* in2,double* out) {
//     out[threadIdx.x] = in1[threadIdx.x] + in2[threadIdx.y];
// }

void checkCusolverStatus(cusolverStatus_t status) {
    switch (status) {
        case CUSOLVER_STATUS_SUCCESS:
            std::cout << "Operation completed successfully." << std::endl;
            break;
        case CUSOLVER_STATUS_NOT_INITIALIZED:
            std::cout << "CUSOLVER_STATUS_NOT_INITIALIZED: The library was not initialized." << std::endl;
            break;
        case CUSOLVER_STATUS_ALLOC_FAILED:
            std::cout << "CUSOLVER_STATUS_ALLOC_FAILED: The resources could not be allocated." << std::endl;
            break;
        case CUSOLVER_STATUS_INVALID_VALUE:
            std::cout << "CUSOLVER_STATUS_INVALID_VALUE: Invalid parameters were passed." << std::endl;
            break;
        case CUSOLVER_STATUS_ARCH_MISMATCH:
            std::cout << "CUSOLVER_STATUS_ARCH_MISMATCH: The device only supports compute capability 5.0 and above." << std::endl;
            break;
        case CUSOLVER_STATUS_INTERNAL_ERROR:
            std::cout << "CUSOLVER_STATUS_INTERNAL_ERROR: An internal operation failed." << std::endl;
            break;
        case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            std::cout << "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED: The matrix type is not supported." << std::endl;
            break;
        default:
            std::cout << "Unknown error occurred." << std::endl;
    }
}


int main() {


    cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);

printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
printf("Max Threads per Multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
printf("Number of Multiprocessors: %d\n", prop.multiProcessorCount);
printf("Max blocks per Multiprozessor: %d\n", prop.maxBlocksPerMultiProcessor); 
printf("Warp size: %d\n", prop.warpSize);
// exit(1);

    std::cout << "optim start" << std::endl;
    
    std::vector<Eigen::Vector3i> triangles;
    std::vector<Eigen::Vector3d> vertices;
    

    // //  Datei öffnen
    std::ifstream file("/home/anonym/Schreibtisch/PhD/code/GPU_Distance_only/data/phi_SfT/real/S1/templates/template_mesh_final.obj");
    
    if (!file.is_open()) {
        std::cerr << "Konnte die Datei nicht öffnen!" << std::endl;
        return 1;
    }

    std::string line;
    while (std::getline(file, line)) {
        // Stream zur Zeilenverarbeitung
        std::istringstream iss(line);
        
        std::string prefix;
        double x, y, z;  // Verwendung von double für höhere Genauigkeit
        
        // Erstes Element (z.B. "v") einlesen
        iss >> prefix;
        
        // Die nächsten drei Elemente (Koordinaten) einlesen
        if (prefix == "v") {
            iss >> x >> y >> z;
            Eigen::Vector3d v_tmp;
            v_tmp << x,y,z;
            vertices.push_back(v_tmp);
        }
        if (prefix == "f"){
            std::string block;
            int f1, f2, f3;
            int counter = 0;
            while (iss >> block) {  // Blöcke einzeln lesen
                std::istringstream blockStream(block);
                std::string firstNumber;
                
                // Erste Zahl extrahieren (vor dem '/')
                std::getline(blockStream, firstNumber, '/');

                // std::cout << std::stoi(firstNumber) << std::endl;
                
                switch (counter)
                {
                case 0:
                    f1 = std::stoi(firstNumber);
                    break;
                case 1:
                    f2 = std::stoi(firstNumber);
                    break;
                case 2:
                    f3 = std::stoi(firstNumber);
                    break;
                
                default:
                    break;
                }
                counter++;
                
                
            }
            // exit(1);
            // iss >> x >> y >> z;
            Eigen::Vector3i t_tmp;
            t_tmp << f1-1,f2-1,f3-1;
            // std::cout << t_tmp << std::endl;
            triangles.push_back(t_tmp);
        }
    }


    file.close();
// exit(1);

    // /home/anonym/Schreibtisch/PhD/code/GPU_Distance_only/data/phi_SfT/real/S1/templates/template_mesh_final.obj

    // Eigen::Vector3i tmp; 
    // tmp << 0,1,2;
    // triangles.push_back(tmp);
    // tmp << 1,2,3;
    // triangles.push_back(tmp);

    // Eigen::Vector3d tmp1;
    // tmp1 << 1,1,6;
    // vertices.push_back(tmp1);
    // tmp1 << 1,2,6;
    // vertices.push_back(tmp1);
    // tmp1 << 2,2,6;
    // vertices.push_back(tmp1);
    // tmp1 << 3,2,6;
    // vertices.push_back(tmp1);

    std::cout <<  "number vertices " << vertices.size() << "\tnumber triangles: " << triangles.size() << std::endl;

    std::cout << "Set Camera calibration matrix\n";
    // Eigen::Matrix3d K;
    // K << 2,0,400,
    //      0,2,400,
    //      0,0,1;

    Eigen::Matrix3d K;
    K <<  971.522,0,962.134,
         0,944.575,554.778,
         0,0,1;

    optim::optGPU* opt = new optim::optGPU(10, vertices, triangles, K, true);


    std::vector<double> obs;
    int num_vertices = vertices.size();
    for(int i=0;i<num_vertices;i++) {

        Eigen::Vector3d vertex = vertices[i];

        Eigen::Vector3d tmp;
        tmp = K*vertex;

        tmp /= tmp.z();



        // obs.push_back(0);
        obs.push_back(tmp.x());
        obs.push_back(tmp.y());
        // obs.push_back(0);
        // obs.push_back(0);
        // obs.push_back(0);
    }
    
    opt->setParamater(obs, vertices.size());

    auto start = std::chrono::high_resolution_clock::now();
    
    opt->run();
    
    // cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    // opt->getVertices();
    // Berechnung der verstrichenen Zeit
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Ausgabe der verstrichenen Zeit in Sekunden
    std::cout <<" Verstrichene Zeit: " << elapsed.count() << " Mikrosekunden" << std::endl;


    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }



    // Test for cusolver

    // 4x4 Matrix

//     std::cout << "Test CuSolver!" << std::endl;

//     const int N=3;
//     const int nnz =3;
//     int h_csrRowPtr[N+1] = {0, 1, 2, 3}; // Zeilenstartindizes
//     int h_csrColInd[nnz] = {0, 1, 2};    // Spaltenindizes
//     double h_csrVal[nnz] = {4.0, 5.0, 6.0}; // Werte

//     double h_b[N] = {9.0, 10.0, 12.0}; // Rechte Seite b
//     double h_x[N]; // Ergebnis x
    
//      // Device-Seite Arrays
//     int *d_csrRowPtr, *d_csrColInd;
//     double *d_csrVal, *d_b, *d_x;

//     cudaMalloc((void**)&d_csrRowPtr, (N+1) * sizeof(int));
//     cudaMalloc((void**)&d_csrColInd, nnz * sizeof(int));
//     cudaMalloc((void**)&d_csrVal, nnz * sizeof(double));
//     cudaMalloc((void**)&d_b, N * sizeof(double));
//     cudaMalloc((void**)&d_x, N * sizeof(double));

//     cudaMemcpy(d_csrRowPtr, h_csrRowPtr, (N+1) * sizeof(int), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_csrColInd, h_csrColInd, nnz * sizeof(int), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_csrVal, h_csrVal, nnz * sizeof(double), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_b, h_b, N * sizeof(double), cudaMemcpyHostToDevice);

//     // cusolver handle erstellen
//     cusolverSpHandle_t cusolverH = nullptr;
//     cusolverStatus_t status = cusolverSpCreate(&cusolverH);   
//     checkCusolverStatus(status);


//     int singularity = 0;

//     // CSR Cholesky info erstellen
//     // csrcholInfo_t cholInfo = nullptr;
//     // cusolverSpCreateCsrcholInfo(&cholInfo);

//     // cusparse Matrix Deskriptor erstellen
//     cusparseMatDescr_t descrA = nullptr;
//     cusparseCreateMatDescr(&descrA);
    
//     cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
//     cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

// double xx[3] = {0,0,0};
// double *xxx;
// xxx = &xx[0];
//     // Lösen der Gleichung Ax = b mit Cholesky-Zerlegung
//     status = cusolverSpDcsrlsvchol(cusolverH, N, nnz, descrA, d_csrVal, d_csrRowPtr, d_csrColInd, d_b, 0, 0, d_x, &singularity);
//     checkCusolverStatus(status);


//     if (singularity > 0) {
//         std::cout << "Warnung: Die Matrix ist singulär bei index " << singularity << std::endl;
//     }

//     // Ergebnis zurück zum Host kopieren
//     cudaMemcpy(h_x, d_x, N * sizeof(double), cudaMemcpyDeviceToHost);

//     // Ergebnis ausgeben
//     std::cout << "Ergebnis: ";
//     for (int i = 0; i < N; i++) {
//         std::cout << h_x[i] << " ";
//     }
//     std::cout << std::endl;

//     // Ressourcen freigeben
//     cudaFree(d_csrRowPtr);
//     cudaFree(d_csrColInd);
//     cudaFree(d_csrVal);
//     cudaFree(d_b);
//     cudaFree(d_x);

//     cusolverSpDestroy(cusolverH);

//     std::cout << "singularity: " << singularity << std::endl;
//     std::cout << "Did CuSolver" << std::endl;
     

}