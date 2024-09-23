#include "optGPU.cuh"
#include "gauss_newton.cuh"
#include <chrono>
#include <thread>
#include "database.h"
// #include <fstream>
// #include <sstream>
// #include <string>
#include <iostream>
#include <fstream>

namespace stbr {
    __host__ optGPU::optGPU(int max_iteration, std::vector<Eigen::Vector3d> &reference, std::vector<Eigen::Vector3i> &triangles, Eigen::Matrix3d K, bool verbose) {
        
        number_vertices_ = reference.size();
        number_triangles_ = triangles.size();
        triangles_ = triangles;
        for (int i=0; i < number_triangles_*3; i++) {
            ver.push_back(reference[1]);
        }

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
                    if (v1 >= v2) {
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

        nnz_ = nnz;


        // for (int i=0; i< 4; i++) {
        //     for(int j=0; j< 4; j++) {

        //         std::cout << (int)W_aux[i*4 +j] << "\t";

        //     }
        //     std::cout << "\n";

        // }

        

        cholmod_start(&m_cS);


        m_cholSparseE = cholmod_zeros( number_vertices_, 1, CHOLMOD_REAL, &m_cS);
        m_cholSparseS = cholmod_allocate_sparse(number_vertices_,number_vertices_,nnz_,true,true,1,CHOLMOD_REAL,&m_cS); 

        Ap_  = (int*)malloc((number_vertices_ + 1)*sizeof(int));
	    Aii_ = (int*)malloc(nnz*sizeof(int));
        h_AWA = (double*)malloc(number_vertices_ * number_vertices_ *sizeof(double));
        h_AWb = (double*)malloc(number_vertices_ *sizeof(double));
        

        int *Si = (int *)m_cholSparseS->i;
        int *Sp = (int *)m_cholSparseS->p;



        int* Cp = Ap_;
        int* Ci = Aii_;
        int ii, jj;
        int m = number_vertices_, nZ = 0;
        for ( ii = 0; ii < m; ii++ ) 
        {
            *Cp = nZ;
            *Sp = nZ;
            for( jj=0; jj<=ii; jj++ )
            {
                if ((int)W_aux[jj*m+ii]==1)
                {
                    *Ci++ = jj;
                    *Si++ = jj;
                    nZ++;

                } 
            }

            Cp++;
            Sp++;
        }
        *Cp=nZ;
        *Sp=nZ;



        Eigen::Matrix<int, Eigen::Dynamic, 1> scalarPermutation, blockPermutation;
            
            if (blockPermutation.size() == 0)
				blockPermutation.resize(number_vertices_);
            


                cholmod_sparse auxCholmodSparse;
                auxCholmodSparse.nzmax = nnz_; // Maximale Anzahl der Nicht-Null-Elemente
                auxCholmodSparse.nrow = auxCholmodSparse.ncol = number_vertices_;
                auxCholmodSparse.p = Ap_;   // Spaltenzeiger
                auxCholmodSparse.i = Aii_;   // Zeilenindizes
                auxCholmodSparse.nz = 0;      // Setze auf 0 für CHOLMOD_PATTERN
                auxCholmodSparse.x = nullptr; // Kein Wert-Array benötigt
                auxCholmodSparse.z = nullptr;
                auxCholmodSparse.stype = 1;   // Matrix ist symmetrisch
                auxCholmodSparse.xtype = CHOLMOD_PATTERN;
                auxCholmodSparse.itype = CHOLMOD_INT;
                auxCholmodSparse.dtype = CHOLMOD_DOUBLE;
                auxCholmodSparse.sorted = 1;
                auxCholmodSparse.packed = 1;

                // cholmod_print_sparse(&auxCholmodSparse, "aux", &m_cS);

            int amdStatus = cholmod_amd(&auxCholmodSparse, nullptr, 0, blockPermutation.data(), &m_cS);
			if (! amdStatus) {
				std::cout << "AMD error:\n";
                exit(1);
			}

            if (scalarPermutation.size() == 0)
				scalarPermutation.resize(m_cholSparseS->ncol);
			size_t scalarIdx = 0;
			int a = 0;
            
			for ( int i = 0; i < number_vertices_; ++i)
			{
				const int &pp = blockPermutation(i);
				int base = pp*1;
                // std::cout << pp << std::endl;
				// int nCols= (pp==0) ? 6 : 6;

                // int base =  pp*6-1;

				int nCols= 1;

				for ( int j = 0; j < nCols; ++j)
					scalarPermutation(scalarIdx++) = base++;

			}
    
            assert(scalarIdx == m_cholSparseS->ncol);

			// apply the ordering
			m_cS.nmethods = 1 ;
			m_cS.method[0].ordering = CHOLMOD_GIVEN;
            // std::cout << scalarPermutation << std::endl;

            // cholmod_print_sparse(&auxCholmodSparse, "aux", &m_cS);
            // cholmod_print_sparse(&auxCholmodSparse, "aux", &m_cS);
            // cholmod_print_dense(m_cholSparseE, "back", &m_cS);
			m_cholFactorS = cholmod_analyze_p(m_cholSparseS, scalarPermutation.data(), NULL, 0, &m_cS);

        // Si[0] = 0;
        // Si[1] = 0;
        // Si[2] = 1;
        // Si[3] = 0;
        // Si[4] = 1;
        // Si[5] = 2;
        // Si[6] = 1;
        // Si[7] = 2;
        // Si[8] = 3;


        // Sp[0] = 0;
        // Sp[1] = 1;
        // Sp[2] = 3;
        // Sp[3] = 6;
        // Sp[4] = 9;

        // exit(1);


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
        // nnz_ = nnz;

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
        h_vertices_ = (double*)malloc(number_vertices_*3*sizeof(double));
        h_d = (double*)malloc(number_triangles_*3*sizeof(double));
        //  h_vertices = (double*)malloc(number_vertices_*3*sizeof(double));
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


        // cudaError_t err14 = cudaMalloc((void**)&d_colPtrCsr, colPtrCsr.size()*sizeof(int));
        // cudaError_t err15 = cudaMalloc((void**)&d_rowPtrCsr, rowPtrCsr.size()*sizeof(int));
        // cudaError_t err16 = cudaMalloc((void**)&d_valPtrCsr, valPtrCsr.size()*sizeof(double));
        // cudaError_t err17 = cudaMalloc((void**)&d_b, number_vertices_*sizeof(double));
        // cudaError_t err18 = cudaMalloc((void**)&d_x, number_vertices_*sizeof(double));
        cudaError_t err19 = cudaMalloc((void**)&d_information_matrix, number_triangles_*9*sizeof(double));
        // cudaError_t err20 = cudaMalloc((void**)&d_nnz, 1*sizeof(int));
        cudaError_t err21 = cudaMalloc((void**)&d_d, number_triangles_*3*sizeof(double));

        cudaError_t err22 = cudaMalloc((void**)&d_c_unit, number_triangles_*3*sizeof(double));
        cudaError_t err23 = cudaMalloc((void**)&d_determinant, number_triangles_*6*sizeof(double));
        cudaError_t err24 = cudaMalloc((void**)&d_g, number_triangles_*3*sizeof(double));
        cudaError_t err25 = cudaMalloc((void**)&d_H, number_triangles_*9*sizeof(double));
        cudaError_t err26 = cudaMalloc((void**)&d_AWA, number_vertices_*number_vertices_*sizeof(double));
        cudaError_t err27 = cudaMalloc((void**)&d_AWb, number_vertices_*sizeof(double));



      


        if ((err1 != cudaSuccess) || (err2 != cudaSuccess) || (err3 != cudaSuccess) || (err4 != cudaSuccess)
        || (err5 != cudaSuccess) || (err7 != cudaSuccess) || (err8 != cudaSuccess) 
        || (err9 != cudaSuccess) || (err10 != cudaSuccess) || (err11 != cudaSuccess) || (err12 != cudaSuccess)
        || (err13 != cudaSuccess) || (err19 != cudaSuccess) 
        || (err21 != cudaSuccess) || (err22 != cudaSuccess) || (err23 != cudaSuccess) || (err24 != cudaSuccess)
        || (err25 != cudaSuccess)) {
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
            // printf("CUDA Error14: %s\n", cudaGetErrorString(err14));
            // printf("CUDA Error15: %s\n", cudaGetErrorString(err15));
            // printf("CUDA Error16: %s\n", cudaGetErrorString(err16));
            // printf("CUDA Error17: %s\n", cudaGetErrorString(err17));
            // printf("CUDA Error18: %s\n", cudaGetErrorString(err18));
            printf("CUDA Error19: %s\n", cudaGetErrorString(err19));
            // printf("CUDA Error20: %s\n", cudaGetErrorString(err20));
            printf("CUDA Error21: %s\n", cudaGetErrorString(err21));
            printf("CUDA Error22: %s\n", cudaGetErrorString(err22));
            printf("CUDA Error23: %s\n", cudaGetErrorString(err23));
            printf("CUDA Error24: %s\n", cudaGetErrorString(err24));
            printf("CUDA Error25: %s\n", cudaGetErrorString(err25));
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
        cudaMemset(d_AWA, 0, number_vertices_ * number_vertices_ * sizeof(double));
        cudaMemset(d_AWb, 0, number_vertices_  * sizeof(double));
        // cudaMemcpy(d_rowPtrCsr, rowPtrCsr.data(), rowPtrCsr.size() * sizeof(int), cudaMemcpyHostToDevice);
        // cudaMemcpy(d_colPtrCsr, colPtrCsr.data(), colPtrCsr.size() * sizeof(int), cudaMemcpyHostToDevice);
        // cudaMemcpy(d_valPtrCsr, valPtrCsr.data(), nnz * sizeof(double), cudaMemcpyHostToDevice);
        // cudaMemcpy(d_nnz, &nnz, 1 * sizeof(int), cudaMemcpyHostToDevice);

        // std::cout << rowPtrCsr.size() << std::endl; exit(1);

        // cudaMemcpy(d_number_observation_, &d_number_observation_, 1*sizeof(int), cudaMemcpyHostToDevice);
        // cudaMemcpy(d_break_criteria, break_criteria, 1*sizeof(double), cudaMemcpyHostToDevice);

        // Todo: Cuda informationen ausprinten wenn gefordert!

        // Todo: Compute \bar{d_{ij}} hier!
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));
        }


        // for(int i = 0; i<triangles.size();i++) {
        //     Eigen::Vector3i tri = triangles[i];

        //     if ((tri.x() < 0 || tri.x() >= reference.size()) || (tri.y() < 0 || tri.y() >= reference.size()) || (tri.z() < 0 || tri.z() >= reference.size())) {
        //         printf("shit %d, \n", i);
        //     }
        // }
        // std::cout << number_triangles_ << std::endl;


        cudaStream_t stream1, stream2, stream3;
        cudaStreamCreate(&stream1);
        cudaStreamCreate(&stream2);
        cudaStreamCreate(&stream3);

        int numBlocks = 1;
        int numThreads = number_triangles_;
        if(numThreads > 1024) {
            int newBlocksize = std::ceil(static_cast<double>(numThreads) / 1024);
            numThreads = std::ceil(static_cast<double>(numThreads) / newBlocksize);
            numBlocks = newBlocksize;
        }



        compute_d_bar_x<<<numBlocks, numThreads, 0, stream1>>>(d_number_triangles_, d_triangles_, d_reference_, d_bar_);
        compute_d_bar_y<<<numBlocks, numThreads, 0, stream2>>>(d_number_triangles_, d_triangles_, d_reference_, d_bar_);
        compute_d_bar_z<<<numBlocks, numThreads, 0, stream3>>>(d_number_triangles_, d_triangles_, d_reference_, d_bar_);

        cudaDeviceSynchronize();



        // cudaStreamSynchronize(stream1);
        // cudaStreamSynchronize(stream2);
        // cudaStreamSynchronize(stream3);



        // compute_d_bar<<<number_triangles_,3>>>(d_triangles_, d_reference_, d_bar_);
        // cudaDeviceSynchronize();

        cudaStreamCreate(&stream_cost);
        cudaStreamCreate(&stream_G);
        cudaStreamCreate(&stream_H);
        cudaStreamCreate(&stream_det);
        cudaStreamCreate(&stream_adj);
        cudaStreamCreate(&stream_up);
        cudaStreamCreate(&stream_AWA);
        cudaStreamCreate(&stream_AWb);

        cudaEventCreate(&event_cost);
        cudaEventCreate(&event_G);
        cudaEventCreate(&event_H);
        cudaEventCreate(&event_det);
        cudaEventCreate(&event_adj);
        cudaEventCreate(&event_up);
        cudaEventCreate(&event_AWA);
        cudaEventCreate(&event_AWb);


        int blocksize;
        int threadsize;

        blocksize = 1;
        threadsize = number_triangles_ * 1;

        if(threadsize > 1024) {
            int newBlocksize = std::ceil(static_cast<float>(threadsize)/1024);
            blocksize = newBlocksize;
            threadsize = std::ceil(static_cast<float>(threadsize) / newBlocksize);
        }

        blocksize_cost = blocksize;
        threadsize_cost = threadsize;


        blocksize = 1;
        threadsize = number_triangles_ * 3;

        if(threadsize > 1024) {
            int newBlocksize = std::ceil(static_cast<float>(threadsize)/1024);
            blocksize = newBlocksize;
            threadsize = std::ceil(static_cast<float>(threadsize) / newBlocksize);

            threadsize += (3 - (threadsize % 3));

            if(threadsize > 1022) {
                threadsize = 1022;
                blocksize += 1;
            }
        }

        blocksize_G = blocksize;
        threadsize_G = threadsize;

        blocksize_H = blocksize;
        threadsize_H = threadsize;

        blocksize = 1;
        threadsize = number_triangles_ * 6;

        if(threadsize > 1024) {
            int newBlocksize = std::ceil(static_cast<float>(threadsize)/1024);
            blocksize = newBlocksize;
            threadsize = std::ceil(static_cast<float>(threadsize) / newBlocksize);

            threadsize += (6 - (threadsize % 6));

            if(threadsize > 1020) {
                threadsize = 1020;
                blocksize += 1;
            }
        }

        blocksize_det = blocksize;
        threadsize_det = threadsize;

        blocksize = 1;
        threadsize = number_triangles_ * 9;
        if(threadsize > 1024) {
            int newBlocksize = std::ceil(static_cast<float>(threadsize)/1024);
            blocksize = newBlocksize;
            threadsize = std::ceil(static_cast<float>(threadsize) / newBlocksize);

            threadsize += (9 - (threadsize % 9));

            if(threadsize > 1017) {
                threadsize = 1017;
                blocksize += 1;
            }
        }


        blocksize_adj = blocksize;
        threadsize_adj = threadsize;


        blocksize = 1;
        threadsize = number_triangles_ * 3;
        if(threadsize > 1024) {
            int newBlocksize = std::ceil(static_cast<float>(threadsize)/1024);
            blocksize = newBlocksize;
            threadsize = std::ceil(static_cast<float>(threadsize) / newBlocksize);

            threadsize += (3 - (threadsize % 3));

            if(threadsize > 1022) {
                threadsize = 1022;
                blocksize += 1;
            }
        }

        blocksize_up = blocksize;
        threadsize_up = threadsize;

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


        // status_ = cusolverSpCreate(&cusolverH_); 
        // checkCusolverStatus(status_);


        // cusparseCreateMatDescr(&descrA_);
        // cusparseSetMatType(descrA_, CUSPARSE_MATRIX_TYPE_GENERAL);
        // cusparseSetMatIndexBase(descrA_, CUSPARSE_INDEX_BASE_ZERO);
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



       
        // cudaMemset(d_AWA, 0, number_vertices_ * number_vertices_ * sizeof(double));
        // cudaMemset(d_, 0, number_vertices_  * sizeof(do));
        cudaMemset(d_AWA, 0, number_vertices_*number_vertices_*sizeof(double));
        cudaMemset(d_AWb, 0,  number_vertices_*sizeof(double));
        cudaMemset(d_d, 0, number_triangles_ * 3 * sizeof(double));
        cudaMemset(d_c_unit, 0, number_triangles_*3*sizeof(double));
        cudaMemset(d_vertices_, 0, number_vertices_*3*sizeof(double));


        cudaError_t err1 = cudaGetLastError();
        if (err1 != cudaSuccess) {
            printf("CUDA Error: %s\n", cudaGetErrorString(err1));
            exit(1);
        }
        
        int num_obs[1];
        *num_obs = number_observation/2; 

        std::cout << "obs\n";
        std::cout << observation[0] << " " << observation[3] << std::endl << std::endl;;

        // std::cout << number_vertices_ << " " << number_observation << std::endl; exit(1);
        // std::cout << "hier\n";
        cudaMemcpy(d_obs_, observation.data(), number_vertices_*2*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_number_observation_, num_obs, 1*sizeof(int), cudaMemcpyHostToDevice); 

        // auto start1 = std::chrono::high_resolution_clock::now();

        // std::cout << "hier\n"; 

        int numBlock = 1;
        int numthread = number_vertices_;


        if (numthread> 1024) {
            int newBlocksize = std::ceil(static_cast<float>(numthread)/1024);
            numBlock = newBlocksize;
            numthread = std::ceil(static_cast<float>(numthread) / newBlocksize);
        }

        
 
        obs2unitvector<<<numBlock,numthread>>>(d_obs_, d_vertices_, d_K_, d_number_observation_);
        cudaDeviceSynchronize();

        // double *h_vertices;
        // h_vertices = (double*)malloc(number_vertices_*3*sizeof(double));
        cudaMemcpy(h_vertices_, d_vertices_, number_vertices_*3*sizeof(double), cudaMemcpyDeviceToHost);

        // for(int i=0; i<number_vertices_;i++) {
        //     std::cout << h_vertices_[i*3+0] << " " << h_vertices_[i*3+1] << " " << h_vertices_[i*3+2] << std::endl;
        // }
        // exit(1);

        numBlock = 1;
        numthread = number_triangles_*3;

        if (numthread > 1024) {
            int newBlocksize = std::ceil(static_cast<float>(numthread)/1024);
            numBlock = newBlocksize;
            numthread = std::ceil(static_cast<float>(numthread) / newBlocksize);

            numthread += (3 - (numthread % 3));

            if(numthread > 1022) {
                numthread = 1022;
                numBlock += 1;
            }
        }
        cudaDeviceSynchronize();


    //     cudaMemcpy(h_vertices_, d_vertices_, number_vertices_*sizeof(double), cudaMemcpyDeviceToHost);

    //     for (int i = 0; i<number_vertices_;i++) {
    //         std::cout <<h_vertices_[i*3] << " " << h_vertices_[i*3+1] << " " << h_vertices_[i*3+2] << std::endl;
    //     }
    // exit(1);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));
            exit(1);
        }

        // std::cout << numBlock << " " << numthread << std::endl; exit(1); 
        compute_constantUnits<<<numBlock, numthread>>>(d_c_unit, d_triangles_, d_number_triangles_, d_vertices_);
        cudaDeviceSynchronize();
        // std::cout << number_triangles_ << " "<< number_vertices_ << std::endl; 
        // exit(1);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));
            exit(1);
        }



        numBlock = 1;
        numthread = number_triangles_ * 3;

        if (numthread > 1024) {
            int newBlocksize = std::ceil(static_cast<float>(numthread)/1024);
            numBlock = newBlocksize;
            numthread = std::ceil(static_cast<float>(numthread) / newBlocksize);

            numthread += (3 - (numthread % 3));

            if(numthread > 1022) {
                numthread = 1022;
                numBlock += 1;
            }
        }


        compute_d<<<numBlock, numthread>>>(d_d, d_reference_, d_number_triangles_, d_triangles_);



        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));
        }

        // std::cout << "hier\n"; 
        
        // auto end1 = std::chrono::high_resolution_clock::now();
        // // Berechnung der verstrichenen Zeit
        // auto elapsed1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);

        // // Ausgabe der verstrichenen Zeit in Sekunden
        // std::cout <<" Verstrichene Zeit: " << elapsed1.count() << " Mikrosekunden" << std::endl;



    }


    __host__ void optGPU::getVertices(std::vector<Eigen::Vector3d> &vertices) {

        std::cout << vertices.size() << std::endl;
            double *rx = (double*)m_cholSparseR->x;

            //  for(int i=0; i<number_vertices_; i++) {
            //     std::cout <<rx[i] << std::endl;
            // }

            // cudaMemcpy(d_vertices_, h_vertices_, number_vertices_*3*sizeof(double), cudaMemcpyDeviceToHost);
                // cudaError_t err = cudaGetLastError();
                // if (err != cudaSuccess) {
                //     printf("CUDA Error: %s\n", cudaGetErrorString(err));
                // }

            // std::vector<Eigen::Vector3d> tmp_vec;
            for(int i = 0; i< number_vertices_; i++) {
                // Eigen::Vector3d tmp;

                vertices[i].x() = h_vertices_[i*3] * rx[i];
                vertices[i].y() = h_vertices_[i*3+1] * rx[i];
                vertices[i].z() = h_vertices_[i*3+2] * rx[i];


                // std::cout <<h_vertices_[i*3] * rx[i] << " " << h_vertices_[i*3+1] * rx[i] << " " << h_vertices_[i*3+2] * rx[i]<< std::endl;

                // tmp << h_vertices_[i*3] * rx[i], h_vertices_[i*3+1] * rx[i], h_vertices_[i*3+2] * rx[i];
            }

            // vertices = tmp_vec;
            // exit(1);
    }

__host__ void optGPU::setDB(database *db) {
    db_ = db;
}




    __host__ void optGPU::run() {

        std::vector<double> timer_tmp;
        auto start_gesamt = std::chrono::high_resolution_clock::now();

        for (int iter = 0; iter < max_iteration_; iter++) {

            // cudaMemset(d_d, 0, number_triangles_ * 3 * sizeof(double));
    //         cudaMemcpy(h_d, d_d, number_triangles_*3*sizeof(double), cudaMemcpyDeviceToHost);

    //         std::vector<Eigen::Vector3i> tri;
    //         // std::vector<Eigen::Vector3d> ver;
    //         std::cout << "hier" << std::endl;
    //         for(int i=0; i < number_triangles_; i++) {

    //             int t1 = triangles_[i].x();
    //             int t2 = triangles_[i].y();
    //             int t3 = triangles_[i].z();

    //             Eigen::Vector3i tmp_tri;
    //             tmp_tri << i*3, i*3+1, i*3+2;

    //             Eigen::Vector3d v1, v2, v3;
    //             v1 << h_vertices_[t1*3] * h_d[i*3], h_vertices_[t1*3+1] * h_d[i*3], h_vertices_[t1*3+2] * h_d[i*3]; 
    //             v2 << h_vertices_[t2*3] * h_d[i*3+1], h_vertices_[t2*3+1] * h_d[i*3+1], h_vertices_[t2*3+2] * h_d[i*3+1]; 
    //             v3 << h_vertices_[t3*3] * h_d[i*3+2], h_vertices_[t3*3+1] * h_d[i*3+2], h_vertices_[t3*3+2] * h_d[i*3+2]; 

    //             //  v1 <<1,12,13; 
    //             //  v2 <<1,21,31; 
    //             //  v3 <<11,12,31; 

    // // std::cout << ver.size() << " " << i*3 << std::endl;
    //             tri.push_back(tmp_tri);
    //             ver[i*3] = v1;
    //             ver[i*3+1] = v2;
    //             ver[i*3+2] = v3;
    //         }
            
    //         db_->setVerticesAndTriangles(ver, tri);

    //         static int iii= 0;
    //         if(iii <= 10)
    //         std::this_thread::sleep_for(std::chrono::milliseconds(500));
    //         else 
    //         std::this_thread::sleep_for(std::chrono::milliseconds(50));

            // iii++;
            // database::setVerticesAndTriangles(std::vector<Eigen::Vector3d> vertices, std::vector<Eigen::Vector3i> triangles)
            auto start1 = std::chrono::high_resolution_clock::now();


            cudaMemset(d_information_matrix, 0, number_triangles_ * 9  * sizeof(double));
            cudaMemset(d_determinant, 0, number_triangles_*6*sizeof(double));   
            cudaMemset(d_g, 0, number_triangles_ * 3 * sizeof(double));
            cudaMemset(d_H, 0, number_triangles_ * 9 * sizeof(double));
            // cudaEventRecord(start_, 0);
            
            

            compute_cost_new<<<blocksize_cost, threadsize_cost, 0, stream_cost>>>(d_bar_, d_triangles_, d_number_triangles_, d_d, d_vertices_, d_cost_);
            cudaEventRecord(event_cost, stream_cost);
            // cudaDeviceSynchronize();
            computeG_new<<<blocksize_G, threadsize_G, 0, stream_G>>>(d_g, d_triangles_, d_vertices_, d_d, d_bar_, d_c_unit, d_number_triangles_);
            cudaEventRecord(event_G, stream_G);
            // cudaDeviceSynchronize();

            computeH_new<<<blocksize_H, threadsize_H, 0, stream_H>>>(d_H, d_triangles_, d_vertices_, d_d, d_c_unit, d_number_triangles_);
            cudaEventRecord(event_H, stream_H);
            // cudaDeviceSynchronize();
            
            cudaStreamWaitEvent(stream_det, event_H, 0);
            compute_determinante_new<<<blocksize_det, threadsize_det, 0, stream_det>>>(d_H, d_number_triangles_, d_determinant);
            cudaEventRecord(event_det, stream_det);
            // cudaDeviceSynchronize();

            cudaStreamWaitEvent(stream_adj, event_H, 0);
            compute_adjugate_new<<<blocksize_adj, threadsize_adj, 0, stream_adj>>>(d_H, d_number_triangles_, d_information_matrix);
            cudaEventRecord(event_adj, stream_adj);

            // cudaDeviceSynchronize();
            


            cudaStreamWaitEvent(stream_up, event_det, 0);
            cudaStreamWaitEvent(stream_up, event_adj, 0);
            cudaStreamWaitEvent(stream_up, event_G, 0);

            update_dx<<<blocksize_up, threadsize_up, 0, stream_up>>>(d_d, d_number_triangles_, d_information_matrix, d_determinant, d_g);
            cudaEventRecord(event_up, stream_up);
            // cudaDeviceSynchronize();


            cudaStreamSynchronize(stream_up);
            cudaStreamSynchronize(stream_cost);
            // cudaStreamWaitEvent(stream_cost, event_G, 0);



            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("CUDA Error: %s\n", cudaGetErrorString(err));
            }
            auto end1 = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end1 - start1);

            double cost1[20];
            cudaMemcpy(cost1, d_cost_, 1*20*sizeof(double), cudaMemcpyDeviceToHost);

            // std::cout << "Itertation: " << iter <<" Error: " << cost1[0] << " Time: " << elapsed.count() << "ns" << std::endl;
            timer_tmp.push_back(elapsed.count());
            // printf("Iteration: %d\ttime iteration: %dus\t cuda time: \tcost: \n", iter, elapsed.count(), cost1[0]);

        }
        cudaError_t err1 = cudaGetLastError();
        if (err1 != cudaSuccess) {
            printf("CUDA Error: %s\n", cudaGetErrorString(err1));
        }

       

        int blocksize;
        int threadsize;

        blocksize = 1;
        threadsize = 3*number_triangles_;

        if(threadsize > 1024) {
            int newBlocksize = std::ceil(static_cast<float>(threadsize)/1024);
            blocksize = newBlocksize;
            threadsize = std::ceil(static_cast<float>(threadsize) / newBlocksize);
        }

        auto start_AWAb = std::chrono::high_resolution_clock::now();

        compute_AWA<<<blocksize, threadsize, 0, stream_AWA>>>(d_H, d_triangles_, d_number_triangles_, d_number_vertices_, d_AWA);
        cudaEventRecord(event_AWA, stream_AWA);

        compute_AWB<<<blocksize, threadsize, 0, stream_AWb>>>(d_H, d_d, d_triangles_, d_number_triangles_, d_number_vertices_, d_AWb);
        cudaEventRecord(event_AWb, stream_AWb);

        

        cudaStreamSynchronize(stream_AWb);
        cudaStreamSynchronize(stream_AWA);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));
        }

        cudaMemcpy(h_AWA, d_AWA, number_vertices_*number_vertices_*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_AWb, d_AWb, number_vertices_*sizeof(double), cudaMemcpyDeviceToHost);

        auto end_AWAb = std::chrono::high_resolution_clock::now();

        
  

        auto start_chol = std::chrono::high_resolution_clock::now();

        double *Sx = (double *)m_cholSparseS->x;
        int *Sp = (int *) m_cholSparseS->p;
        int *Si = (int *) m_cholSparseS->i;

        for(int col_id = 0; col_id < number_vertices_; col_id++) {

            int start = Sp[col_id];
            int end = Sp[col_id+1];
            for(int idx = start; idx < end; idx++) {
                int row = Si[idx];
                Sx[idx] = h_AWA[row * number_vertices_ + col_id];
                // std::cout <<  col_id << " " << row << " " << Sx[idx] << std::endl;
            }
        }



        double *Ex = (double*)m_cholSparseE->x;

        for(int ii = 0; ii < number_vertices_; ii++) {
            Ex[ii] = h_AWb[ii];//hhhh[ii];

            // std::cout << Ex[ii] << std::endl;
        }

 

            cholmod_factorize(m_cholSparseS, m_cholFactorS, &m_cS); 
            
            m_cholSparseR = cholmod_solve (CHOLMOD_A, m_cholFactorS, m_cholSparseE, &m_cS) ;

            //double *rx = (double*)m_cholSparseR->x;
            auto end_chol = std::chrono::high_resolution_clock::now();

            auto end_gesamt = std::chrono::high_resolution_clock::now();
            auto elapsed_gesamt = std::chrono::duration_cast<std::chrono::microseconds>(end_gesamt - start_gesamt);
            auto elapsed_chol = std::chrono::duration_cast<std::chrono::microseconds>(end_chol - start_chol);
            auto elapsed_AWAb = std::chrono::duration_cast<std::chrono::microseconds>(end_AWAb- start_AWAb);


            // std::cout << "AWAb time: " << elapsed_AWAb.count() << "ms cholmod time: " << elapsed_chol.count() << "ms total_time: " << elapsed_gesamt.count() << "ms" << std::endl;
            
            double summe = 0;
            for(int i=0; i<timer_tmp.size(); i++) {
                summe += timer_tmp[i];
            }
            summe /= timer_tmp.size();

            total_time_.push_back(elapsed_gesamt.count());
            cholmod_timer_.push_back(elapsed_chol.count());
            preparation_.push_back(elapsed_AWAb.count());
            subtriangle_.push_back(summe);



        //     std::string filename = "timer_" + std::to_string(number_triangles_) + ".csv";

        // std::ofstream file(filename);

        // if (file.is_open()) {
        //     // Erste Zeile: Nummerierung
        //      file << number_triangles_;
        //       file << ",";
        //       file << "total_time [us]";
        //       file << ",";
        //       file << "preparation AWA [us]";
        //       file << ",";
        //       file << "cholmod_timer [us]";
        //       file << ",";
        //       file << "subtriangle [ns]";
        //       file << "\n";
              
        //     for (size_t i = 0; i < total_time_.size(); ++i) {
        //         file << " ";
        //         file << ",";
        //         file << total_time_[i];  // Schreibe den Index
        //         file << ",";
        //         file << preparation_[i];
        //         file << ",";
        //         file << cholmod_timer_[i];
        //         file << ",";
        //         file << subtriangle_[i];
        //         file << "\n";
                
        //     }

        //     file << "\n";
        //     file << "\n";

        //     for (size_t i = 0; i < timer_tmp.size(); ++i) {
        //         file << timer_tmp[i];  // Schreibe den Index
        //         file << ",";                
        //     }
        //     file << "\n";  // Neue Zeile

        //     // Zweite Zeile: Datenwerte
        //     // for (size_t i = 0; i < data.size(); ++i) {
        //     //     file << data[i];  // Schreibe den Datenwert
        //     //     if (i < data.size() - 1) {
        //     //         file << ",";  // Komma als Trennzeichen
        //     //     }
        //     // }
        //     // file << "\n";  // Datei schließen

        //     file.close();
        //     std::cout << "Daten erfolgreich in " << filename << " gespeichert." << std::endl;
        // } else {
        //     std::cerr << "Fehler beim Öffnen der Datei!" << std::endl;
        // }
    }



}