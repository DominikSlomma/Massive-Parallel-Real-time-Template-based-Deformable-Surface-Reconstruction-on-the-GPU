#include "main.cuh"
#include <chrono>


namespace test_opt {
    __host__ optGPU::optGPU(int max_iteration, std::vector<Eigen::Vector3d> &vertices, std::vector<Eigen::Vector3i> &triangles, Eigen::Matrix3d K, bool verbose) 
    : e_vertices_(vertices), e_triangles_(triangles), e_reference_(vertices)
    {
        invK_ = K.inverse();
        int num_triangles = triangles.size();
        int num_vertices = vertices.size();
    }

    __host__ void optGPU::setParamater(double *observation, std::unordered_map<int,int> &unordered_mapping_vertices, std::unordered_map<int,int> &unordered_mapping_triangles, int number_vertices, int number_triangles, int number_observation) {
        observation_ = observation;
        // vertices_unordered_mapping_ = unordered_mapping_vertices;
        // triangle_unordered_mapping_ = unordered_mapping_triangles;
        number_vertices_ = number_vertices;
        number_triangles_ = number_triangles;
        number_observation_ = number_observation;
    }

    __host__ void optGPU::initialize() {

        
        // change the vertices to unit vector and depth!

        for(int i=0; i < number_observation_; i++) {
            int face_id = observation_[i*6];
  
            // double alpha = observation_[i*6+3];
            // double beta = observation_[i*6+4];
            // double gamma = observation_[i*6+5];
            // std::cout << "begin\n";
        

            // Todo:  just 3 observations! optimise it!
            double *vertex, *vertex1;
            // if(int(alpha) == 1) {
                vertex = e_vertices_[i].data();
                vertex1 = e_reference_[i].data();
            // } else if (int(beta) == 1)
            // {
            //     vertex = e_vertices_[e_triangles_[face_id].y()].data();
            //     vertex1 = e_reference_[e_triangles_[face_id].y()].data();

            // } else if (int(gamma) == 1)
            // {
            //     vertex = e_vertices_[e_triangles_[face_id].z()].data();
            //     vertex1 = e_reference_[e_triangles_[face_id].z()].data();
        
            // } else {

            //     continue;
            // }
            Eigen::Vector3d uvt;
            uvt << observation_[i*6+1], observation_[i*6+2], 1;
            uvt = invK_ * uvt; 

            double phi = std::atan2(uvt[0], uvt[2]);
            double theta = std::atan2(uvt[1], std::sqrt(uvt[0] * uvt[0] + uvt[2]*uvt[2]));
            double d = std::sqrt(vertex1[0]*vertex1[0] + vertex1[1]*vertex1[1] + vertex1[2]*vertex1[2]);
            // std::cout << phi << " "<< theta << " "<< d << " " << std::endl;
            
            vertex[0] = phi;
            vertex[1] = theta;
            vertex[2] = d;
            std::cout << phi << " " << theta << " d: " << d << std::endl;
            // break;
        }

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        reference_ = (double*)malloc(number_vertices_*3*sizeof(double)); // kann ich in den anfang packen!
        vertices_ = (double*)malloc(number_vertices_*3*sizeof(double)); // ist ok
        sub_vert_ = (double*)malloc(number_triangles_*3*3*sizeof(double));
        cost_ = (double*)malloc(number_triangles_*3*3*sizeof(double));
        dx_ = (double*)malloc(number_triangles_*20*sizeof(double));
    std::cout << "her\n";

        memset( reference_, 0, number_vertices_*3*sizeof(double));
        memset( vertices_, 0, number_vertices_*3*sizeof(double));
        memset( sub_vert_, 0, number_triangles_*3*3*sizeof(double));
        memset( cost_, 0, number_triangles_*20*sizeof(double));
        memset( dx_, 0, number_triangles_*20*sizeof(double));

    std::cout << "her\n";

        for(int i=0;i<number_triangles_;i++) {
            triangles_.push_back(e_triangles_[i].x());
            triangles_.push_back(e_triangles_[i].y());
            triangles_.push_back(e_triangles_[i].z());
        }
        std::cout << triangles_.size() << std::endl;


        for(int id=0; id< number_vertices_; id++) {
            Eigen::Vector3d vertex = e_vertices_[id];
            Eigen::Vector3d ref_vertex = e_reference_[id];

    
            vertices_[id * 3] = vertex.x();
            vertices_[id * 3 + 1] = vertex.y();
            vertices_[id * 3 + 2] = vertex.z();

            std::cout << vertex.z() << " " << ref_vertex.norm() << std::endl;

            reference_[id * 3] = ref_vertex.x();
            reference_[id * 3 + 1] = ref_vertex.y();
            reference_[id * 3 + 2] = ref_vertex.z();// v_mask_ = (char*)malloc(number_vertices_*number_vertices_*sizeof(char)); // v_mask initialisieren!  
        }
    
        // Cuda init
        // Todo: bring das cuda init für die triangles und reference informationen in den konstruktor, da man nur einmal die init brauch und nicht jedes mal!
        // Todo: Bring alle mallocs die nicht mehr als einmal aufgerufen werden in den Konstruktor!

        cudaError_t err1 = cudaMalloc((void**)&d_vertices_, number_vertices_*3*sizeof(double));
        cudaError_t err2 = cudaMalloc((void**)&d_reference_, number_vertices_*3*sizeof(double));
        cudaError_t err3 = cudaMalloc((void**)&d_triangles_, number_triangles_*3*sizeof(int));
        cudaError_t err4 = cudaMalloc((void**)&d_sub_vert_, number_triangles_*3*3*sizeof(double));
        cudaError_t err5 = cudaMalloc((void**)&d_cost_, number_triangles_*20*sizeof(double));
        cudaError_t err6 = cudaMalloc((void**)&d_dx_, number_triangles_*20*sizeof(double));

        if ((err1 != cudaSuccess) || (err2 != cudaSuccess) || (err3 != cudaSuccess) || (err4 != cudaSuccess)) {
            std::cerr << "cudaMalloc Fehler: " << cudaGetErrorString(err1) << "\t" << cudaGetErrorString(err2) << "\t" << cudaGetErrorString(err3) << "\t" << cudaGetErrorString(err4) << std::endl;
            exit(1);
        }

        // copy data fromhost to device
        cudaMemcpy(d_vertices_, vertices_, number_vertices_*3*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_reference_, reference_, number_vertices_*3*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_triangles_, triangles_.data(), number_triangles_*3*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_sub_vert_, sub_vert_, number_triangles_*3*3*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_cost_, cost_, number_triangles_*20*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_dx_, dx_, number_triangles_*20*sizeof(double), cudaMemcpyHostToDevice);


        
        int num_points = number_vertices_;
        int num_obs = number_observation_;
        int num_faces = number_triangles_;
   

    }

    __host__ void optGPU::getVertices(std::vector<Eigen::Vector3d> &vertices) {
        std::cout << "\n Daten\n";
        // Kopiere Daten von der GPU zur CPU
        cudaMemcpy(cost_, d_cost_, number_triangles_*20*sizeof(double), cudaMemcpyDeviceToHost);

        std::cout << "error norm" << std::endl;
        for(int i=0;i<20;i++) {
            std::cout <<"Iteration: " << i <<"\ttriangle1: " << cost_[i] <<"\ttriangle2: " << cost_[i+20] << std::endl; 
        }

        std::cout << "\ndx\n";
        cudaMemcpy(dx_, d_dx_, number_triangles_*20*sizeof(double), cudaMemcpyDeviceToHost);
        for(int i=0;i<20;i++) {
            std::cout <<"Iteration: " << i <<"\ttriangle1: " << dx_[i] <<"\ttriangle2: " << dx_[i+20] << std::endl; 
        }

        std::cout << "\n Vertices \n";
        
        cudaMemcpy(vertices_, d_vertices_,  number_vertices_*3*sizeof(double), cudaMemcpyDeviceToHost);
        for(int i=0;i<number_vertices_;i++) {
            std::cout << vertices_[i*3] << " " << vertices_[i*3+1] << " " << vertices_[i*3+2] << std::endl;
        }

        std::cout << "\n sub_vert \n";

        cudaMemcpy(sub_vert_, d_sub_vert_, number_triangles_*3*3*sizeof(double), cudaMemcpyDeviceToHost);
        for(int i=0;i<number_triangles_;i++) {
            std::cout << sub_vert_[i*9+0] << " " << sub_vert_[i*9+1] << " " << sub_vert_[i*9+2] << "\t\t"
                      << sub_vert_[i*9+3] << " " << sub_vert_[i*9+4] << " " << sub_vert_[i*9+5] << "\t\t"
                      << sub_vert_[i*9+6] << " " << sub_vert_[i*9+7] << " " << sub_vert_[i*9+8] << "\t\t" 
                      << std::endl;
        }

        std::cout << "\n ende \n";
    }
    
    __host__ void optGPU::run() {
        std::cout << "Starting the computation on the GPU" << std::endl;

        // Kernel call!
        optimise_single_triangles<<<2,1>>>(d_triangles_, d_reference_, d_vertices_, d_sub_vert_, d_cost_, d_dx_);

    }   

}

__device__ void compute_error(Eigen::Vector3d &error, double* reference, double* vertices, double* test) {
    

    double v1[3];
    double v2[3];
    double v3[3];
    v1[0]= sin(vertices[0]) * cos(vertices[1]) * vertices[2]; 
    v1[1]= sin(vertices[1]) * vertices[2]; 
    v1[2]= cos(vertices[0]) * cos(vertices[1]) * vertices[2];
    
    v2[0]= sin(vertices[3]) * cos(vertices[4]) * vertices[5]; 
    v2[1]= sin(vertices[4]) * vertices[5]; 
    v2[2]= cos(vertices[3]) * cos(vertices[4]) * vertices[5];
    
    v3[0]= sin(vertices[6]) * cos(vertices[7]) * vertices[8]; 
    v3[1]= sin(vertices[7]) * vertices[8]; 
    v3[2]= cos(vertices[6]) * cos(vertices[7]) * vertices[8];


    double v1_ref[3];
    double v2_ref[3];
    double v3_ref[3];
    v1_ref[0]=reference[0];
    v1_ref[1]=reference[1];
    v1_ref[2]=reference[2];
    
    v2_ref[0]=reference[3];
    v2_ref[1]=reference[4];
    v2_ref[2]=reference[5];
    
    v3_ref[0]=reference[6];
    v3_ref[1]=reference[7];
    v3_ref[2]=reference[8];


    double d12, d12_ref, d13, d13_ref, d23, d23_ref;
    d12 = sqrt((v1[0]-v2[0])*(v1[0]-v2[0])+(v1[1]-v2[1])*(v1[1]-v2[1])+(v1[2]-v2[2])*(v1[2]-v2[2]));//  (v1-v2).norm();
    d12_ref = sqrt((v1_ref[0]-v2_ref[0])*(v1_ref[0]-v2_ref[0])+(v1_ref[1]-v2_ref[1])*(v1_ref[1]-v2_ref[1])+(v1_ref[2]-v2_ref[2])*(v1_ref[2]-v2_ref[2]));//(v1_ref-v2_ref).norm();

    d13 = sqrt((v1[0]-v3[0])*(v1[0]-v3[0])+(v1[1]-v3[1])*(v1[1]-v3[1])+(v1[2]-v3[2])*(v1[2]-v3[2]));//(v1-v3).norm();
    d13_ref = sqrt((v1_ref[0]-v3_ref[0])*(v1_ref[0]-v3_ref[0])+(v1_ref[1]-v3_ref[1])*(v1_ref[1]-v3_ref[1])+(v1_ref[2]-v3_ref[2])*(v1_ref[2]-v3_ref[2]));//(v1_ref-v3_ref).norm();

    d23 = sqrt((v2[0]-v3[0])*(v2[0]-v3[0])+(v2[1]-v3[1])*(v2[1]-v3[1])+(v2[2]-v3[2])*(v2[2]-v3[2]));//(v2-v3).norm();
    d23_ref = sqrt((v2_ref[0]-v3_ref[0])*(v2_ref[0]-v3_ref[0])+(v2_ref[1]-v3_ref[1])*(v2_ref[1]-v3_ref[1])+(v2_ref[2]-v3_ref[2])*(v2_ref[2]-v3_ref[2]));//(v2_ref-v3_ref).norm();
    double e1 = d12 - d12_ref;
    double e2 = d13 - d13_ref;
    double e3 = d23 - d23_ref;

    test[0] = e1;
    test[1] = e2;
    test[2] = e3;
    
    error << e1, e2, e3;
}
__device__ void compute_Jacobian(Eigen::Matrix3d &H, double* vertices) { 
   
    double v1[3];
    double v2[3];
    double v3[3];
    v1[0]= sin(vertices[0]) * cos(vertices[1]) * vertices[2]; 
    v1[1]= sin(vertices[1]) * vertices[2]; 
    v1[2]= cos(vertices[0]) * cos(vertices[1]) * vertices[2];
    
    v2[0]= sin(vertices[3]) * cos(vertices[4]) * vertices[5]; 
    v2[1]= sin(vertices[4]) * vertices[5]; 
    v2[2]= cos(vertices[3]) * cos(vertices[4]) * vertices[5];
    
    v3[0]= sin(vertices[6]) * cos(vertices[7]) * vertices[8]; 
    v3[1]= sin(vertices[7]) * vertices[8]; 
    v3[2]= cos(vertices[6]) * cos(vertices[7]) * vertices[8];

    double u1[3];
    double u2[3];
    double u3[3];
    u1[0]= sin(vertices[0]) * cos(vertices[1]); 
    u1[1]= sin(vertices[1]); 
    u1[2]= cos(vertices[0]) * cos(vertices[1]);
    
    u2[0]= sin(vertices[3]) * cos(vertices[4]); 
    u2[1]= sin(vertices[4]); 
    u2[2]= cos(vertices[3]) * cos(vertices[4]);
    
    u3[0]= sin(vertices[6]) * cos(vertices[7]); 
    u3[1]= sin(vertices[7]); 
    u3[2]= cos(vertices[6]) * cos(vertices[7]);


   


    double d12, d13, d23;
    d12 = sqrt((v1[0]-v2[0])*(v1[0]-v2[0])+(v1[1]-v2[1])*(v1[1]-v2[1])+(v1[2]-v2[2])*(v1[2]-v2[2]));//v12.norm();
    d13 = sqrt((v1[0]-v3[0])*(v1[0]-v3[0])+(v1[1]-v3[1])*(v1[1]-v3[1])+(v1[2]-v3[2])*(v1[2]-v3[2]));//(v1-v3).norm();
    d23 = sqrt((v2[0]-v3[0])*(v2[0]-v3[0])+(v2[1]-v3[1])*(v2[1]-v3[1])+(v2[2]-v3[2])*(v2[2]-v3[2]));//(v2-v3).norm();

    double v12[3];
    double v13[3];
    double v23[3];
    v12[0] = v1[0]-v2[0];
    v12[1] = v1[1]-v2[1];
    v12[2] = v1[2]-v2[2];
    
    v13[0] = v1[0]-v3[0];
    v13[1] = v1[1]-v3[1];
    v13[2] = v1[2]-v3[2];

    v23[0] = v2[0]-v3[0];
    v23[1] = v2[1]-v3[1];
    v23[2] = v2[2]-v3[2];


    double J12_1, J12_2, J13_1, J13_3, J23_2, J23_3;

    J12_1 = u1[0]*v12[0] / d12 + u1[1]*v12[1] / d12 + u1[2]*v12[2] / d12;
    J13_1 = u1[0]*v13[0] / d13 + u1[1]*v13[1] / d13 + u1[2]*v13[2] / d13;

    J12_2 = -u2[0]*v12[0] / d12 + (-u2[1]*v12[1] / d12) + (-u2[2]*v12[2] / d12);
    J23_2 = u2[0]*v23[0] / d23 + u2[1]*v23[1] / d23 + u2[2]*v23[2] / d23;

    J13_3 = -u3[0]*v13[0] / d13 + (-u3[1]*v13[1] / d13) + (-u3[2]*v13[2] / d13);
    J23_3 = -u3[0]*v23[0] / d23 + (-u3[1]*v23[1] / d23) + (-u3[2]*v23[2] / d23);

    H << J12_1, J12_2, 0,
         J13_1, 0, J13_3,
         0, J23_2, J23_3;
    // H = J;
}

__device__
void distance_only_gn(int* triangles, double* reference, double* vertices, double* sub_vert, double* cost_, double* dx_) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ double sub_ver[9], sub_ref[9];
    int triangle1 = triangles[blockIdx.x*3];
    int triangle2 = triangles[blockIdx.x*3+1];
    int triangle3 = triangles[blockIdx.x*3+2];

    // cost_[blockIdx.x*3 + 0] = (int)triangle1;
    // cost_[blockIdx.x*3 + 1] = (int)triangle2;
    // cost_[blockIdx.x*3 + 2] = (int)triangle3;
    // return;

    sub_ver[0] = vertices[triangle1*3+0];
    sub_ver[1] = vertices[triangle1*3+1];
    sub_ver[2] = vertices[triangle1*3+2];

    sub_ver[3] = vertices[triangle2*3+0];
    sub_ver[4] = vertices[triangle2*3+1];
    sub_ver[5] = vertices[triangle2*3+2];

    sub_ver[6] = vertices[triangle3*3+0];
    sub_ver[7] = vertices[triangle3*3+1];
    sub_ver[8] = vertices[triangle3*3+2];


    sub_ref[0] = reference[triangle1*3];
    sub_ref[1] = reference[triangle1*3+1];
    sub_ref[2] = reference[triangle1*3+2];

    sub_ref[3] = reference[triangle2*3];
    sub_ref[4] = reference[triangle2*3+1];
    sub_ref[5] = reference[triangle2*3+2];

    sub_ref[6] = reference[triangle3*3];
    sub_ref[7] = reference[triangle3*3+1];
    sub_ref[8] = reference[triangle3*3+2];
    
    int iter_max = 20;
    double cost = 0;
    Eigen::Vector3d e, dx, g;
    Eigen::Matrix3d H, J, invH;
    // double J[9];
    double test[3];
    test[0]=0;
    test[1]=0;
    test[2]=0;
    for(int iter=0; iter<iter_max; iter++) {
        cost = 0;
        compute_error(e, sub_ref, sub_ver, test);
        compute_Jacobian(J, sub_ver);

        H = J.transpose() * J;
        g = -J.transpose() * e;

        invH = H.inverse();

        dx = invH*g;

        
        sub_ver[2] += dx.x();
        sub_ver[5] += dx.y();
        sub_ver[8] += dx.z();

        cost = e.norm(); //sqrt(((e[0]*e[0])+(e[1]*e[1])+(e[2]*e[2]))/3);//sqrt(test[0]*test[0] + test[1]*test[1] + test[2]*test[2]); //e.norm();
        dx_[blockIdx.x*20+iter] = dx.norm();
        cost_[blockIdx.x*20+iter] = cost;
    }
    // vertices = sub_ver;

    double v1[3];
    double v2[3];
    double v3[3];
    // v1[0]= std::sin(vertices[0]) * std::cos(vertices[1]) * vertices[2]; 
    // v1[1]= std::sin(vertices[1]) * vertices[2]; 
    // v1[2]= std::cos(vertices[0]) * std::cos(vertices[1]) * vertices[2];
    
    // v2[0]= std::sin(vertices[3]) * std::cos(vertices[4]) * vertices[5]; 
    // v2[1]= std::sin(vertices[4]) * vertices[5]; 
    // v2[2]= std::cos(vertices[3]) * std::cos(vertices[4]) * vertices[5];
    
    // v3[0]= sin(vertices[6]) * cos(vertices[7]) * vertices[8]; 
    // v3[1]= sin(vertices[7]) * vertices[8]; 
    // v3[2]= cos(vertices[6]) * cos(vertices[7]) * vertices[8];

    // sub_vert[blockIdx.x*9+0] = J(0,0);
    // sub_vert[blockIdx.x*9+1] = J(0,1);
    // sub_vert[blockIdx.x*9+2] = J(0,2);
    
    // sub_vert[blockIdx.x*9+3] = J(1,0);
    // sub_vert[blockIdx.x*9+4] = J(1,1);
    // sub_vert[blockIdx.x*9+5] = J(1,2);

    // sub_vert[blockIdx.x*9+6] = J(2,0);
    // sub_vert[blockIdx.x*9+7] = J(2,1);
    // sub_vert[blockIdx.x*9+8] = J(2,2);

    sub_vert[blockIdx.x*9+0] = sub_ver[0];
    sub_vert[blockIdx.x*9+1] = sub_ver[1];
    sub_vert[blockIdx.x*9+2] = sub_ver[2];
    
    sub_vert[blockIdx.x*9+3] = sub_ver[3];
    sub_vert[blockIdx.x*9+4] = sub_ver[4];
    sub_vert[blockIdx.x*9+5] = sub_ver[5];

    sub_vert[blockIdx.x*9+6] = sub_ver[6];
    sub_vert[blockIdx.x*9+7] = sub_ver[7];
    sub_vert[blockIdx.x*9+8] = sub_ver[8];


}

__global__ void optimise_single_triangles(int* triangles, double* reference, double* vertices, double* sub_vert, double* cost, double *dx) {
    // starte sub optimisation for each triangle. Each block represents a sub-triangle
    int num_triangles = 2;

    distance_only_gn(triangles, reference, vertices, sub_vert, cost, dx);
}

int main() {

    std::cout << "Test cuda optimisation\n"; 

    std::cout << "Create test mesh!\n";

    std::vector<Eigen::Vector3i> triangles;
    std::vector<Eigen::Vector3d> vertices;

    Eigen::Vector3i tmp; 
    tmp << 0,1,2;
    triangles.push_back(tmp);
    tmp << 1,2,3;
    triangles.push_back(tmp);

    Eigen::Vector3d tmp1;
    tmp1 << 1,1,6;
    vertices.push_back(tmp1);
    tmp1 << 1,2,6;
    vertices.push_back(tmp1);
    tmp1 << 2,2,6;
    vertices.push_back(tmp1);
    tmp1 << 3,2,6;
    vertices.push_back(tmp1);

    std::cout << "Set Camera calibration matrix\n";
    Eigen::Matrix3d K;
    K << 2,0,400,
         0,2,400,
         0,0,1;

    test_opt::optGPU* optim = new test_opt::optGPU(10, vertices, triangles, K, false);

    /*
     *
     * Create optimal observations for the testing!  
     *    
    */
   std::vector<double> obs;
   int num_vertices = vertices.size();
   for(int i=0;i<num_vertices;i++) {
    
    Eigen::Vector3d vertex = vertices[i];

    Eigen::Vector3d tmp;
    tmp = K*vertex;

    tmp /= tmp.z();



    obs.push_back(0);
    obs.push_back(tmp.x());
    obs.push_back(tmp.y());
    obs.push_back(0);
    obs.push_back(0);
    obs.push_back(0);
   }

   std::unordered_map<int,int> tmp123;
   // Startzeitpunkt


    optim->setParamater(obs.data(), tmp123, tmp123,num_vertices,triangles.size(),num_vertices);
    // for testing I need observations! create them manuelly just for testing case
    optim->initialize();
    std::cout << "her\n";
    auto start = std::chrono::high_resolution_clock::now();

    optim->run();
    
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    optim->getVertices(vertices);

     // Endzeitpunkt

    // Berechnung der verstrichenen Zeit
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Ausgabe der verstrichenen Zeit in Sekunden
    std::cout << "Verstrichene Zeit: " << elapsed.count() << " Mikrosekunden" << std::endl;



    //////////////////////////



//     cholmod_common c;
//     // c.useGPU=1;

//     cholmod_l_start(&c);
//     // setenv("CHOLMOD_USE_GPU", "1", 1);

//     c.useGPU=1;
//     c.supernodal = CHOLMOD_SUPERNODAL;

//     // int aaa=cholmod_l_gpu_stats(&c;)
 
//     int n = 3;
// long Ap[] = {0, 1, 3, 4};        // Zeilenzeiger
// long Ai[] = {0, 0, 1, 2};        // Spaltenindizes
// double Ax[] = {10, 2, 50, 30};  // Werte

//     // Vektor b
//     double b[] = {7, 10, 8, 4};

//     // CHOLMOD-Matrix erstellen
    
//     cholmod_sparse *A = cholmod_l_allocate_sparse(n, n, 4, 1, 1, 1, CHOLMOD_REAL, &c);
//      if (A == nullptr) {
//         std::cerr << "Fehler bei der Erstellung der Sparse-Matrix!" << std::endl;
//         return -1;
//     }

//     int test[] ={2};
//     // Zeiger zuweisen
//     // A->p = Ap;
//     // A->i = Ai;
//     // A->x = Ax;
//     long* app = (long*) A->p;
//     long* aii = (long*) A->i;
//     double* axx = (double*)A->x; 
//     for(int i=0;i< 4; i++) {
//         app[i] = Ap[i];
//         aii[i] = Ai[i];
//         axx[i] = Ax[i];
//     }
//     // std::memcpy(A->p, Ap, (n + 1) * sizeof(int));
//     // std::memcpy(A->i, Ai, 4 * sizeof(int));
//     // std::memcpy(A->x, Ax, 4 * sizeof(double));
//     // A->nz =test;
//     // A->z=CHOLMOD_DOUBLE;
//     // int testt= 4;
//     // std::memcpy(A->nzmax, testt, sizeof(int));
//     // A->nzmax
//     // c.precise=true;
//     // A->nzmax = 4;  // Anzahl der Nicht-Null-Elemente
//     // A->ncol = n;   // Anzahl der Spalten
//     // A->nrow = n;   // Anzahl der Zeilen
//     // A->stype = 1;  // Die Matrix ist symmetrisch
//     // A->itype = CHOLMOD_LONG; // Typ der Indizes
//     // A->xtype = CHOLMOD_REAL; // Typ der Werte
//     // A->dtype = CHOLMOD_DOUBLE; // Typ der Daten
//     // A->sorted = 1;  // Annahme: Die Matrix ist sortiert
//     // A->packed = 1;  // Annahme: Das Format ist "packed"

//     int stat = cholmod_l_check_sparse(A, &c);
//     std::cout << "status A: " << stat << std::endl;

//     std::cout << "Sparse-Matrix A:" << std::endl;
//     cholmod_l_print_sparse(A, "A", &c);

//     // Rechte Seite b als Dense Vektor
//     cholmod_dense *B = cholmod_l_allocate_dense(n, 1, n, CHOLMOD_REAL, &c);
//     if (B == nullptr) {
//         std::cerr << "Fehler bei der Erstellung des Dense-Vektors!" << std::endl;
//         cholmod_l_free_sparse(&A, &c);
//         cholmod_l_finish(&c);
//         return -1;
//     }
//     // c.cholmod_gpu_trsm_time
//     std::cout << "method: "<<  c.nmethods << std::endl;;
//     // cholmod_gpu_stats
    
//     double *Bx = (double *)B->x;
//     for (int i = 0; i < n; i++) {
//         Bx[i] = b[i];
//     }

//     std::cout << "useGPU: " << c.useGPU << std::endl;
//     std::cout << "quick_return_if_not_posdef: " << c.quick_return_if_not_posdef << std::endl;
    

//     std::cout << "hier:\n";
//     // Lösen des Gleichungssystems Ax = b
//     cholmod_factor *L = cholmod_l_analyze(A, &c); 
//     // L->is_super
//     std::cout << "super? :" << L->is_super << "\n";

//     cholmod_l_factorize(A, L, &c);
//     std::cout << "hier:\n";

//     cholmod_dense *X = cholmod_l_solve(CHOLMOD_A, L, B, &c);
//     std::cout << "hier:\n";

//     // Ergebnis anzeigen
//     double *Xx = (double *)X->x;
//     std::cout << "Lösung x = ";
//     for (int i = 0; i < n; i++) {
//         std::cout << Xx[i] << " ";
//     }
//     std::cout << std::endl;

//     std::cout << "GPU: "<<  cholmod_l_gpu_stats(&c) << std::endl;;

//     // Speicher freigeben
//     cholmod_l_free_sparse(&A, &c);
//     cholmod_l_free_dense(&B, &c);
//     cholmod_l_free_dense(&X, &c);
//     cholmod_l_free_factor(&L, &c);
//     cholmod_l_finish(&c);

}