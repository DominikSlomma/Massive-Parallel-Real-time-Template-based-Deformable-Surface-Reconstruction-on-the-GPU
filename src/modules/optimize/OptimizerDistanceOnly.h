#include <Eigen/Core>
#include <unordered_map>
#include <vector>
#include "suitesparse/cholmod.h"
#include "suitesparse/amd.h"

#include "open3d/Open3D.h"

#include <random>
#include <opencv2/opencv.hpp>
#include <chrono>



namespace stbr {
struct distance_sba_crsm	
{
    int nr, nc;   
    int nnz;      
    long *val;     
    long *colidx;  
    long *rowptr;  
};

class OptimizerDistanceOnly {
    public:
        OptimizerDistanceOnly(int max_iteration, std::vector<Eigen::Vector3d> &vertices, std::vector<Eigen::Vector3i> &triangles, Eigen::Matrix3d K, bool verbose);

        void setParamater(double *observation, std::unordered_map<int,int> &unordered_mapping_vertices, std::unordered_map<int,int> &unordered_mapping_triangles, int number_vertices, int number_triangles, int number_observation);
        void initialize();
        void run();
        void getVertices(std::vector<Eigen::Vector3d> &vertices);

    private:
        void distance_sba_crsm_alloc(struct distance_sba_crsm *sm, int nr, int nc, int nnz);
        void distance_sba_crsm_free(struct distance_sba_crsm *sm);
        int distance_sba_crsm_elmidx(struct distance_sba_crsm *sm, int i, int j);
        void constructSmask( distance_sba_crsm& Sidxij, int m, int& m_nS, char* m_smask);
        void compute_reprojection_error(double *e, double *obs, double *xyz, int *faces, Eigen::Matrix3d &K, int num_obs);
        bool solveCholmodGN( long* Ap, long* Aii, bool init, bool ordering, int m_ncams, int nnz);
        void constructCSSGN( long* Si, long* Sp, double* Sx, double* S, distance_sba_crsm& Sidxij, bool init, char* m_smask, int m_ncams);
        void constructAuxCSSGN( long *Ap, long *Aii, char* m_smask, int m_ncams);
        void compute_reprojection_jacobian(double *V, double *g, double *error, double *obs, double *xyz, int *faces, Eigen::Matrix3d K, int num_obs, distance_sba_crsm& Sidxij);
        void storeInV(double* V, int idx1, int idx2, double* J1, double* J2, distance_sba_crsm& Uidxij);
        void storeInV(double* V, int idx, double* J, distance_sba_crsm& Uidxij);
        void storeInG(double *g, double *J, double *e);
        void storeInG_distance(double *g, double J, double *e);
        void compute_distance_error(double *error, double *xyz, double *ref,int *faces, int number_faces, int offset);
        void compute_distance_jacobian(double *V, double *g, double *error, double *xyz, int *faces, int num_faces, int offset, distance_sba_crsm& Uidxij);
        void storeInV_distance(double* V, int idx, double J, distance_sba_crsm& Uidxij);
        void storeInV_distance(double* V, int idx1, int idx2, double J1, double J2, distance_sba_crsm& Uidxij);


        int max_iteration_ = 10;
        bool verbose_ = false;

        //Solve Sparse Matrix using CHOLMOD (http://www.cise.ufl.edu/research/sparse/SuiteSparse/) 
        cholmod_sparse *m_cholSparseS;				
        cholmod_factor *m_cholFactorS; 
        cholmod_common m_cS; 
        cholmod_dense  *m_cholSparseR, *m_cholSparseE;
        struct distance_sba_crsm Sidxij;

        

        std::vector<Eigen::Vector3d> e_vertices_;
        std::vector<Eigen::Vector3d> e_reference_;
        std::vector<Eigen::Vector3i> e_triangles_;

        Eigen::Matrix3d K_, invK_;

        double *vertices_;
        double *reference_;
        std::vector<int> triangles_;
        double *observation_ = nullptr;
        int number_vertices_ = 0;
        int number_triangles_ = 0;
        int number_observation_ = 0;
        int nnz=0;

        std::unordered_map<int, int> triangle_unordered_mapping_;
        std::unordered_map<int, int> vertices_unordered_mapping_;

        double* V_	=	nullptr;        
        double* g_	=	nullptr;        
        double* dx_	=	nullptr;        
        double* error_ = nullptr;
        char* v_mask_;

        long *Ap_  = nullptr;
	    long *Aii_ = nullptr;  
        long *Sp_, *Si_;
	    double* Sx_ = NULL;  
        bool init_ = false;
        bool ordering_ = true;
        double *rx_;    

	    double* Ex_;
	    int nMaxS_;	//maximum non-zero element in S matrix 


};

OptimizerDistanceOnly::OptimizerDistanceOnly(int max_iteration, std::vector<Eigen::Vector3d> &vertices, std::vector<Eigen::Vector3i> &triangles, Eigen::Matrix3d K, bool verbose) : verbose_(verbose), K_(K), max_iteration_(max_iteration), e_vertices_(vertices), e_triangles_(triangles) {
    // create reference
    for(int i=0; i< e_vertices_.size(); i++)
        e_reference_.push_back(e_vertices_[i]);

    invK_= K.inverse();

    setenv("CHOLMOD_USE_GPU", "1", 1);
}

void OptimizerDistanceOnly::getVertices(std::vector<Eigen::Vector3d> &vertices) {
    vertices = e_vertices_;
}

static void visualizer(std::shared_ptr<open3d::geometry::TriangleMesh> &mesh) {
    open3d::visualization::Visualizer visualizer;
    visualizer.CreateVisualizerWindow("Mesh Visualisierung", 1600, 900);
    visualizer.AddGeometry(mesh);
    open3d::visualization::ViewControl &view_control = visualizer.GetViewControl();
    view_control.SetLookat({0.0, 0.0, 0.0}); // Setze den Startpunkt der Kamera auf (0, 0, 0)
    view_control.SetFront({0.0, 0.0, -1.0});
    visualizer.Run();

}

void OptimizerDistanceOnly::run() {

    Eigen::Matrix3d K;
    K = K_;

    int num_obs = number_observation_;
    int num_faces = number_triangles_;
    int num_points = number_vertices_;

    double *reference = reference_;
    double *vertices = vertices_;
    int *new_faces = &triangles_[0];
    double *obs = observation_;
    
    long *Ap = Ap_;
    long *Aii = Aii_;  
    long *Sp = Sp_, *Si = Si_;
    double* Sx = Sx_;  
    bool init = init_;
    bool ordering = ordering_;
    double *rx = rx;;    

    double* Ex = Ex_;
    int nMaxS = nMaxS_;	


    // Zufallsgenerator initialisieren
    // std::random_device rd;
    // std::mt19937 gen(rd());
    // std::uniform_real_distribution<double> dis(0.0, 1.0); // Verteilung zwischen 0.0 und 1.0

    // Zufälligen double-Wert erzeugen
    // double random_value = dis(gen);
    // std::cout << random_value << " \n";
    // double mult = 10;
    // for(int i=0; i< num_points;i++) {
    //     double random_value = dis(gen);
    //     vertices[i*3+2] += random_value*mult; 
    // }

    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    double ed = 0;
    // std::chrono::duration<double> avg = (std::chrono::duration<double>)0.0;
    int iter = 1;
    double total_time = 0;
    for(iter = 1; iter < max_iteration_;iter++) {
        start = std::chrono::high_resolution_clock::now();
        memset( error_, 0, (( (num_faces*3))*sizeof(double) ));
        memset( V_, 0, nnz  * sizeof(double));
        memset( g_, 0, num_points * sizeof(double));

        // compute_reprojection_error(error_, obs, vertices, new_faces, K, num_obs);
        compute_distance_error(error_, vertices, reference, new_faces, num_faces, num_obs);

        // compute_reprojection_jacobian(V_, g_, error_, obs, vertices, new_faces, K, num_obs, Sidxij);
        compute_distance_jacobian(V_,g_,error_, vertices, new_faces, num_faces, num_obs, Sidxij);
    
        // double lambda = 10;
        // for(int i=0;i<num_points; i++)
        // {
        //     int pos;
        //     double* ppUpa;
        //     double sum=0;	
        //     pos = distance_sba_crsm_elmidx( &Sidxij, i, i);

        //     pos += 1 * lambda;
        // }
        constructCSSGN( Si, Sp, Sx, V_, Sidxij, init, v_mask_, num_points); //set CSS format using S matrix
            for(int ii=0; ii< num_points; ii++) {
                Ex[ii] = g_[ii];
            //    std::cout << g_[ii] << std::endl;
            }
            
        solveCholmodGN( Ap, Aii, init, ordering, num_points, nnz);

            

            init = true;
            rx = (double*)m_cholSparseR->x;
            
            if (m_cS.status == CHOLMOD_NOT_POSDEF)
            {
                printf ( "Cholesky failure, writing debug.txt (Hessian loadable by Octave)" );
                exit(1);
            }
            double dx = 0;
            for(int i=0; i < num_points; i++) {
                double update = rx[i];
                double d = vertices[i*3+2];




                d += update;
            


                if( d < 0) {
                    // std::cout << d << std::endl;
                    d = std::abs(d);
                    // std::cout << d << std::endl;

                }
                vertices[i*3+2] = d;

                dx += update * update;

            }
            double cost = 0;
            for(int i=0; i < ( (num_faces*3)); i++) {
                cost += (error_[i] * error_[i]);
            }
            // double er=0;
            // for(int i=0; i < ((num_obs*2)); i++) {
            //     er += error_[i] * error_[i];
            // }
            // er /= num_obs*2;
            ed = 0;
            for(int i=0; i < ((num_faces*3)); i++) {
                ed += error_[i] * error_[i];
            }
            ed /= (num_faces*3);
            
            cost /= ((num_faces*3));
            // std::cout << "Itertation: " << iter <<" Error: " << sqrt(cost) << " dx: " << 0 << " ed: " << ed << std::endl;

            end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = end-start;
            total_time += duration.count();
            // avg += duration;
            // std::cout << "ttttttt\n"; 
            if (verbose_)
                std::cout << "Itertation: " << iter <<" Error: " << sqrt(cost) << " dx: " << dx / num_points << " ed: " << ed << " Time: " << duration.count() << " Total time: " << total_time << std::endl;

            if((cost < 0.000001) || (dx < 0.000001))
                break;
            // if((cost < 0.00000000001) || (dx < 0.00000000001)){
            //     // cv::waitKey(0);
            //     break;
            // }
    }
    // double dd = avg.count() / iter;
    // std::cout << "avg: " << dd << std::endl;


    // if (ed > 100000){
    //     for(int i = 0; i < num_points; i++) {
    //         double psi = vertices[i*3];
    //         double theta = vertices[i*3+1];
    //         double d = vertices[i*3+2];

    //         vertices_[i*3] = std::sin(psi) * std::cos(theta) * d;
    //         vertices_[i*3+1] = std::sin(theta) * d;
    //         vertices_[i*3+2] = std::cos(psi) * std::cos(theta) * d;
            
    //     }

        
    //     for(auto map1 : triangle_unordered_mapping_) {
    //         Eigen::Vector3i e_triangle = e_triangles_[map1.first];
    //         int *triangle = &triangles_[map1.second * 3]; 
        
    //         e_vertices_[e_triangle.x()].x() = vertices_[triangle[0]*3];
    //         e_vertices_[e_triangle.x()].y() = vertices_[triangle[0]*3+1];
    //         e_vertices_[e_triangle.x()].z() = vertices_[triangle[0]*3+2];

    //         e_vertices_[e_triangle.y()].x() = vertices_[triangle[1]*3];
    //         e_vertices_[e_triangle.y()].y() = vertices_[triangle[1]*3+1];
    //         e_vertices_[e_triangle.y()].z() = vertices_[triangle[1]*3+2];

    //         e_vertices_[e_triangle.z()].x() = vertices_[triangle[2]*3];
    //         e_vertices_[e_triangle.z()].y() = vertices_[triangle[2]*3+1];
    //         e_vertices_[e_triangle.z()].z() = vertices_[triangle[2]*3+2];
    //     }
    // }


    for(int i = 0; i < num_points; i++) {
        double psi = vertices[i*3];
        double theta = vertices[i*3+1];
        double d = vertices[i*3+2];

        vertices_[i*3] = std::sin(psi) * std::cos(theta) * d;
        vertices_[i*3+1] = std::sin(theta) * d;
        vertices_[i*3+2] = std::cos(psi) * std::cos(theta) * d;
        
    }

    
    for(auto map1 : triangle_unordered_mapping_) {
        Eigen::Vector3i e_triangle = e_triangles_[map1.first];
        int *triangle = &triangles_[map1.second * 3]; 
    
        e_vertices_[e_triangle.x()].x() = vertices_[triangle[0]*3];
        e_vertices_[e_triangle.x()].y() = vertices_[triangle[0]*3+1];
        e_vertices_[e_triangle.x()].z() = vertices_[triangle[0]*3+2];

        e_vertices_[e_triangle.y()].x() = vertices_[triangle[1]*3];
        e_vertices_[e_triangle.y()].y() = vertices_[triangle[1]*3+1];
        e_vertices_[e_triangle.y()].z() = vertices_[triangle[1]*3+2];

        e_vertices_[e_triangle.z()].x() = vertices_[triangle[2]*3];
        e_vertices_[e_triangle.z()].y() = vertices_[triangle[2]*3+1];
        e_vertices_[e_triangle.z()].z() = vertices_[triangle[2]*3+2];
    }

    // for(int i=0;i<e_vertices_.size();i++) {
    //     if(e_vertices_[i].x() == e_reference_)
    // }
    cholmod_l_gpu_stats(&m_cS);
    distance_sba_crsm_free(&Sidxij);

	cholmod_free_factor(&m_cholFactorS, &m_cS);              
	cholmod_l_free_dense(&m_cholSparseE, &m_cS);
	cholmod_l_free_dense(&m_cholSparseR, &m_cS);
	cholmod_l_finish (&m_cS);
    free(Ap);
	free(Aii);
	cholmod_free_sparse(&m_cholSparseS, &m_cS) ;

    free(v_mask_);
    free(V_);
    free(error_);
    free(vertices_);
    free(reference_);

    // cv::waitKey(0);
    // std::shared_ptr<open3d::geometry::TriangleMesh> mesh = std::make_shared<open3d::geometry::TriangleMesh>();

    // // std::shared_ptr<open3d::geometry::TriangleMesh> mesh;
    // mesh->vertices_ = e_vertices_;
    // mesh->triangles_ = e_triangles_;

    // visualizer(mesh);
    // for(int i=0; i< num_points*3;i++) {
    //     vertices[i] += 1; 
    // }

    
}

void OptimizerDistanceOnly::initialize() {
    
    for(auto& unordered_map: triangle_unordered_mapping_) {
        Eigen::Vector3i triangle = e_triangles_[unordered_map.first];
        triangles_.push_back(vertices_unordered_mapping_[triangle.x()]);
        triangles_.push_back(vertices_unordered_mapping_[triangle.y()]);
        triangles_.push_back(vertices_unordered_mapping_[triangle.z()]);
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // change the vertices to unit vector and depth!

    for(int i=0; i < number_observation_; i++) {
        int face_id = observation_[i*6];
        // if(face_id == id_tmp)
        //     continue;
        // id_tmp = face_id;
        double alpha = observation_[i*6+3];
        double beta = observation_[i*6+4];
        double gamma = observation_[i*6+5];
        // std::cout << "begin\n";
       
        double *vertex, *vertex1;
        if(int(alpha) == 1) {
            vertex = e_vertices_[e_triangles_[face_id].x()].data();
            // vertex = e_reference_[e_triangles_[face_id].x()].data();
            vertex1 = e_reference_[e_triangles_[face_id].x()].data();
            // tmp = e_triangles_[face_id].x();
        } else if (int(beta) == 1)
        {
            vertex = e_vertices_[e_triangles_[face_id].y()].data();
            // vertex = e_reference_[e_triangles_[face_id].y()].data();
            vertex1 = e_reference_[e_triangles_[face_id].y()].data();
            // tmp = e_triangles_[face_id].y();

        } else if (int(gamma) == 1)
        {
            vertex = e_vertices_[e_triangles_[face_id].z()].data();
            // vertex = e_reference_[e_triangles_[face_id].z()].data();
            vertex1 = e_reference_[e_triangles_[face_id].z()].data();
            // tmp = e_triangles_[face_id].z();
    
        } else {
            // std::cout << alpha << " "<< beta << " "<< gamma << "dasd " << std::endl;

            continue;
        }
        // std::cout << alpha << " "<< beta << " "<< gamma << " " << std::endl;
        // std::cout << vertex[0] << " "<< vertex[1] << " "<< vertex[2] << " " << std::endl;
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
        // break;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    reference_ = (double*)malloc(number_vertices_*3*sizeof(double));
    vertices_ = (double*)malloc(number_vertices_*3*sizeof(double));
    memset( reference_, 0, number_vertices_*3*sizeof(double));
    memset( vertices_, 0, number_vertices_*3*sizeof(double));

     for(auto& unordered_map: vertices_unordered_mapping_) {
        Eigen::Vector3d vertex = e_vertices_[unordered_map.first];
        // Eigen::Vector3d vertex = e_reference_[unordered_map.first];
        // Eigen::Vector3d vertex = e_reference_[unordered_map.first];
        Eigen::Vector3d ref_vertex = e_reference_[unordered_map.first];
        // std::cout << num_vertices << std::endl;
        int id = unordered_map.second;
        vertices_[id * 3] = vertex.x();
        vertices_[id * 3 + 1] = vertex.y();
        vertices_[id * 3 + 2] = vertex.z();

        // std::cout << ref_vertex.norm() << " " << vertex.z() << std::endl;

        reference_[id * 3] = ref_vertex.x();
        reference_[id * 3 + 1] = ref_vertex.y();
        reference_[id * 3 + 2] = ref_vertex.z();
    }
    // exit(1);
    


    v_mask_ = (char*)malloc(number_vertices_*number_vertices_*sizeof(char)); // v_mask initialisieren!
    memset( v_mask_, 0, number_vertices_*number_vertices_*sizeof(char) );

    for(int i=0; i < number_triangles_;i++) {
        int f1 = triangles_[i*3];
        int f2 = triangles_[i*3+1];
        int f3 = triangles_[i*3+2];

        v_mask_[f1*number_vertices_+f1] = 1;
        v_mask_[f2*number_vertices_+f2] = 1;
        v_mask_[f3*number_vertices_+f3] = 1;

        if(f1 < f2)
            v_mask_[f1*number_vertices_+f2] = 1;
        else
            v_mask_[f2*number_vertices_+f1] = 1;

        if(f1 < f3)
            v_mask_[f1*number_vertices_+f3] = 1;
        else
            v_mask_[f3*number_vertices_+f1] = 1;

        if(f2 < f3)
            v_mask_[f2*number_vertices_+f3] = 1;
        else
            v_mask_[f3*number_vertices_+f2] = 1;     
    }

    int num_points = number_vertices_;
    int num_obs = number_observation_;
    int num_faces = number_triangles_;
    nnz = 0;


    constructSmask( Sidxij, num_points, nnz, v_mask_); //, Uidxij, v_mask);

    V_	=	(double *)malloc(nnz  * sizeof(double));
    g_	=	(double *)malloc(num_points  * sizeof(double));
    // dx	=	(double *)malloc((num_points*3)*sizeof(double));
    error_ = (double *)malloc(((num_faces * 3))*sizeof(double));
    cholmod_l_start(&m_cS);
    // cholmod_start (&m_cS) ; 
    m_cS.useGPU = 1;
    m_cS.print = 3;
    Ap_  = (long*)malloc((num_points + 1)*sizeof(long));
	Aii_ = (long*)malloc(nnz*sizeof(long));
	constructAuxCSSGN( Ap_, Aii_, v_mask_, num_points );

    m_cholSparseE = cholmod_l_zeros( num_points, 1, CHOLMOD_REAL, &m_cS); // Achtung! Warum sieben?
	Ex_ = (double*)m_cholSparseE->x;
	nMaxS_ = nnz;//(nnz-num_points)*1+num_points*1;	//maximum non-zero element in S matrix 

    m_cholSparseS = cholmod_l_allocate_sparse(num_points,num_points,nMaxS_,true,true,1,CHOLMOD_REAL,&m_cS); 
	
	Sp_ = (long*)m_cholSparseS->p;		//column pointer
	Si_ = (long*)m_cholSparseS->i;		//row pointer

    // int id_tmp = -1;
    // int tmp;
    

    // double *vertex1;
    // vertex1 = e_vertices_[tmp].data();
    // std::cout << vertex1[0] << " "<< vertex1[1] << " "<< vertex1[2] << " test" << std::endl;
    // std::cout << "end\n";
    // exit(1);
    // Sx_ = NULL;  
    // init_ = false;
    // ordering_ = true;
}

void OptimizerDistanceOnly::setParamater(double *observation, std::unordered_map<int,int> &unordered_mapping_vertices, std::unordered_map<int,int> &unordered_mapping_triangles, int number_vertices, int number_triangles, int number_observation) {
    observation_ = observation;
    vertices_unordered_mapping_ = unordered_mapping_vertices;
    triangle_unordered_mapping_ = unordered_mapping_triangles;
    number_vertices_ = number_vertices;
    number_triangles_ = number_triangles;
    number_observation_ = number_observation;
}


void OptimizerDistanceOnly::distance_sba_crsm_alloc(struct distance_sba_crsm *sm, int nr, int nc, int nnz)
{   
	long msz;
	sm->nr=nr;
	sm->nc=nc;
	sm->nnz=nnz;
	msz=2*nnz+nr+1;
	sm->val=(long *)malloc(msz*sizeof(long));  /* required memory is allocated in a single step */
	if(!sm->val){
		fprintf(stderr, "memory allocation request failed in distance_sba_crsm_alloc() [nr=%d, nc=%d, nnz=%d]\n", nr, nc, nnz);
		exit(1);
	}
	sm->colidx=sm->val+nnz;
	sm->rowptr=sm->colidx+nnz;
}

void OptimizerDistanceOnly::distance_sba_crsm_free(struct distance_sba_crsm *sm)
{
	 sm->nr=sm->nc=sm->nnz=-1;
	free(sm->val);
	sm->val=sm->colidx=sm->rowptr=NULL;
}

/* returns the index of the (i, j) element. No bounds checking! */ // from PBA
int OptimizerDistanceOnly::distance_sba_crsm_elmidx(struct distance_sba_crsm *sm, int i, int j)
{
	int low, high, mid, diff;

	low=sm->rowptr[i];
	high=sm->rowptr[i+1]-1;

	/* binary search for finding the element at column j */
	while(low<=high)
	{
		mid=(low+high)>>1; //(low+high)/2;
		diff=j-sm->colidx[mid];
		if(diff<0)
			 high=mid-1;
		else if(diff>0)
			low=mid+1;
		else
		return mid;
	}

	return -1; /* not found */
}

void OptimizerDistanceOnly::constructSmask( distance_sba_crsm& Sidxij, int m, int& m_nS, char* m_smask)//, distance_sba_crsm& Uidxij, char* m_umask)
{
	int i, j, k, ii, jj;
    for ( i = 0; i < m*m; i++ )
	{
		if ( (m_smask[i] == 1) )
		{
            
			m_smask[i] = 1;
			m_nS += 1;
		}
	}	
	distance_sba_crsm_alloc(&Sidxij, m, m, m_nS);
	for(i=k=0; i<m; ++i)
	{
		Sidxij.rowptr[i]=k;
		ii=i*m;
		for(j=0; j<m; ++j)
			if(m_smask[ii+j])
			{
				Sidxij.val[k]=k;
				Sidxij.colidx[k++]=j;
			}
	}
	Sidxij.rowptr[m]=m_nS;
    

}





void OptimizerDistanceOnly::storeInG_distance(double *g, double J, double *e) {
    g[0] += -J*e[0];
    // g[1] += -J[1]*e[0];
    // g[2] += -J[2]*e[0];

}

void OptimizerDistanceOnly::storeInV_distance(double* V, int idx1, int idx2, double J1, double J2, distance_sba_crsm& Uidxij) {
    int pos;
    double* ppUpa;
    double JJ1, JJ2;
    double sum=0;	
    if (idx1 < idx2) {
        JJ1 = J1;
        JJ2 = J2;
        pos = distance_sba_crsm_elmidx( &Uidxij, idx1, idx2);
    } else {
        JJ1 = J2;
        JJ2 = J1;
        pos = distance_sba_crsm_elmidx( &Uidxij, idx2, idx1); 
    }

    if(pos == -1) {
        std::cerr << "Pos -1 in StoreInV" << std::endl; 
    }
    ppUpa = V + pos;
    ppUpa[0] += JJ1 * JJ2;
    // ppUpa[1] += JJ1[0] * JJ2[1];
    // ppUpa[2] += JJ1[0] * JJ2[2];
    // ppUpa[3] += JJ1[1] * JJ2[0];
    // ppUpa[4] += JJ1[1] * JJ2[1];
    // ppUpa[5] += JJ1[1] * JJ2[2];
    // ppUpa[6] += JJ1[2] * JJ2[0];
    // ppUpa[7] += JJ1[2] * JJ2[1];
    // ppUpa[8] += JJ1[2] * JJ2[2];
    
}

void OptimizerDistanceOnly::storeInV_distance(double* V, int idx, double J, distance_sba_crsm& Uidxij) {
    int pos;
    double* ppUpa;
    double sum=0;	
    pos = distance_sba_crsm_elmidx( &Uidxij, idx, idx);

    if(pos == -1) {
        std::cerr << "Pos -1 in StoreInV" << std::endl; 
    }
    ppUpa = V + pos;
    ppUpa[0] += J * J;
    // ppUpa[1] += J[0] * J[1];
    // ppUpa[2] += J[0] * J[2];

    // ppUpa[4] += J[1] * J[1];
    // ppUpa[5] += J[1] * J[2];

    // ppUpa[8] += J[2] * J[2];
}

void OptimizerDistanceOnly::compute_distance_jacobian(double *V, double *g, double *error, double *xyz, int *faces, int num_faces, int offset, distance_sba_crsm& Uidxij) {
    int counter = 0;
    for(int i=0;i < num_faces; i++) {
        
        int f1 = faces[i*3];
        int f2 = faces[i*3+1];
        int f3 = faces[i*3+2];

        Eigen::Vector3d v1, v2, v3, v12, v13, v23, u1,u2,u3;
        v1 << std::sin(xyz[f1*3]) * std::cos(xyz[f1*3+1]) * xyz[f1*3+2], std::sin(xyz[f1*3+1]) * xyz[f1*3+2], std::cos(xyz[f1*3]) * std::cos(xyz[f1*3+1]) * xyz[f1*3+2];
        v2 << std::sin(xyz[f2*3]) * std::cos(xyz[f2*3+1]) * xyz[f2*3+2], std::sin(xyz[f2*3+1]) * xyz[f2*3+2], std::cos(xyz[f2*3]) * std::cos(xyz[f2*3+1]) * xyz[f2*3+2];
        v3 << std::sin(xyz[f3*3]) * std::cos(xyz[f3*3+1]) * xyz[f3*3+2], std::sin(xyz[f3*3+1]) * xyz[f3*3+2], std::cos(xyz[f3*3]) * std::cos(xyz[f3*3+1]) * xyz[f3*3+2];


        u1 << std::sin(xyz[f1*3]) * std::cos(xyz[f1*3+1]), std::sin(xyz[f1*3+1]), std::cos(xyz[f1*3]) * std::cos(xyz[f1*3+1]);
        u2 << std::sin(xyz[f2*3]) * std::cos(xyz[f2*3+1]), std::sin(xyz[f2*3+1]), std::cos(xyz[f2*3]) * std::cos(xyz[f2*3+1]);
        u3 << std::sin(xyz[f3*3]) * std::cos(xyz[f3*3+1]), std::sin(xyz[f3*3+1]), std::cos(xyz[f3*3]) * std::cos(xyz[f3*3+1]);



        double d12, d13, d23;
        v12 = v1-v2;
        d12 = v12.norm();
        v13 = v1-v3;
        d13 = v13.norm();
        v23 = v2 - v3;
        d23 = v23.norm();

        double J12_1, J12_2, J13_1, J13_3, J23_2, J23_3;

        J12_1 = u1[0]*v12[0] / d12 + u1[1]*v12[1] / d12 + u1[2]*v12[2] / d12;
        J13_1 = u1[0]*v13[0] / d13 + u1[1]*v13[1] / d13 + u1[2]*v13[2] / d13;

        J12_2 = -u2[0]*v12[0] / d12 + (-u2[1]*v12[1] / d12) + (-u2[2]*v12[2] / d12);
        J23_2 = u2[0]*v23[0] / d23 + u2[1]*v23[1] / d23 + u2[2]*v23[2] / d23;

        J13_3 = -u3[0]*v13[0] / d13 + (-u3[1]*v13[1] / d13) + (-u3[2]*v13[2] / d13);
        J23_3 = -u3[0]*v23[0] / d23 + (-u3[1]*v23[1] / d23) + (-u3[2]*v23[2] / d23);



        // J12_1[0] = v12[0] / (d12);
        // J12_1[1] = v12[1] / (d12);
        // J12_1[2] = v12[2] / (d12);

        // J13_1[0] = v13[0] / (d13);
        // J13_1[1] = v13[1] / (d13);
        // J13_1[2] = v13[2] / (d13);

        // J23_2[0] = v23[0] / (d23);
        // J23_2[1] = v23[1] / (d23);
        // J23_2[2] = v23[2] / (d23);

        // J12_2[0] = -v12[0] / (d12);
        // J12_2[1] = -v12[1] / (d12);
        // J12_2[2] = -v12[2] / (d12);

        // J13_3[0] = -v13[0] / (d13);
        // J13_3[1] = -v13[1] / (d13);
        // J13_3[2] = -v13[2] / (d13);
        
        // J23_3[0] = -v23[0] / (d23);
        // J23_3[1] = -v23[1] / (d23);
        // J23_3[2] = -v23[2] / (d23);

        storeInG_distance(g+f1, J12_1, error + counter);
        storeInG_distance(g+f2, J12_2, error + counter);
        counter++;
        storeInG_distance(g+f1, J13_1, error  + counter);
        storeInG_distance(g+f3, J13_3, error  + counter);
        counter++;
        storeInG_distance(g+f2, J23_2, error  + counter);
        storeInG_distance(g+f3, J23_3, error  + counter);
        counter++;
        // storeInG(g+f1*3, J1, error+obs_id);

        storeInV_distance(V, f1, J12_1, Uidxij);
        storeInV_distance(V, f2, J12_2, Uidxij);

        storeInV_distance(V, f1, J13_1, Uidxij);
        storeInV_distance(V, f3, J13_3, Uidxij);

        storeInV_distance(V, f2, J23_2, Uidxij);
        storeInV_distance(V, f3, J23_3, Uidxij);

        storeInV_distance(V, f1, f2, J12_1, J12_2, Uidxij);
        storeInV_distance(V, f1, f3, J13_1, J13_3, Uidxij);
        storeInV_distance(V, f2, f3, J23_2, J23_3, Uidxij);
    }
    // std::cout << "dsad";
}


void OptimizerDistanceOnly::compute_distance_error(double *error, double *xyz, double *ref,int *faces, int number_faces, int offset) {
    int counter = 0;
    for(int i=0;i<number_faces; i++) {
        
        int f1 = faces[i*3];
        int f2 = faces[i*3+1];
        int f3 = faces[i*3+2];

        Eigen::Vector3d v1, v2, v3, v1_ref, v2_ref, v3_ref;
        // std::cout << "\n\n\n" << xyz[f1*3] << " " << xyz[f1*3+1] << " " << xyz[f1*3+2] << std::endl;
        // std::cout << "\n\n\n" << ref[f1*3] << " " << ref[f1*3+1] << " " << ref[f1*3+2] << std::endl;
        // exit(12);
        v1 << std::sin(xyz[f1*3]) * std::cos(xyz[f1*3+1]) * xyz[f1*3+2], std::sin(xyz[f1*3+1]) * xyz[f1*3+2], std::cos(xyz[f1*3]) * std::cos(xyz[f1*3+1]) * xyz[f1*3+2];
        v2 << std::sin(xyz[f2*3]) * std::cos(xyz[f2*3+1]) * xyz[f2*3+2], std::sin(xyz[f2*3+1]) * xyz[f2*3+2], std::cos(xyz[f2*3]) * std::cos(xyz[f2*3+1]) * xyz[f2*3+2];
        v3 << std::sin(xyz[f3*3]) * std::cos(xyz[f3*3+1]) * xyz[f3*3+2], std::sin(xyz[f3*3+1]) * xyz[f3*3+2], std::cos(xyz[f3*3]) * std::cos(xyz[f3*3+1]) * xyz[f3*3+2];

        v1_ref << ref[f1*3], ref[f1*3+1], ref[f1*3+2];
        v2_ref << ref[f2*3], ref[f2*3+1], ref[f2*3+2];
        v3_ref << ref[f3*3], ref[f3*3+1], ref[f3*3+2];


        double d12, d12_ref, d13, d13_ref, d23, d23_ref;
        d12 = (v1-v2).norm();
        d12_ref = (v1_ref-v2_ref).norm();

        d13 = (v1-v3).norm();
        d13_ref = (v1_ref-v3_ref).norm();

        d23 = (v2-v3).norm();
        d23_ref = (v2_ref-v3_ref).norm();
        double e1 = d12 - d12_ref;
        double e2 = d13 - d13_ref;
        double e3 = d23 - d23_ref;
        error[counter] = e1;
        counter++;
        error[counter] = e2;
        counter++;
        error[counter] = e3;
        counter++;
    }
}



void OptimizerDistanceOnly::constructAuxCSSGN( long *Ap, long *Aii, char* m_smask, int m_ncams)
{
	long* Cp = Ap;
	long* Ci = Aii;
	int ii, jj;
	int m = m_ncams, nZ = 0;
	for ( ii = 0; ii < m; ii++ ) 
	{
		*Cp = nZ;
		for( jj=0; jj<=ii; jj++ )
		{
			if (m_smask[jj*m+ii]==1)
			{
				*Ci++ = jj;
				nZ++;
			}
		}
		Cp++;
	}
	*Cp=nZ;
}

void OptimizerDistanceOnly::constructCSSGN( long* Si, long* Sp, double* Sx, double* S, distance_sba_crsm& Sidxij, bool init, char* m_smask, int m_ncams)
{
	int ii, jj, jjj, k;
	int pos1, m = m_ncams;
	//Copy S matrix and E matrix to specific format structure for Cholmod 
	double *ptr5;
	int nZ = 0;
	Sx = (double*)m_cholSparseS->x;

	if ( !init)
	{
		for ( ii = 0; ii < m; ii++ )  //column
		{
			for ( k = 0; k < 1; k++ )
			{
				*Sp = nZ;
				// if ((ii*6+k)==(9+nft))
				// 	continue;

				for ( jj = 0; jj <= ii; jj++ )	//row
				{   
                    // std::cout << (int)m_smask[ii+jj*m] << "\t";
					if ((m_smask[jj*m+ii]==1))
					{   

						pos1 = distance_sba_crsm_elmidx( &Sidxij, jj, ii );
						ptr5 = S + pos1*1;
						
						if( ii == jj )
						{
							for ( jjj = 0; jjj <= k; jjj++)
							{
								// if ( (jj*6+jjj) != (9+nft))
								// {
									// if ( jj*6+jjj < 9+nft)
									// 	*Si++ = jj*6+jjj - 6;
									// else
									// 	*Si++ = jj*6+jjj - 7;
                                    *Si++ = jj*1+jjj;
									*Sx++ = ptr5[jjj*1+k];
									nZ++;
                                    // std::cout << ptr5[jjj*6+k] << std::endl;
								// }						
							}
                            // exit(1);
						}
						else
						{
							for ( jjj = 0; jjj < 1; jjj++)
							{
								// if ((jj*6+jjj) != (9+nft) )
								// {
									// if ( jj*6+jjj < 9+nft )
									// 	*Si++ = jj*6+jjj - 6;
									// else
									// 	*Si++ = jj*6+jjj - 7;										

                                    *Si++ = jj*1+jjj;
									*Sx++ = ptr5[jjj*1+k];
									nZ++;
								// }
							}
						}
					}
				} 
                // std::cout << "\n";
				Sp++;
			}
		}
		*Sp=nZ;
	}
	else
	{
        // std::cout << "else" << std::endl;
		for ( ii = 0; ii < m; ii++ )  //column
		{
			for ( k = 0; k < 1; k++ )
			{
				// if ((ii*6+k)==(9+nft))
				// 	continue;

				for ( jj = 0; jj <= ii; jj++ )	//row
				{
					if ((m_smask[jj*m+ii]==1))
					{
						pos1 = distance_sba_crsm_elmidx( &Sidxij, jj, ii );
						ptr5 = S + pos1*1;

						if( ii == jj )
						{
							for ( jjj = 0; jjj <= k; jjj++)
							{
								// if ( (jj*6+jjj) != (9+nft))
									*Sx++ = ptr5[jjj*1+k];
							}
						}
						else
						{
							for ( jjj = 0; jjj < 1; jjj++)
							{
								// if ((jj*6+jjj) != (9+nft) )
									*Sx++ = ptr5[jjj*1+k];
							}
						}
					}
				}
			}
		}
	}
}

bool OptimizerDistanceOnly::solveCholmodGN( long* Ap1, long* Aii1, bool init, bool ordering, int m_ncams, int nnz1)
{
	int i, j;
	int m = m_ncams;
	Eigen::Matrix<long int, Eigen::Dynamic, 1> scalarPermutation, blockPermutation;

	ordering = true;
	if (!init)
	{
		if (!ordering)
		{
			m_cS.nmethods = 1;
			m_cS.method[0].ordering = CHOLMOD_AMD; //CHOLMOD_COLAMD
			m_cholFactorS = cholmod_l_analyze(m_cholSparseS, &m_cS); // symbolic factorization
		}
		else
		{

			// get the ordering for the block matrix
			if (blockPermutation.size() == 0)
				blockPermutation.resize((long)m_ncams);
            
			// prepare AMD call via CHOLMOD
			cholmod_sparse auxCholmodSparse;
			auxCholmodSparse.nzmax = nnz1;
			auxCholmodSparse.nrow = auxCholmodSparse.ncol = m;
			auxCholmodSparse.p = Ap1;
			auxCholmodSparse.i = Aii1;
			auxCholmodSparse.nz = 0;
			auxCholmodSparse.x = nullptr;
			auxCholmodSparse.z = nullptr;
			auxCholmodSparse.stype = 1;
			auxCholmodSparse.xtype = CHOLMOD_PATTERN;
			auxCholmodSparse.itype = CHOLMOD_LONG;
			auxCholmodSparse.dtype = CHOLMOD_DOUBLE;
			auxCholmodSparse.sorted = 1;
			auxCholmodSparse.packed = 1;
            // std::cout << "test" <<std::endl;
            
			int amdStatus = cholmod_l_amd(&auxCholmodSparse, nullptr, 0, blockPermutation.data(), &m_cS);
			if (! amdStatus) {
				return false;
			}
            // std::cout << scalarPermutation.size() << " " << m_cholSparseS->ncol << std::endl;
			// blow up the permutation to the scalar matrix
			if (scalarPermutation.size() == 0)
				scalarPermutation.resize(m_cholSparseS->ncol);
			size_t scalarIdx = 0;
			int a = 0;
            
			for ( i = 0; i < m_ncams; ++i)
			{
				const int &pp = blockPermutation(i);
				int base = pp*1;
                // std::cout << pp << std::endl;
				// int nCols= (pp==0) ? 6 : 6;

                // int base =  pp*6-1;

				int nCols= 1;

				for ( j = 0; j < nCols; ++j)
					scalarPermutation(scalarIdx++) = base++;

			}
            // std::cout << scalarIdx << " " << m_cholSparseS->ncol << std::endl;

			assert(scalarIdx == m_cholSparseS->ncol);

			// apply the ordering
			m_cS.nmethods = 1 ;
			m_cS.method[0].ordering = CHOLMOD_GIVEN;
            // std::cout << scalarPermutation << std::endl;

			m_cholFactorS = cholmod_l_analyze_p(m_cholSparseS, scalarPermutation.data(), NULL, 0, &m_cS);
		}
        
		init = true;
	}
// std::cout << "test" << std::endl;
		
	//Cholmod package for solving sparse linear equation              
	cholmod_l_factorize(m_cholSparseS, m_cholFactorS, &m_cS); 
	m_cholSparseR = cholmod_l_solve (CHOLMOD_A, m_cholFactorS, m_cholSparseE, &m_cS) ;

// std::cout << "test" << std::endl;
// exit(1);

	return true;
}
}