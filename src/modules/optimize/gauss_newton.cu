#include "gauss_newton.cuh"


__global__ void compute_AWB(double *info_matrix, double *d, int *triangle, int *num_triangle, int *numVertices, double *AWB) {
    int global_thread = blockDim.x * blockIdx.x + threadIdx.x;

    int triangle_id = global_thread / 3;

    if (triangle_id >= *num_triangle)
        return;

    int tid1 = global_thread % 3;
    int tid2 = (tid1 + 1) % 3;
    int tid3 = (tid1 + 2) % 3;

    // for(int i=0; i < *num_triangle; i++) {

        int tri1 = triangle[triangle_id * 3 + tid1];    
        int tri2 = triangle[triangle_id * 3 + tid2];    
        int tri3 = triangle[triangle_id * 3 + tid3];

        double tid1_val = info_matrix[triangle_id * 9 + tid1 * 3 + tid1];
        double tid2_val = info_matrix[triangle_id * 9 + tid1 * 3 + tid2];
        double tid3_val = info_matrix[triangle_id * 9 + tid1 * 3 + tid3];


        double d1 = d[triangle_id * 3 + tid1];
        double d2 = d[triangle_id * 3 + tid2];
        double d3 = d[triangle_id * 3 + tid3];

    atomicAdd_double(&AWB[tri1], (tid1_val * d1) + (tid2_val * d2) + (tid3_val * d3));

}

__global__ void compute_AWA(double *info_matrix, int *triangle, int *num_triangle, int *numVertices, double *AWA) {

     int global_thread = blockDim.x * blockIdx.x + threadIdx.x;

    int triangle_id = global_thread / 3;

    if (triangle_id >= *num_triangle)
        return;

    int tid1 = global_thread % 3;
    int tid2 = (tid1 + 1) % 3;
    int tid3 = (tid1 + 2) % 3;

    int tri1 = triangle[triangle_id * 3 + tid1];    
    int tri2 = triangle[triangle_id * 3 + tid2];    
    int tri3 = triangle[triangle_id * 3 + tid3];

    double tid1_val = info_matrix[triangle_id * 9 + tid1 * 3 + tid1];
    double tid2_val = info_matrix[triangle_id * 9 + tid1 * 3 + tid2];
    double tid3_val = info_matrix[triangle_id * 9 + tid1 * 3 + tid3];


        atomicAdd_double(&AWA[tri1 * *numVertices + tri1], tid1_val);
        atomicAdd_double(&AWA[tri1 * *numVertices + tri2], tid2_val);
        // AWA[tri1 * *numVertices + tri3] += tid3_val;
        atomicAdd_double(&AWA[tri2 * *numVertices + tri1], tid2_val);
        // AWA[tri3 * *numVertices + tri1] += tid3_val;
     
}

__global__ void compute_cost_new(double *d_bar, int *triangle, int *num_triangle, double *d, double *vertices, double *cost) {
    int global_thread = blockDim.x * blockIdx.x + threadIdx.x;

    int triangle_id = global_thread;

    if(triangle_id < *num_triangle) {

       

        int tri1 = triangle[triangle_id * 3 + 0];
        int tri2 = triangle[triangle_id * 3 + 1];
        int tri3 = triangle[triangle_id * 3 + 2];

        
        double dbar1 = d_bar[triangle_id * 3 + 0];
        double dbar2 = d_bar[triangle_id * 3 + 1];
        double dbar3 = d_bar[triangle_id * 3 + 2];

        double d1 = d[triangle_id*3 + 0];
        double d2 = d[triangle_id*3 + 1];
        double d3 = d[triangle_id*3 + 2];

        double *u1 = vertices + tri1 * 3; 
        double *u2 = vertices + tri2 * 3; 
        double *u3 = vertices + tri3 * 3; 


        double v12x = u1[0] * d1 - u2[0] * d2;
        double v12y = u1[1] * d1 - u2[1] * d2;
        double v12z = u1[2] * d1 - u2[2] * d2;

        double v23x = u2[0] * d2 - u3[0] * d3;
        double v23y = u2[1] * d2 - u3[1] * d3;
        double v23z = u2[2] * d2 - u3[2] * d3;

        double v31x = u3[0] * d3 - u1[0] * d1;
        double v31y = u3[1] * d3 - u1[1] * d1;
        double v31z = u3[2] * d3 - u1[2] * d1;
        

        double e12 = dbar1 - sqrt(v12x*v12x + v12y*v12y + v12z*v12z);
        double e23 = dbar2 - sqrt(v23x*v23x + v23y*v23y + v23z*v23z);
        double e31 = dbar3 - sqrt(v31x*v31x + v31y*v31y + v31z*v31z);

        cost[triangle_id * 20 + 0] = sqrt(e12 * e12 + e23 * e23 + e31 * e31);
    

    }

}

__global__ void compute_d(double *d, double *ref, int *num_triangle, int *triangles) {

    int total = blockDim.x * blockIdx.x + threadIdx.x;

    int triangle_id = total / 3;
    int tid = total % 3;

    if (triangle_id >= *num_triangle)
        return;

    
    int tri = triangles[triangle_id * 3 + tid];

    double *vertex = ref + tri * 3; 

    d[triangle_id * 3 + tid]  = sqrt(vertex[0] * vertex[0] + vertex[1] * vertex[1] + vertex[2] * vertex[2]);

}

__global__ void update_dx(double *d, int *num_triangles, double *adjugate_matrix, double *determinante, double *g) {
    int global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;

    int tid = global_thread_id % 3;
    int triangle_id = global_thread_id / 3;

    if(triangle_id >= *num_triangles) 
        return;

    double det = determinante[triangle_id * 6];

    double dx = ((adjugate_matrix[triangle_id * 9 + tid * 3 + 0] * g[triangle_id * 3] ) +
                 (adjugate_matrix[triangle_id * 9 + tid * 3 + 1] * g[triangle_id * 3 + 1] ) +
                 (adjugate_matrix[triangle_id * 9 + tid * 3 + 2] * g[triangle_id * 3 + 2] )) / det;

    atomicAdd_double(&d[triangle_id*3 + tid], dx);

}

__global__ void compute_adjugate_new(double *H, int *num_triangles, double *adjugate_matrix) {
    int global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;

    int tid = global_thread_id % 9;
    int triangle_id = global_thread_id / 9;

    if (triangle_id >= *num_triangles)
        return;

    

    int row = tid / 3;
    int col = tid % 3;

    int m = col % 2;
    int m1 = row % 2;
    int m2 = -1+2*((tid+1)%2);

    int left = (col + 1 + 1*m) % 3;
    int right = (col + 2 + 2*m) % 3;

    int top = ((row+1+1*m1)*3) % 9;
    int bot = ((row+2+2*m1)*3) % 9;

    int a1 = top + left;
    int a2 = bot + left;
    
    int b1 = top + right;
    int b2 = bot + right;

    int start_address = triangle_id * 9;
    // adjugate_matrix[triangle_id*9+tid] = a1;

    atomicAdd_double(&adjugate_matrix[triangle_id*9 + col * 3 + row], (H[start_address +a1]*H[start_address + b2] - H[start_address + b1] * H[start_address + a2]) * m2);
}


__global__ void compute_determinante_new(double *H, int *num_triangles, double *determinante) {
    int global_thread_id = blockDim.x*blockIdx.x+threadIdx.x;

    int triangle_id = global_thread_id / 6;
    int tid;
    if(triangle_id < *num_triangles) {
        tid = global_thread_id % 6;

        int m1 = tid / 3;

        int first = ((tid + 1*m1) + (tid + 1*m1)*m1) % 3;
        int second = 3 + ((tid + 1 + 1*m1) + (tid + 1 + 1*m1)*m1)%3;
        int third  = 6 + ((tid + 2 + 1*m1) + (tid + 2 + 1*m1)*m1)%3;

        atomicAdd_double(&determinante[triangle_id * 6 + tid], H[triangle_id * 9 + first] * H[triangle_id * 9 + second] * H[triangle_id * 9 + third]);
        // determinante[triangle_id * 6 + tid] = third;
    }

    __syncthreads();

    if ((triangle_id >= *num_triangles) || (tid >=1))
        return;
    
    // determinante[triangle_id * 6] = triangle_id;
    atomicAdd_double(&determinante[triangle_id * 6], determinante[triangle_id * 6 + 0] + determinante[triangle_id * 6 + 1] + determinante[triangle_id * 6 + 2] - 
                                    determinante[triangle_id * 6 + 3] - determinante[triangle_id * 6 + 4] - determinante[triangle_id * 6 + 5]);

}



__global__ void compute_constantUnits(double *c_unit, int *triangles, int *num_triangles, double *vertices) {

    int total_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    
    int triangle_id = total_thread_id / 3;
    if (triangle_id >= *num_triangles)
        return;

    int tid = total_thread_id % 3;
    int tid_next_id = (tid + 1) % 3;

    int triangle_face_id1 = triangles[triangle_id*3 + tid];
    int triangle_face_id2 = triangles[triangle_id*3 + tid_next_id]; 

    double *u1 = vertices + triangle_face_id1*3;
    double *u2 = vertices + triangle_face_id2*3;

    double value = u1[0] * u2[0] + u1[1] * u2[1] + u1[2] * u2[2];

    c_unit[triangle_id*3 + tid] = value;
}

__global__ void obs2unitvector(double* obs, double* vertices, double* K, int* number_obs) {
    
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid > *number_obs)
        return;

    int obs_id = tid;

    double uvt[3];
    uvt[0] = (obs[obs_id*2] - K[1])/K[0];
    uvt[1] = (obs[obs_id*2+1] - K[3])/K[2];

    double d = sqrt(uvt[0]*uvt[0]+uvt[1]*uvt[1]+1); // get the distance to compute the unit vector!

    uvt[0] /= d;
    uvt[1] /= d;
    uvt[2] = 1 / d;

    vertices[obs_id*3]   = uvt[0];
    vertices[obs_id*3+1] = uvt[1];
    vertices[obs_id*3+2] = uvt[2];
}


__global__ void computeH_new(double *H, int *triangles, double* vertices, double *d, double *constant_unit, int *num_triangles) {

    int total_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    
    int triangle_id = total_thread_id / 3;

    double J2J2;
    int tid_next_id ;
    if (triangle_id < *num_triangles) {
        

        int tid = total_thread_id % 3;
        tid_next_id = (tid + 1) % 3;

        int triangle_face_id1 = triangles[triangle_id*3 + tid];
        int triangle_face_id2 = triangles[triangle_id*3 + tid_next_id]; 

        double d1 = d[triangle_id * 3 + tid];
        double d2 = d[triangle_id * 3 + tid_next_id];

        double c_unit = constant_unit[triangle_id*3 + tid];

        double J1_part = d1 - c_unit * d2;
        double J2_part = d2 - c_unit * d1;

        double x = (vertices[triangle_face_id1*3+0] * d1 - vertices[triangle_face_id2*3+0] * d2);
        double y = (vertices[triangle_face_id1*3+1] * d1 - vertices[triangle_face_id2*3+1] * d2);
        double z = (vertices[triangle_face_id1*3+2] * d1 - vertices[triangle_face_id2*3+2] * d2);
        

        double dnorm =  sqrt(x*x + y*y + z*z);

        double J1 = J1_part/ dnorm;
        double J2 = J2_part/ dnorm;
        double J1J2 = J1 * J2;
        J2J2 = J2*J2;
        
        // safe in H matrix on diagonal
        atomicAdd_double(&H[triangle_id*9 + tid*3+tid], J1*J1);
        // safe on non diagonal
        // atomicAdd_double(&H[triangle_id*9 + tid*3+tid_next_id], J1J2);   
        // // safe on a placeholder field
        // atomicAdd_double(&H[triangle_id*9 + tid_next_id*3+tid], J1J2);   

        H[triangle_id*9 + tid*3+tid_next_id] =  J1J2;   
        // safe on a placeholder field
        H[triangle_id*9 + tid_next_id*3+tid] =  J1J2;   
    } 

    __syncthreads();

    if(triangle_id >=  *num_triangles) 
        return;

    atomicAdd_double(&H[triangle_id*9 + tid_next_id*3+tid_next_id], J2J2);
    

}


__global__ void computeG_new(double *g, int *triangles, double * vertices, double *d, double *d_bar, double *constant_unit, int *num_triangles) {
    int total_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    
    int triangle_id = total_thread_id / 3;
    double g2;
    int tid_next_id;

    if(triangle_id <  *num_triangles){

        int tid = total_thread_id % 3;
        tid_next_id = (tid + 1) % 3;

        int triangle_face_id1 = triangles[triangle_id*3 + tid];
        int triangle_face_id2 = triangles[triangle_id*3 + tid_next_id]; 

        double d1 = d[triangle_id * 3 + tid];
        double d2 = d[triangle_id * 3 + tid_next_id];

        double c_unit = constant_unit[triangle_id*3 + tid];

        double d_bar12 = d_bar[triangle_id*3+tid];

        double x = (vertices[triangle_face_id1*3+0] * d1 - vertices[triangle_face_id2*3+0] * d2);
        double y = (vertices[triangle_face_id1*3+1] * d1 - vertices[triangle_face_id2*3+1] * d2);
        double z = (vertices[triangle_face_id1*3+2] * d1 - vertices[triangle_face_id2*3+2] * d2);
        
        double dnorm =  sqrt(x*x + y*y + z*z);

        double left_part = d_bar12 / dnorm - 1 ;

        double g1= left_part*(d1-c_unit*d2);
               g2= left_part*(d2-c_unit*d1);

        atomicAdd_double(&g[triangle_id * 3 + tid], g1);
        // g[triangle_id * 3 + tid] = 0;
    }
    __syncthreads();
    
    if(triangle_id >=  *num_triangles) 
        return;

    atomicAdd_double(&g[triangle_id * 3 + tid_next_id], g2);
}


__global__ void compute_d_bar_x(int* numTriangles, int *triangles, double *reference, double *d_bar) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if(tid >=  *numTriangles)
        return;

    int triangleId = tid;
    int triangle[3];
    triangle[0] = triangles[triangleId*3];
    triangle[1] = triangles[triangleId*3+1];
    triangle[2] = triangles[triangleId*3+2];

    double v1[3];
    double v2[3];
    double tmp[3];

    v1[0] = reference[triangle[0]*3];
    v1[1] = reference[triangle[0]*3+1];
    v1[2] = reference[triangle[0]*3+2];

    v2[0] = reference[triangle[1]*3];
    v2[1] = reference[triangle[1]*3+1];
    v2[2] = reference[triangle[1]*3+2];
    
    tmp[0] = v1[0] - v2[0];
    tmp[1] = v1[1] - v2[1];
    tmp[2] = v1[2] - v2[2];
    d_bar[triangleId*3] = sqrt(tmp[0]*tmp[0] + tmp[1]*tmp[1] + tmp[2]*tmp[2]); 

}

__global__ void compute_d_bar_y(int* numTriangles, int *triangles, double *reference, double *d_bar) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if(tid >=  *numTriangles)
        return;

    int triangleId = tid;
    int triangle[3];
    triangle[0] = triangles[triangleId*3];
    triangle[1] = triangles[triangleId*3+1];
    triangle[2] = triangles[triangleId*3+2];

    double v1[3];
    double v2[3];
    double tmp[3];

    v1[0] = reference[triangle[1]*3];
    v1[1] = reference[triangle[1]*3+1];
    v1[2] = reference[triangle[1]*3+2];

    v2[0] = reference[triangle[2]*3];
    v2[1] = reference[triangle[2]*3+1];
    v2[2] = reference[triangle[2]*3+2];
    
    tmp[0] = v1[0] - v2[0];
    tmp[1] = v1[1] - v2[1];
    tmp[2] = v1[2] - v2[2];
    d_bar[triangleId*3+1] = sqrt(tmp[0]*tmp[0] + tmp[1]*tmp[1] + tmp[2]*tmp[2]); 

}

__global__ void compute_d_bar_z(int* numTriangles, int *triangles, double *reference, double *d_bar) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if(tid >=  *numTriangles)
        return;

    int triangleId = tid;
    int triangle[3];
    triangle[0] = triangles[triangleId*3];
    triangle[1] = triangles[triangleId*3+1];
    triangle[2] = triangles[triangleId*3+2];

    double v1[3];
    double v2[3];
    double tmp[3];

    v1[0] = reference[triangle[2]*3];
    v1[1] = reference[triangle[2]*3+1];
    v1[2] = reference[triangle[2]*3+2];

    v2[0] = reference[triangle[0]*3];
    v2[1] = reference[triangle[0]*3+1];
    v2[2] = reference[triangle[0]*3+2];
    
    tmp[0] = v1[0] - v2[0];
    tmp[1] = v1[1] - v2[1];
    tmp[2] = v1[2] - v2[2];
    d_bar[triangleId*3+2] = sqrt(tmp[0]*tmp[0] + tmp[1]*tmp[1] + tmp[2]*tmp[2]); 

}


__device__ double atomicAdd_double(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*) address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);

    return __longlong_as_double(old);
}