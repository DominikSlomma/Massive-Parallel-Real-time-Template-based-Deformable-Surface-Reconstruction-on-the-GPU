#ifndef GAUSS_NEWTON_H
#define GAUSS_NEWTON_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cusolverSp.h>
// #include <math_functions.h>
#include <cuda_runtime_api.h>


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

__global__ void update_dx(double *d, int *num_triangles, double *adjugate_matrix, double *determinante, double *g);

__global__ void compute_d(double *d, double *ref, int *num_tria, int *triangles);

__global__ void compute_cost_new(double *d_bar, int *triangle, int *num_triangle, double *d, double *vertices, double *cost);


__global__ void compute_AWA(double *info_matrix, int *triangle, int *num_triangle, int *numVertices, double *AWA);

__global__ void compute_AWB(double *info_matrix, double *d, int *triangle, int *num_triangle, int *numVertices, double *AWB);


__device__ double atomicAdd_double(double* address, double val);


#endif