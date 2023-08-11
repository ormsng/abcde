#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <math.h>
#include "Proto.h"
#include "Structs.h"

__global__ void calculatePoint(Axis *axisArr, Point *pointArr, int numElements, double t)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        pointArr[i].x = ((axisArr[i].x2 - axisArr[i].x1) / 2) * sin(t * PI / 2) + (axisArr[i].x2 + axisArr[i].x1) / 2;
        pointArr[i].y = axisArr[i].a * pointArr[i].x + axisArr[i].b;
    }
}

__global__ void ProximityCriteria(int rank, int chunkSize, int *flags, Point *pointArr, int numElements, float D, int K)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x; 
    int index = tid + rank * chunkSize; 
    int counter = 0;
    if (index < numElements)
    {
        Point p1 = pointArr[index]; 
        for (int i = 0; i < numElements && counter < K; i++) 
        {
            if (index == i)
                continue;
            Point p2 = pointArr[i]; 
            if (sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2)) <= D)
                counter++;
        }
        if (counter == K)
            flags[tid] = 1;
    }
}

int computePointsOnGPU(Axis *axisArr, Point *pointArr, int numElements, double t)
{
    cudaError_t err = cudaSuccess;

    size_t size = numElements * sizeof(Axis);
    Axis *d_Axis;
    err = cudaMalloc((void **)&d_Axis, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(d_Axis, axisArr, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy data from host to device - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    size = numElements * sizeof(Point);
    Point *d_Points;
    err = cudaMalloc((void **)&d_Points, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    int threadsPerBlock = 100;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    calculatePoint<<<blocksPerGrid, threadsPerBlock>>>(d_Axis, d_Points, numElements, t); 
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel -  %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(pointArr, d_Points, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy data from device to host - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaFree(d_Axis);
    cudaFree(d_Points);
    return 1;
}

void checkProximityCriteriaOnGPU(int rank, Point *allPoints, int N,int* flags, int chunkSize, float D, int K)
{
    cudaError_t err = cudaSuccess;

    size_t size = N * sizeof(Point);
    Point *d_Points;
    err = cudaMalloc((void **)&d_Points, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(d_Points, allPoints, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        cudaFree(d_Points);
        fprintf(stderr, "Failed to copy data from host to device - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    int *d_Flags;
    size = chunkSize * sizeof(int);
    err = cudaMalloc((void **)&d_Flags, size);
    if (err != cudaSuccess)
    {   cudaFree(d_Flags);
        fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemset(d_Flags, 0, size);
    if (err != cudaSuccess)
    {   
        cudaFree(d_Flags);
        cudaFree(d_Points);
        fprintf(stderr, "Failed to set device memory to zero- %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    int threadsPerBlock = 100;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    ProximityCriteria<<<blocksPerGrid, threadsPerBlock>>>(rank, chunkSize, d_Flags, d_Points, N, D, K);//set flagArr

    err = cudaMemcpy(flags, d_Flags, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        cudaFree(d_Flags);
        cudaFree(d_Points);
        free(flags);
        fprintf(stderr, "Failed to copy data from device to host - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaFree(d_Flags);
    cudaFree(d_Points);
}
