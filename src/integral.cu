#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "integral.h"
#include "helpers.cu"

#define THREADS_PER_BLOCK 32




__global__ 
void horizontal_kernel(float* data, int rows, int cols, size_t stride ) {
    // start from row 0
	int row = blockIdx.x * blockDim.x + threadIdx.x;

	if(row < rows){
		for(int col = 1; col<cols; ++col){
				data[row*stride + col] = data[row*stride + col] + data[row*stride + col-1];
		}
	}

	
}

__global__ 
void vertical_kernel(float* data, int rows, int cols, size_t stride ) {
    // Start from column 1
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(col < cols){
		for(int row = 1; row<rows; ++row){
			data[row*stride + col] = data[row*stride + col] + data[(row-1)*stride + col] ;
		}
	}
}



void cuda_integrate_image(float* data, int rows, int cols, size_t stride){
	float *dev_data, *dev_fin_data;
	cudaMalloc( &dev_data, rows*cols*sizeof(float));
	cudaMalloc( &dev_fin_data, rows*cols*sizeof(float));
	
	checkCUDAError("malloc");
	
	cudaMemcpy(dev_data, data, rows*cols*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_fin_data, data, rows*cols*sizeof(float), cudaMemcpyHostToDevice);
	
	checkCUDAError("memcpy host to device");
	
	int num_blocks = rows < THREADS_PER_BLOCK ? 1 : rows/THREADS_PER_BLOCK + 1;
	
	horizontal_kernel<<<num_blocks , THREADS_PER_BLOCK>>>(dev_data, rows, cols, stride);
	
	num_blocks = cols < THREADS_PER_BLOCK ? 1 : cols/THREADS_PER_BLOCK + 1;
	
	cudaThreadSynchronize();
	checkCUDAError("horizontal kernel");
	
	vertical_kernel<<<num_blocks , THREADS_PER_BLOCK>>>(dev_data, rows, cols, stride);
	cudaThreadSynchronize();
	checkCUDAError("vertical kernel");
	
	cudaMemcpy(data, dev_data, rows*cols*sizeof(float), cudaMemcpyDeviceToHost);
	checkCUDAError("memcpy device to host");
	
	cudaFree(dev_data);
	cudaFree(dev_fin_data);
	
	checkCUDAError("free");
	
}