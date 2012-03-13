#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "cuda_detect_faces.h"
#include "cuda_helpers.h"
#include "cudpp.h"
#include "identify1.cu"

#define TH_PER_BLOCK 64


void cuda_detect_faces(float* intImg, int rows, int cols, size_t stride, int* windowOffsets, int numWindows, int windowSize){
	CUDPPResult res;
	
	float* results = (float*) malloc(numWindows*sizeof(float));
	float* results2 = (float*) malloc(numWindows*sizeof(float));
	float* results_d;
	float* intImg_d;
	int*		winOffsets_d;
	cudaMalloc(&results_d, numWindows*sizeof(float));
	cudaMalloc(&intImg_d, rows*cols*sizeof(float));
	cudaMalloc(&winOffsets_d, numWindows*sizeof(int));
	checkCUDAError("malloc");
	
	float* results_d2;
	float* intImg_d2;
	int*		winOffsets_d2;
	cudaMalloc(&results_d2, numWindows*sizeof(float));
	cudaMalloc(&winOffsets_d2, numWindows*sizeof(int));
	checkCUDAError("malloc2");
	
	
	cudaMemcpy(winOffsets_d, windowOffsets, numWindows*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(intImg_d, intImg, rows*cols*sizeof(float), cudaMemcpyHostToDevice);
	checkCUDAError("memcpy");
	
	cudaMemcpy(winOffsets_d2, windowOffsets, numWindows*sizeof(int), cudaMemcpyHostToDevice);
	checkCUDAError("memcpy2");
	
	ID1kernel<<<1, TH_PER_BLOCK>>>(winOffsets_d, windowSize, windowSize/10, intImg_d, stride, numWindows, results_d);
	ID1kernel<<<1, TH_PER_BLOCK>>>(winOffsets_d2, windowSize, windowSize/15, intImg_d, stride, numWindows, results_d2);
	cudaThreadSynchronize();
	checkCUDAError("kernel");

	cudaMemcpy(results, results_d, numWindows*sizeof(float), cudaMemcpyDeviceToHost);
	checkCUDAError("results memcpy");
	
	cudaMemcpy(results2, results_d2, numWindows*sizeof(float), cudaMemcpyDeviceToHost);
	checkCUDAError("results memcpy2");


	for(size_t i = 0; i < numWindows; ++i){
		if(results[i] > 1.0f || results[i] < -1.0f){
			intImg[windowOffsets[i]] = 1.0;
		}
		printf("%d - %f - %d\n", i, results[i], windowOffsets[i]);
	}
	
	for(size_t i = 0; i < numWindows; ++i){
		if(results[i] > 1.0f || results[i] < -1.0f){
			intImg[windowOffsets[i]] = 1.0;
		}
		printf("%d - %f - %d\n", i, results2[i], windowOffsets[i]);
	}
	
	// printf("run 2:\n");
	
	// float* results_d2;
	// 	float* intImg_d2;
	// 	int*		winOffsets_d2;
	// 	cudaMalloc(&results_d2, numWindows*sizeof(float));
	// 	cudaMalloc(&winOffsets_d2, numWindows*sizeof(int));
	// 	checkCUDAError("malloc");
	// 	
	// 	
	// 	cudaMemcpy(winOffsets_d2, windowOffsets, numWindows*sizeof(int), cudaMemcpyHostToDevice);
	// 	checkCUDAError("memcpy");
	// 	
	// 	ID1kernel<<<1, TH_PER_BLOCK>>>(winOffsets_d2, windowSize, windowSize/15, intImg_d, stride, numWindows, results_d2);
	// 	cudaThreadSynchronize();
	// 	checkCUDAError("kernel");
	// 
	// 	cudaMemcpy(results, results_d2, numWindows*sizeof(float), cudaMemcpyDeviceToHost);
	// 	checkCUDAError("results memcpy");
	// 
	// 
	// 	for(size_t i = 0; i < numWindows; ++i){
	// 		if(results[i] > 1.0f || results[i] < -1.0f){
	// 			intImg[windowOffsets[i]] = 1.0;
	// 		}
	// 		printf("%d - %f - %d\n", i, results[i], windowOffsets[i]);
	// 	}

}