#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "integral.h"
#include "cuda_helpers.h"

__device__
void cuda_test_classifier(int* winXs, int* winYs, int winSize, int featSize, float* intImg, size_t stride, bool* results){
	int thread = blockIdx.x * blockDim.x + threadIdx.x;
	int winX = winXs[thread];
	int winY = winYs[thread];
	
	results[thread] = thread%2 ? true : false;
}