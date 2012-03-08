#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "integral.h"
#include "helpers.cu"

void cuda_test_classifier(int* winX, int* winY, size_t stride, int winSize, int featSize, float* intImg, bool* results){
	int thread = blockIdx.x * blockDim.x + threadIdx.x;
}