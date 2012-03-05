#include <cuda.h>
#include <cuda_runtime_api.h>
#include "integral.h"

int test(){
	return 7;
}

__global__ 
void kernel(float* d_counters) {
    // Increment the counter
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	// Declare a bunch or registers and init to 0.0f
	float reg0, reg1, reg2,  reg3,  reg4,  reg5,  reg6,  reg7;
	
	// 1 FLOP per assignment = 8 FLOPs total
	reg0  = reg1  = reg2  = reg3  = 9.765625e-10f * threadIdx.x;
	reg4  = reg5  = reg6  = reg7  = 9.765625e-10f * threadIdx.y;
	

	// 8 More flops.
	d_counters[i] = reg0 + reg1 + reg2  + reg3  + reg4  + reg5  + reg6  + reg7  + 8.0f;
}