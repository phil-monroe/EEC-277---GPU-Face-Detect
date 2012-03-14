#include <cuda.h>
#include <cuda_runtime_api.h>

__global__ 
void brute_merge_results(int* ar0, int* ar1, int* ar2, int* ar3, int array_len) {
	int threadNum = blockIdx.x * blockDim.x + threadIdx.x;
	if(threadNum < array_len){
		int val = 0;
		if(ar0[threadNum] !=0 && ar1[threadNum] !=0 && ar2[threadNum] !=0 && ar3[threadNum] !=0){
			val = 1;
		}
		ar0[threadNum] = val;
	}
}