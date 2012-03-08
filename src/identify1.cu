#include <cuda.h>
#include <cuda_runtime_api.h>
#include "identify1.h"

#define BASE_WIDTH = 8
#define BASE_HEIGHT = 2
#define THRESHOLD = 0.85 //definitely needs to be changed
#define SKIP_AMOUNT = 4 //amount to skip in pixels, we can change this to be multiplied by scale if necessary/desirable

__global__ 
void ID1kernel(int* xVals, int* yVals, int windowSize, int scale, float* intImage, size_t stride, bool* results ) {
	//Does the intImage need to be float** so we can address into it twice or is this how it works in Cuda

	int threadNum = blockIdx.x * blockDim.x + threadIdx.x;
	int startX = xVals[threadNum];
	int startY = yVals[threadNum];
	for (int i = startX; i < startX+windowSize; i = i+SKIP_AMOUNT){ //use SKIP_AMOUNT * scale for it to scale up as identifier scales
		for (int j = startY; j < startY + windowSize; j = j+SKIP_AMOUNT){
			// take important corners from image
			int upperLeft 		= intImage[i*stride + j];
			int upperRight 		= intImage[(i+BASE_WIDTH*scale)*stride + j];
			int midLeft 		= intImage[i*stride + j+BASE_HEIGHT*scale];
			int midRight 		= intImage[(i+BASE_WIDTH*scale)*stride + j+BASE_HEIGHT*scale];
			int lowerLeft 		= intImage[i*stride + j+BASE_HEIGHT*scale<<1];
			int lowerRight 		= intImage[(i+BASE_WIDTH*scale)*stride + j+BASE_HEIGHT*scale<<1];
			
			//calculate fit value based on identifier (hard-coded)
			int fitValue = midRight<<1-midLeft<<1 + upperLeft - lowerRight - upperRight + lowerLeft;
			float goodnessValue = fitValue*1.0f/(BASE_WIDTH*scale*BASE_HEIGHT*scale<<1); // goodnessValue = fit/area
			
			results[i*stride + j] = (goodnessValue>THRESHOLD);
		}
	}
    
}
