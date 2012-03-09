#include <cuda.h>
#include <cuda_runtime_api.h>
#include "identify3.h"

#define BASE_WIDTH = 6
#define BASE_HEIGHT = 6
#define MID_WIDTH = 2
#define THRESHOLD = 0.85 //definitely needs to be changed
#define SKIP_AMOUNT = 4 //amount to skip in pixels, we can change this to be multiplied by scale if necessary/desirable


//This identifier is 3 vertical bars going dark light dark
__global__ 
void ID3kernel(int* xVals, int* yVals, int windowSize, int scale, float* intImage, size_t stride, bool* results ) {

	int threadNum = blockIdx.x * blockDim.x + threadIdx.x;
	int startX = xVals[threadNum];
	int startY = yVals[threadNum];
	for (int i = startX; (i+BASE_WIDTH*scale) < (startX+windowSize); i = i+SKIP_AMOUNT){ //use SKIP_AMOUNT * scale for it to scale up as identifier scales
		for (int j = startY; (j+BASE_HEIGHT*scale) < (startY + windowSize); j = j+SKIP_AMOUNT){
			// take important corners from image
			int upperLeft 		= intImage[i*stride + j];
			int upperRight 		= intImage[(i+BASE_WIDTH*scale)*stride + j];
			
			int midLeftTop 		= intImage[(i+BASE_WIDTH*scale>>1 - MID_WIDTH*scale>>1)*stride + j];
			int midRightTop		= intImage[(i+BASE_WIDTH*scale>>1 + MID_WIDTH*scale>>1)*stride + j];
			int midLeftBot 		= intImage[(i+BASE_WIDTH*scale>>1 - MID_WIDTH*scale>>1)*stride + j+BASE_HEIGHT*scale];
			int midRightBot		= intImage[(i+BASE_WIDTH*scale>>1 + MID_WIDTH*scale>>1)*stride + j+BASE_HEIGHT*scale];
			
			int lowerLeft 		= intImage[i*stride + j+(BASE_HEIGHT*scale)];
			int lowerRight 		= intImage[(i+BASE_WIDTH*scale)*stride + j+(BASE_HEIGHT*scale)];
			
			//calculate fit value based on identifier (hard-coded)
			int fitValue = midRightBot + midLeftTop - midRightTop - midLeftBot - lowerRight - upperLeft + upperRight + lowerLeft;
			float goodnessValue = fitValue*1.0f/(BASE_WIDTH*scale*BASE_HEIGHT*scale); // goodnessValue = fit/area
			
			results[i*stride + j] = (goodnessValue>THRESHOLD);
		}
	}
    
}
