#include <cuda.h>
#include <cuda_runtime_api.h>
#include "identify_glasses.h"

#define FRAME_HEIGHT = 2
#define BASE_WIDTH = 4
#define BASE_HEIGHT = 4
#define THRESHOLD = 0.85 //definitely needs to be changed
#define SKIP_AMOUNT = 4 //amount to skip in pixels, we can change this to be multiplied by scale if necessary/desirable

//This identifier is the glasses identifier with 3 horizontal bars going:
//light 
//dark 
//light
__global__ 
void ID4kernel(int* xVals, int* yVals, int windowSize, int scale, float* intImage, size_t stride, bool* results ) {

	int threadNum = blockIdx.x * blockDim.x + threadIdx.x;
	int startX = xVals[threadNum];
	int startY = yVals[threadNum];
	for (int i = startX; (i+BASE_WIDTH*scale) < (startX+windowSize); i = i+SKIP_AMOUNT){ //use SKIP_AMOUNT * scale for it to scale up as identifier scales
		for (int j = startY; (j+(BASE_HEIGHT)*scale) < (startY + windowSize); j = j+SKIP_AMOUNT){
			// take important corners from image
			int upperLeft 		= intImage[i*stride + j];
			int upperRight 		= intImage[(i+BASE_WIDTH*scale)*stride + j];
			
			int midLeftTop 		= intImage[i*stride + j + ((BASE_HEIGHT>>1 - FRAME_HEIGHT>>1) * scale)];
			int midRightTop 		= intImage[(i+BASE_WIDTH*scale)*stride + j + ((BASE_HEIGHT>>1 - FRAME_HEIGHT>>1) * scale)];
			
			int midLeftBot 		= intImage[i*stride + j + ((BASE_HEIGHT>>1 + FRAME_HEIGHT>>1) * scale)];
			int midRightBot 		= intImage[(i+BASE_WIDTH*scale)*stride + j + ((BASE_HEIGHT>>1 + FRAME_HEIGHT>>1) * scale)];
			
			int lowerLeft 		= intImage[i*stride + j+((FRAME_HEIGHT+EYE_HEIGHT)*scale)];
			int lowerRight 		= intImage[(i+BASE_WIDTH*scale)*stride + j+((FRAME_HEIGHT+EYE_HEIGHT)*scale)];
			
			//calculate fit value based on identifier (hard-coded)
			int fitValue = upperLeft - lowerLeft - upperRight + lowerRight + (midRightTop + midRightBot - midLeftTop - midRightBot)<<1;
			float goodnessValue = fitValue*1.0f/(BASE_WIDTH*scale*(FRAME_HEIGHT + EYE_HEIGHT)*scale); // goodnessValue = fit/area
			
			results[i*stride + j] = (goodnessValue>THRESHOLD);
		}
	}
    
}
