#include <cuda.h>
#include <cuda_runtime_api.h>

#define GLASSES_FRAME_HEIGHT = 2
#define GLASSES_BASE_WIDTH = 4
#define GLASSES_BASE_HEIGHT = 4
#define GLASSES_THRESHOLD = 0.85 //definitely needs to be changed
#define GLASSES_SKIP_AMOUNT = 4 //amount to skip in pixels, we can change this to be multiplied by scale if necessary/desirable

//This identifier is the glasses identifier with 3 horizontal bars going:
//light 
//dark 
//light
__global__ 
void ID4kernel(int* xVals, int* yVals, int windowSize, int scale, float* intImage, size_t stride, bool* results ) {

	int threadNum = blockIdx.x * blockDim.x + threadIdx.x;
	int startX = xVals[threadNum];
	int startY = yVals[threadNum];
	for (int i = startX; (i+GLASSES_BASE_WIDTH*scale) < (startX+windowSize); i = i+GLASSES_SKIP_AMOUNT){ //use GLASSES_SKIP_AMOUNT * scale for it to scale up as identifier scales
		for (int j = startY; (j+(GLASSES_BASE_HEIGHT)*scale) < (startY + windowSize); j = j+GLASSES_SKIP_AMOUNT){
			// take important corners from image
			int upperLeft 		= intImage[i*stride + j];
			int upperRight 		= intImage[(i+GLASSES_BASE_WIDTH*scale)*stride + j];
			
			int midLeftTop 		= intImage[i*stride + j + ((GLASSES_BASE_HEIGHT>>1 - GLASSES_FRAME_HEIGHT>>1) * scale)];
			int midRightTop 		= intImage[(i+GLASSES_BASE_WIDTH*scale)*stride + j + ((GLASSES_BASE_HEIGHT>>1 - GLASSES_FRAME_HEIGHT>>1) * scale)];
			
			int midLeftBot 		= intImage[i*stride + j + ((GLASSES_BASE_HEIGHT>>1 + GLASSES_FRAME_HEIGHT>>1) * scale)];
			int midRightBot 		= intImage[(i+GLASSES_BASE_WIDTH*scale)*stride + j + ((GLASSES_BASE_HEIGHT>>1 + GLASSES_FRAME_HEIGHT>>1) * scale)];
			
			int lowerLeft 		= intImage[i*stride + j+((GLASSES_FRAME_HEIGHT+EYE_HEIGHT)*scale)];
			int lowerRight 		= intImage[(i+GLASSES_BASE_WIDTH*scale)*stride + j+((GLASSES_FRAME_HEIGHT+EYE_HEIGHT)*scale)];
			
			//calculate fit value based on identifier (hard-coded)
			int fitValue = upperLeft - lowerLeft - upperRight + lowerRight + (midRightTop + midRightBot - midLeftTop - midRightBot)<<1;
			float goodnessValue = fitValue*1.0f/(GLASSES_BASE_WIDTH*scale*(GLASSES_FRAME_HEIGHT + EYE_HEIGHT)*scale); // goodnessValue = fit/area
			
			results[i*stride + j] = (goodnessValue>GLASSES_THRESHOLD);
		}
	}
    
}
