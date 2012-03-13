#include <cuda.h>
#include <cuda_runtime_api.h>

#define BASE_WIDTH	8
#define BASE_HEIGHT	4
#define THRESHOLD		0.85 //definitely needs to be changed
#define SKIP_AMOUNT	4 //amount to skip in pixels, we can change this to be multiplied by scale if necessary/desirable

//This identifier is 2 horizontal bars with light (positive) on top and dark (negative) on bottom
__global__ 
void ID1kernel(int* offsets, int windowSize, int scale, float* intImage, size_t stride, int numSubWindows, float* results) {
	int threadNum = blockIdx.x * blockDim.x + threadIdx.x;
	if(threadNum < numSubWindows){
		float maxFitValue;
		int maxX;
		int maxY;
		int startX = offsets[threadNum]/(stride);
		int startY = offsets[threadNum]%stride;
		for (int i = startX; (i+BASE_WIDTH*scale) < (startX+windowSize); i = i+SKIP_AMOUNT*scale){ //use SKIP_AMOUNT * scale for it to scale up as identifier scales
			for (int j = startY; (j+BASE_HEIGHT*scale) < (startY + windowSize); j = j+SKIP_AMOUNT*scale){
				// take important corners from image
				float upperLeft 		= intImage[i*stride + j];
				float upperRight 		= intImage[(i+BASE_WIDTH*scale)*stride + j];
				float midLeft 			= intImage[i*stride + j+(BASE_HEIGHT*scale>>1)];
				float midRight 		= intImage[(i+BASE_WIDTH*scale)*stride + j+(BASE_HEIGHT*scale>>1)];
				float lowerLeft 		= intImage[i*stride + j+(BASE_HEIGHT*scale)];
				float lowerRight 		= intImage[(i+BASE_WIDTH*scale)*stride + j+(BASE_HEIGHT*scale)];
		
				//calculate fit value based on identifier (hard-coded)
				float fitValue = midRight*2-midLeft*2 + upperLeft - lowerRight - upperRight + lowerLeft;
				maxFitValue = (maxFitValue < fitValue) ? fitValue : maxFitValue;
				maxX = (maxFitValue <= fitValue) ? i : maxX;
				maxY = (maxFitValue <= fitValue) ? j : maxY;
		
			}
		}
		float goodnessValue = maxFitValue/(BASE_WIDTH*scale*BASE_HEIGHT*scale); // goodnessValue = fit/area
		offsets[threadNum] = maxX;//(goodnessValue>THRESHOLD) ? offsets[threadNum] : -1.0;
		results[threadNum] = maxFitValue;//(float) maxX;
	}

    
}

