#include <cuda.h>
#include <cuda_runtime_api.h>

#define ID1_BASE_WIDTH			8
#define ID1_BASE_HEIGHT			4
#define ID1_THRESHOLD			100.0f 	//definitely needs to be changed
#define ID1_SKIP_AMOUNT			4 			//amount to skip in pixels, we can change this to be multiplied by scale if necessary/desirable

//This identifier is 2 horizontal bars with light (positive) on top and dark (negative) on bottom
__global__ 
void ID1kernel(float* intImage, size_t stride, int* offsets, int windowSize, int numSubWindows, int scale, int* faceDetected, float* results, float* heatMap) {
	int threadNum = blockIdx.x * blockDim.x + threadIdx.x;
	if(threadNum < numSubWindows){
		float maxFitValue = 0.0f;
		int startX = offsets[threadNum]/(stride);
		int startY = offsets[threadNum]%stride;
		for (int i = startX; (i+ID1_BASE_WIDTH*scale) < (startX+windowSize); i = i+ID1_SKIP_AMOUNT){ //use ID1_SKIP_AMOUNT * scale for it to scale up as identifier scales
			for (int j = startY; (j+ID1_BASE_HEIGHT*scale) < (startY + windowSize); j = j+ID1_SKIP_AMOUNT){
				// take important corners from image
				float upperLeft 		= intImage[i*stride + j];
				float upperRight 		= intImage[(i+ID1_BASE_WIDTH*scale)*stride + j];
				float midLeft 			= intImage[i*stride + j+(ID1_BASE_HEIGHT*scale>>1)];
				float midRight 		= intImage[(i+ID1_BASE_WIDTH*scale)*stride + j+(ID1_BASE_HEIGHT*scale>>1)];
				float lowerLeft 		= intImage[i*stride + j+(ID1_BASE_HEIGHT*scale)];
				float lowerRight 		= intImage[(i+ID1_BASE_WIDTH*scale)*stride + j+(ID1_BASE_HEIGHT*scale)];
						
				//calulate fit value based on identifier (hard-coded)
				float fitValue = midRight*2 - midLeft*2 + upperLeft - lowerRight - upperRight + lowerLeft;
				
				if(fitValue > maxFitValue){
					maxFitValue = fitValue;
				}
			}
		}
		float goodnessValue = maxFitValue;//(ID1_BASE_WIDTH*scale*ID1_BASE_HEIGHT*scale); // goodnessValue = fit/area
		results[threadNum] = goodnessValue;
		
		if(goodnessValue > ID1_THRESHOLD){
			faceDetected[threadNum] = 1;

			for(int i = 0; i < windowSize; ++i){
				for(int j = 0; j < windowSize; ++j){
					heatMap[offsets[threadNum] + i*stride + j] = heatMap[offsets[threadNum] + i*stride + j] + 1.0f;
				}
			}
		}
	}
}

