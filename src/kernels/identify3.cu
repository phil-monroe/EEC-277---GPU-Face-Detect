#include <cuda.h>
#include <cuda_runtime_api.h>

#define ID3_BASE_WIDTH		3	
#define ID3_BASE_HEIGHT		6
#define ID3_MID_WIDTH		1
#define ID3_THRESHOLD		.19f	//definitely needs to be changed
#define ID3_SKIP_AMOUNT		1 		//amount to skip in pixels, we can change this to be multiplied by scale if necessary/desirable


//This identifier is 3 vertical bars going dark light dark
__global__ 
void ID3kernel(float* intImage, size_t stride, int* offsets, int windowSize, int numSubWindows, int scale, int* faceDetected, float* results, float* heatMap) {

	int threadNum = blockIdx.x * blockDim.x + threadIdx.x;
	if(threadNum < numSubWindows){
		int startX = offsets[threadNum]/(stride);
		int startY = offsets[threadNum]%stride;
		float maxFitValue = 0.0f;
		
		for (int i = startX; (i+ID3_BASE_WIDTH*scale) < (startX+windowSize); i = i+ID3_SKIP_AMOUNT){ //use ID3_SKIP_AMOUNT * scale for it to scale up as identifier scales
			for (int j = startY; (j+ID3_BASE_HEIGHT*scale) < (startY + windowSize); j = j+ID3_SKIP_AMOUNT){
				// take important corners from image
				float upperLeft 		= intImage[i*stride + j];
				float upperRight 		= intImage[(i+ID3_BASE_WIDTH*scale)*stride + j];
				
				float midLeftTop 		= intImage[(i+ID3_BASE_WIDTH*scale/2 - ID3_MID_WIDTH*scale/2)*stride + j];
				float midRightTop		= intImage[(i+ID3_BASE_WIDTH*scale/2 + ID3_MID_WIDTH*scale/2)*stride + j];
				float midLeftBot 		= intImage[(i+ID3_BASE_WIDTH*scale/2 - ID3_MID_WIDTH*scale/2)*stride + j+ID3_BASE_HEIGHT*scale];
				float midRightBot		= intImage[(i+ID3_BASE_WIDTH*scale/2 + ID3_MID_WIDTH*scale/2)*stride + j+ID3_BASE_HEIGHT*scale];
				
				float lowerLeft 		= intImage[i*stride + j+(ID3_BASE_HEIGHT*scale)];
				float lowerRight 		= intImage[(i+ID3_BASE_WIDTH*scale)*stride + j+(ID3_BASE_HEIGHT*scale)];
			
				//calculate fit value based on identifier (hard-coded)
				// float fitValue = (midRightBot + midLeftTop - midRightTop - midLeftBot)*2.0 - lowerRight - upperLeft + upperRight + lowerLeft;
				float fitValue = 2.0*(midRightBot - midLeftBot - midRightTop + midLeftTop) - (lowerRight - lowerLeft - upperRight + upperLeft) ;
				
				if(fitValue < 0)	
					fitValue = -fitValue;
				
				if(fitValue > maxFitValue){
					maxFitValue = fitValue;
				}
			}
		}
		float goodnessValue = maxFitValue/(ID3_BASE_WIDTH*scale*ID3_BASE_HEIGHT*scale); // goodnessValue = fit/area
	
		results[threadNum] = goodnessValue;
		
		if(goodnessValue > ID3_THRESHOLD){
			faceDetected[threadNum] = 1;

			// for(int i = 0; i < windowSize; ++i){
			// 	for(int j = 0; j < windowSize; ++j){
			// 		heatMap[offsets[threadNum] + i*stride + j] = heatMap[offsets[threadNum] + i*stride + j] + 1.0f;
			// 	}
			// }
		}
	}
}
