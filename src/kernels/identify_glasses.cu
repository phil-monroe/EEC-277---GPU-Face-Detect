#include <cuda.h>
#include <cuda_runtime_api.h>

#define GLASSES_FRAME_HEIGHT	2
#define GLASSES_BASE_WIDTH		4
#define GLASSES_BASE_HEIGHT	4
#define GLASSES_THRESHOLD		.22f 	//definitely needs to be changed
#define GLASSES_SKIP_AMOUNT	2 		//amount to skip in pixels, we can change this to be multiplied by scale if necessary/desirable

//This identifier is the glasses identifier with 3 horizontal bars going:
//light 
//dark 
//light
__global__ 
void glassesKernel(float* intImage, size_t stride, int* offsets, int windowSize, int numSubWindows, int scale, int* faceDetected, float* results, float* heatMap) {
	int threadNum = blockIdx.x * blockDim.x + threadIdx.x;
	if(threadNum < numSubWindows){
		int startX = offsets[threadNum]/(stride);
		int startY = offsets[threadNum]%stride;
		float maxFitValue = 0.0f;
		
		for (int i = startX; (i+ID4_BASE_WIDTH*scale) < (startX+windowSize); i = i+ID4_SKIP_AMOUNT){ //use ID4_SKIP_AMOUNT * scale for it to scale up as identifier scales
			for (int j = startY; (j+ID4_BASE_HEIGHT*scale) < (startY + windowSize); j = j+ID4_SKIP_AMOUNT){				// take important corners from image
				// take important corners from image
				float upperLeft 		= intImage[i*stride + j];
				float upperRight 		= intImage[(i+GLASSES_BASE_WIDTH*scale)*stride + j];

				float midLeftTop 		= intImage[i*stride + j + ((GLASSES_BASE_HEIGHT/2 - GLASSES_FRAME_HEIGHT/2) * scale)];
				float midRightTop 	= intImage[(i+GLASSES_BASE_WIDTH*scale)*stride + j + ((GLASSES_BASE_HEIGHT/2 - GLASSES_FRAME_HEIGHT/2) * scale)];

				float midLeftBot 		= intImage[i*stride + j + ((GLASSES_BASE_HEIGHT/2 + GLASSES_FRAME_HEIGHT/2) * scale)];
				float midRightBot 	= intImage[(i+GLASSES_BASE_WIDTH*scale)*stride + j + ((GLASSES_BASE_HEIGHT/2 + GLASSES_FRAME_HEIGHT/2) * scale)];

				float lowerLeft 		= intImage[i*stride + j+((GLASSES_FRAME_HEIGHT+GLASSES_BASE_HEIGHT)*scale)];
				float lowerRight 		= intImage[(i+GLASSES_BASE_WIDTH*scale)*stride + j+((GLASSES_FRAME_HEIGHT+GLASSES_BASE_HEIGHT)*scale)];
		
				//calculate fit value based on identifier (hard-coded)
				float fitValue = upperLeft - lowerLeft - upperRight + lowerRight + (midRightTop + midLeftBot - midLeftTop - midRightBot)*2;
						
				
				if(fitValue > maxFitValue){
					maxFitValue = fitValue;
				}
			}
		}
		// float goodnessValue = fitValue*1.0f/(GLASSES_BASE_WIDTH*scale*(GLASSES_FRAME_HEIGHT + GLASSES_BASE_HEIGHT)*scale); // goodnessValue = fit/area
		
		float goodnessValue = maxFitValue/(ID4_BASE_WIDTH*scale*ID4_BASE_HEIGHT*scale); // goodnessValue = fit/area
	
		results[threadNum] = goodnessValue;
		
		if(goodnessValue > GLASSES_THRESHOLD){
			faceDetected[threadNum] = 1;

			// for(int i = 0; i < windowSize; ++i){
			// 	for(int j = 0; j < windowSize; ++j){
			// 		heatMap[offsets[threadNum] + i*stride + j] = heatMap[offsets[threadNum] + i*stride + j] + 1.0f;
			// 	}
			// }
		}
	}
}
