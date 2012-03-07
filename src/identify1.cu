#include <cuda.h>
#include <cuda_runtime_api.h>
#include "identify1.h"

#define BASE_WIDTH = 8
#define BASE_HEIGHT = 2
#define THRESHOLD = 0.85 //definitely needs to be changed

__global__ 
void ID1kernel(int startX, int startY, int scale, int** integImage ) {
    // take important corners from image
	int upperLeft = integImage[startX][startY];
	int upperRight = integImage[startX+BASE_WIDTH*scale][startY];
	int midLeft = integImage[startX][startY+BASE_HEIGHT*scale];
	int midRight = integImage[startX+BASE_WIDTH*scale][startY+BASE_HEIGHT*scale];
	int lowerLeft = integImage[startX][startY+BASE_HEIGHT*scale<<1];
	int lowerRight = integImage[startX+BASE_WIDTH*scale][startY+BASE_HEIGHT*scale<<1];
	
	//calculate fit value based on identifier (hard-coded)
	int fitValue = midRight<<1-midLeft<<1 + upperLeft - lowerRight - upperRight + lowerLeft;
	float goodnessValue = fitValue*1.0f/(BASE_WIDTH*scale*BASE_HEIGHT*scale<<1); // goodnessValue = fit/area

}
