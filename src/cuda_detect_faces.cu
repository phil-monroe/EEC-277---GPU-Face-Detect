#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <thrust/remove.h>


#include "cuda_detect_faces.h"
#include "cuda_helpers.h"
#include "identify1.cu"
#include "identify2.cu"
#include "identify3.cu"


#define TH_PER_BLOCK 64
#define N_KERNELS		1
#define N_SCALES		1


void debugResults(int* facesDetected_d, float* results_d, int nValidSubWindows);
int compact(int* winOffsets_d, int* faceDetected_d,  int nValidSubWindows);


void cuda_detect_faces(float* intImg, int rows, int cols, size_t stride, int* winOffsets, int numWindows, int winSize, float* heatMap){
	
	// Initialize kernel size --------------------------------------------------
	int blocks = 1;
	int th_per_block = TH_PER_BLOCK;
	int threads = blocks*th_per_block;
	
	
	// Initialize clock --------------------------------------------------------
	clock_t start;
	
	
	// Copy Integral Image to device -------------------------------------------
	float* intImg_d;
	cudaMalloc(&intImg_d, rows*cols*sizeof(float));
	cudaMemcpy(intImg_d, intImg, rows*cols*sizeof(float), cudaMemcpyHostToDevice);
	checkCUDAError("integral image");
	
	
	// Copy Heat Map to device -------------------------------------------
	float* heatMap_d;
	cudaMalloc(&heatMap_d, rows*cols*sizeof(float));
	cudaMemcpy(heatMap_d, heatMap, rows*cols*sizeof(float), cudaMemcpyHostToDevice);
	checkCUDAError("heat map image");

	
	// Copy window offsets to device -------------------------------------------
	int* winOffsets_d;
	int nValidSubWindows = numWindows;
	
	cudaMalloc(&winOffsets_d, nValidSubWindows*sizeof(int));
	cudaMemcpy(winOffsets_d, winOffsets, nValidSubWindows*sizeof(int), cudaMemcpyHostToDevice);
	checkCUDAError("window offsets");
	
	
	// Initialize device 'boolean' face detected array -------------------------
	int* faceDetected_d;
	cudaMalloc(&faceDetected_d, nValidSubWindows*sizeof(int));
	cudaMemset(faceDetected_d, 0, nValidSubWindows*sizeof(int));
	checkCUDAError("bool array");
	
	
	// Initialize results array for debugging... -------------------------------
	float* results_d;
	cudaMalloc(&results_d, nValidSubWindows*sizeof(float));
	cudaMemset(results_d, 0, nValidSubWindows*sizeof(float));
	checkCUDAError("debug results");
	
	
	
	//==========================================================================
	// Run ID1 -----------------------------------------------------------------
	printf("\n\n");
	printf("Running ID1 --------\n");
	printf("Blocks:   %d\n", blocks);
	printf("Th/Block: %d\n", th_per_block);
	printf("Threads:  %d\n", threads);
	start = clock();
	for(int i = 2; i < 2+N_SCALES; ++i){
		ID1kernel<<<blocks, th_per_block>>>(intImg_d, 					// Itegral Image
														stride, 						//	Stride
														winOffsets_d, 				//	Sub-Window Offsets
														winSize, 					//	Sub-Window Size
														nValidSubWindows, 		//	Number of Sub Windows
														winSize/(5*(i)), 			// Scale of the feature
														faceDetected_d, 			//	Array to hold if a face was detected
														results_d,					//	Array to hold maximum feature value for each sub window
														heatMap_d					// Heat map
														);
	}
	cudaThreadSynchronize();
	printf("Completed in %f seconds\n", ((double)clock() - start) / CLOCKS_PER_SEC);
	checkCUDAError("kernel ID1");

	
	// Compact -----------------------------------------------------------------
	nValidSubWindows = compact(winOffsets_d, faceDetected_d,  nValidSubWindows);
	printf("Possible faces: %d\n", nValidSubWindows);
	
	
	// Prepare for next run ----------------------------------------------------
	cudaMemset(faceDetected_d, 0, nValidSubWindows*sizeof(float));
	cudaMemset(results_d, 0, nValidSubWindows*sizeof(float));
	
	
	
	//==========================================================================
	// Run ID2 -----------------------------------------------------------------
	printf("\n\n");
	printf("Running ID2 --------\n");
	printf("Blocks:   %d\n", blocks);
	printf("Th/Block: %d\n", th_per_block);
	printf("Threads:  %d\n", threads);
	printf("Windows:  %d\n", nValidSubWindows);
	
	start = clock();
	for(int i = 2; i < 2+N_SCALES; ++i){
		ID2kernel<<<blocks, th_per_block>>>(intImg_d, 					// Itegral Image
														stride, 						//	Stride
														winOffsets_d, 				//	Sub-Window Offsets
														winSize, 					//	Sub-Window Size
														nValidSubWindows, 		//	Number of Sub Windows
														winSize/(5*(i)), 			// Scale of the feature
														faceDetected_d, 			//	Array to hold if a face was detected
														results_d,					//	Array to hold maximum feature value for each sub window
														heatMap_d					// Heat map
														);
	}
	cudaThreadSynchronize();
	printf("Completed in %f seconds\n", ((double)clock() - start) / CLOCKS_PER_SEC);
	checkCUDAError("kernel ID2");

	
	// Compact
	nValidSubWindows = compact(winOffsets_d, faceDetected_d,  nValidSubWindows);
	printf("Possible faces: %d\n", nValidSubWindows);
	
	// Prepare for next run
	cudaMemset(faceDetected_d, 0, nValidSubWindows*sizeof(float));
	cudaMemset(results_d, 0, nValidSubWindows*sizeof(float));



	//==========================================================================
	// Run ID3 -----------------------------------------------------------------
	printf("\n\n");
	printf("Running ID3 --------\n");
	printf("Blocks:   %d\n", blocks);
	printf("Th/Block: %d\n", th_per_block);
	printf("Threads:  %d\n", threads);
	printf("Windows:  %d\n", nValidSubWindows);
	
	start = clock();
	for(int i = 2; i < 2+N_SCALES; ++i){
		ID3kernel<<<blocks, th_per_block>>>(intImg_d, 					// Itegral Image
														stride, 						//	Stride
														winOffsets_d, 				//	Sub-Window Offsets
														winSize, 					//	Sub-Window Size
														nValidSubWindows, 		//	Number of Sub Windows
														winSize/(5*(i)), 			// Scale of the feature
														faceDetected_d, 			//	Array to hold if a face was detected
														results_d,					//	Array to hold maximum feature value for each sub window
														heatMap_d					// Heat map
														);
	}
	cudaThreadSynchronize();
	printf("Completed in %f seconds\n", ((double)clock() - start) / CLOCKS_PER_SEC);
	checkCUDAError("kernel ID3");

	debugResults(faceDetected_d, results_d, nValidSubWindows);
	
	
	// Compact
	nValidSubWindows = compact(winOffsets_d, faceDetected_d,  nValidSubWindows);
	printf("Possible faces: %d\n", nValidSubWindows);
	
	// Prepare for next run
	cudaMemset(faceDetected_d, 0, nValidSubWindows*sizeof(float));
	cudaMemset(results_d, 0, nValidSubWindows*sizeof(float));



	// Cleanup
	cudaMemcpy(heatMap, heatMap_d, rows*cols*sizeof(float), cudaMemcpyDeviceToHost);
	
	
	cudaFree(intImg_d);
	cudaFree(heatMap_d);
	cudaFree(winOffsets_d);
	cudaFree(faceDetected_d);
	cudaFree(results_d);
	checkCUDAError("cudaFree");
}


void debugResults(int* facesDetected_d, float* results_d, int nValidSubWindows){
	int* 		facesDetected 	= (int*) 	malloc(nValidSubWindows*sizeof(int));
	float*	results			= (float*)	malloc(nValidSubWindows*sizeof(float));
	
	cudaMemcpy(facesDetected, 	facesDetected_d, 	nValidSubWindows*sizeof(int), 	cudaMemcpyDeviceToHost);
	cudaMemcpy(results, 			results_d, 			nValidSubWindows*sizeof(float), 	cudaMemcpyDeviceToHost);

	
	for(int i = 0; i < nValidSubWindows; ++i){
		printf("%4d - %f: ", i, results[i]);
		if(facesDetected[i] == 0){
		} else if(facesDetected[i] == 1)	{
			printf(" FACE DETECTED");
		} else {
			printf(" SHIT!!!");
		}
		printf("\n");
	}
	
	free(facesDetected);
	free(results);
}


int compact(int* winOffsets_d, int* faceDetected_d, int nValidSubWindows){
	// Cast to thrust device pointers
	thrust::device_ptr<int> offsets_ptr(winOffsets_d);
	thrust::device_ptr<int> detected_ptr(faceDetected_d);
	
	// Perform the compact!
	thrust::device_ptr<int> new_end = thrust::remove_if(offsets_ptr, offsets_ptr + nValidSubWindows, detected_ptr, thrust::logical_not<int>());
	
	// Return the length of the compacted array
	return new_end - offsets_ptr;
}






