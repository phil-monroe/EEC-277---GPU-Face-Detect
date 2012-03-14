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
#include "identify4.cu"


#define TH_PER_BLOCK 64
#define N_KERNELS		1
#define N_SCALES		1


void debugResults(int* facesDetected_d, float* results_d, int nValidSubWindows);
int compact(int* winOffsets_d, int* faceDetected_d,  int nValidSubWindows);
void kernel_heading(char* heading, int blocks, int th_per_block, int threads, int nValidSubWindows);
void kernel_footer(char* msg, clock_t kernel_start);



void cuda_detect_faces(float* intImg, int rows, int cols, size_t stride, int* winOffsets, int numWindows, int winSize, float* heatMap){
	
	// Initialize kernel size --------------------------------------------------
	int blocks = 1;
	int th_per_block = TH_PER_BLOCK;
	int threads = blocks*th_per_block;
	
	
	// Initialize clock --------------------------------------------------------
	clock_t test_start = clock();
	clock_t kernel_start;
	
	
	// Copy Integral Image to device -------------------------------------------
	float* intImg_d;
	cudaMalloc(&intImg_d, rows*cols*sizeof(float));
	cudaMemcpy(intImg_d, intImg, rows*cols*sizeof(float), cudaMemcpyHostToDevice);
	checkCUDAError("integral image");
	
	
	// Copy Heat Map to device -------------------------------------------
	float* heatMap_d;
	// cudaMalloc(&heatMap_d, rows*cols*sizeof(float));
	// cudaMemcpy(heatMap_d, heatMap, rows*cols*sizeof(float), cudaMemcpyHostToDevice);
	// checkCUDAError("heat map image");

	
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
	// cudaMalloc(&results_d, nValidSubWindows*sizeof(float));
	// cudaMemset(results_d, 0, nValidSubWindows*sizeof(float));
	// checkCUDAError("debug results");
	
	
	
	//==========================================================================
	// Run ID1 -----------------------------------------------------------------
	kernel_heading("ID1", blocks, th_per_block, threads, nValidSubWindows);
	kernel_start = clock();
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
	kernel_footer("ID1", kernel_start);


	
	// Compact -----------------------------------------------------------------
	nValidSubWindows = compact(winOffsets_d, faceDetected_d,  nValidSubWindows);
	
	
	// Prepare for next run ----------------------------------------------------
	cudaMemset(faceDetected_d, 0, nValidSubWindows*sizeof(float));
	// cudaMemset(results_d, 0, nValidSubWindows*sizeof(float));
	
	
	
	//==========================================================================
	// Run ID2 -----------------------------------------------------------------
	kernel_heading("ID2", blocks, th_per_block, threads, nValidSubWindows);
	kernel_start = clock();
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
	kernel_footer("ID2", kernel_start);


	
	// Compact -----------------------------------------------------------------
	nValidSubWindows = compact(winOffsets_d, faceDetected_d,  nValidSubWindows);
	
	// Prepare for next run ----------------------------------------------------
	cudaMemset(faceDetected_d, 0, nValidSubWindows*sizeof(float));
	// cudaMemset(results_d, 0, nValidSubWindows*sizeof(float));



	//==========================================================================
	// Run ID3 -----------------------------------------------------------------
	kernel_heading("ID3", blocks, th_per_block, threads, nValidSubWindows);
	kernel_start = clock();
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
	kernel_footer("ID3", kernel_start);

	
	
	// Compact -----------------------------------------------------------------
	nValidSubWindows = compact(winOffsets_d, faceDetected_d,  nValidSubWindows);
	
	// Prepare for next run ----------------------------------------------------
	cudaMemset(faceDetected_d, 0, nValidSubWindows*sizeof(float));
	// cudaMemset(results_d, 0, nValidSubWindows*sizeof(float));
	
	
	
	//==========================================================================
	// Run ID4 -----------------------------------------------------------------
	kernel_heading("ID4", blocks, th_per_block, threads, nValidSubWindows);
	kernel_start = clock();
	for(int i = 2; i < 2+N_SCALES; ++i){
		ID4kernel<<<blocks, th_per_block>>>(intImg_d, 					// Itegral Image
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
	kernel_footer("ID4", kernel_start);

	
	// Compact -----------------------------------------------------------------
	nValidSubWindows = compact(winOffsets_d, faceDetected_d,  nValidSubWindows);
	
	// Prepare for next run ----------------------------------------------------
	// cudaMemset(faceDetected_d, 0, nValidSubWindows*sizeof(float));
	// cudaMemset(results_d, 0, nValidSubWindows*sizeof(float));
	
	
	// Results -----------------------------------------------------------------
	printf("Results\n\n");
	printf("Completed test in %f seconds\n", ((double)clock() - test_start) / CLOCKS_PER_SEC);
	if(nValidSubWindows > 0){
		printf("A face was detected\n");
	}
	

	// Retrieve the Heat Map ---------------------------------------------------
	// cudaMemcpy(heatMap, heatMap_d, rows*cols*sizeof(float), cudaMemcpyDeviceToHost);
	
	
	// Cleanup -----------------------------------------------------------------
	cudaFree(intImg_d);
	// cudaFree(heatMap_d);
	cudaFree(winOffsets_d);
	cudaFree(faceDetected_d);
	// cudaFree(results_d);
	checkCUDAError("cudaFree");
}

void cuda_detect_faces2(float* intImg, int rows, int cols, size_t stride, int* winOffsets, int numWindows, int winSize, float* heatMap){
	
	// Initialize kernel size --------------------------------------------------
	int blocks = 1;
	int th_per_block = TH_PER_BLOCK;
	int threads = blocks*th_per_block;
	
	
	// Initialize clock --------------------------------------------------------
	clock_t test_start = clock();
	clock_t kernel_start;
	
	
	// Copy Integral Image to device -------------------------------------------
	float* intImg_d;
	cudaMalloc(&intImg_d, rows*cols*sizeof(float));
	cudaMemcpy(intImg_d, intImg, rows*cols*sizeof(float), cudaMemcpyHostToDevice);
	checkCUDAError("integral image");
	
	
	// Copy Heat Map to device -------------------------------------------
	float* heatMap_d;
	// cudaMalloc(&heatMap_d, rows*cols*sizeof(float));
	// cudaMemcpy(heatMap_d, heatMap, rows*cols*sizeof(float), cudaMemcpyHostToDevice);
	// checkCUDAError("heat map image");

	
	// Copy window offsets to device -------------------------------------------
	int* winOffsets_d;
	int nValidSubWindows = numWindows;
	
	cudaMalloc(&winOffsets_d, nValidSubWindows*sizeof(int));
	cudaMemcpy(winOffsets_d, winOffsets, nValidSubWindows*sizeof(int), cudaMemcpyHostToDevice);
	checkCUDAError("window offsets");
	
	
	// Initialize device 'boolean' face detected array -------------------------
	int* faceDetected_d[4];
	cudaMalloc(&faceDetected_d[0], nValidSubWindows*sizeof(int));
	cudaMalloc(&faceDetected_d[1], nValidSubWindows*sizeof(int));
	cudaMalloc(&faceDetected_d[2], nValidSubWindows*sizeof(int));
	cudaMalloc(&faceDetected_d[3], nValidSubWindows*sizeof(int));
	
	
	cudaMemset(faceDetected_d[0], 0, nValidSubWindows*sizeof(int));
	cudaMemset(faceDetected_d[1], 0, nValidSubWindows*sizeof(int));
	cudaMemset(faceDetected_d[2], 0, nValidSubWindows*sizeof(int));
	cudaMemset(faceDetected_d[3], 0, nValidSubWindows*sizeof(int));
	
	checkCUDAError("bool array");
	
	
	// Initialize results array for debugging... -------------------------------
	float* results_d;
	// cudaMalloc(&results_d, nValidSubWindows*sizeof(float));
	// cudaMemset(results_d, 0, nValidSubWindows*sizeof(float));
	// checkCUDAError("debug results");
	
	
	
	//==========================================================================
	// Run kernels -----------------------------------------------------------------
	kernel_heading("All Kernels", blocks, th_per_block, threads, nValidSubWindows);
	kernel_start = clock();
	for(int i = 2; i < 2+N_SCALES; ++i){
		ID1kernel<<<blocks, th_per_block>>>(intImg_d, 					// Itegral Image
														stride, 						//	Stride
														winOffsets_d, 				//	Sub-Window Offsets
														winSize, 					//	Sub-Window Size
														nValidSubWindows, 		//	Number of Sub Windows
														winSize/(5*(i)), 			// Scale of the feature
														faceDetected_d[0], 			//	Array to hold if a face was detected
														results_d,					//	Array to hold maximum feature value for each sub window
														heatMap_d					// Heat map
														);
														
		ID2kernel<<<blocks, th_per_block>>>(intImg_d, 					// Itegral Image
														stride, 						//	Stride
														winOffsets_d, 				//	Sub-Window Offsets
														winSize, 					//	Sub-Window Size
														nValidSubWindows, 		//	Number of Sub Windows
														winSize/(5*(i)), 			// Scale of the feature
														faceDetected_d[1], 			//	Array to hold if a face was detected
														results_d,					//	Array to hold maximum feature value for each sub window
														heatMap_d					// Heat map
														);
														
		ID3kernel<<<blocks, th_per_block>>>(intImg_d, 					// Itegral Image
														stride, 						//	Stride
														winOffsets_d, 				//	Sub-Window Offsets
														winSize, 					//	Sub-Window Size
														nValidSubWindows, 		//	Number of Sub Windows
														winSize/(5*(i)), 			// Scale of the feature
														faceDetected_d[2], 			//	Array to hold if a face was detected
														results_d,					//	Array to hold maximum feature value for each sub window
														heatMap_d					// Heat map
														);

		ID4kernel<<<blocks, th_per_block>>>(intImg_d, 					// Itegral Image
														stride, 						//	Stride
														winOffsets_d, 				//	Sub-Window Offsets
														winSize, 					//	Sub-Window Size
														nValidSubWindows, 		//	Number of Sub Windows
														winSize/(5*(i)), 			// Scale of the feature
														faceDetected_d[3], 			//	Array to hold if a face was detected
														results_d,					//	Array to hold maximum feature value for each sub window
														heatMap_d					// Heat map
														);
	}
	kernel_footer("All kernels", kernel_start);

	
	
	
	
	// Results -----------------------------------------------------------------
	printf("\n\nResults\n\n");
	printf("Completed test in %f seconds\n", ((double)clock() - test_start) / CLOCKS_PER_SEC);
	if(nValidSubWindows > 0){
		printf("A face was detected\n");
	}
	
	
	// Cleanup -----------------------------------------------------------------
	cudaFree(intImg_d);
	// cudaFree(heatMap_d);
	cudaFree(winOffsets_d);
	cudaFree(faceDetected_d[0]);
	cudaFree(faceDetected_d[1]);
	cudaFree(faceDetected_d[2]);
	cudaFree(faceDetected_d[3]);
	
	
	// cudaFree(results_d);
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
			printf(" Well poo!!!");
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
	
	// Compute the length of compacted array
	int len = new_end - offsets_ptr;
	
	printf("Possible faces: %d\n\n", len);
	
	// Return the length of the compacted array
	return len;
}



void kernel_heading(char* heading, int blocks, int th_per_block, int threads, int nValidSubWindows){
	// printf("\n\n");
	printf("Running %s --------\n", heading);
	// printf("Blocks:   %d\n", blocks);
	// printf("Th/Block: %d\n", th_per_block);
	// printf("Threads:  %d\n", threads);
	// printf("Windows:  %d\n", nValidSubWindows);
}

void kernel_footer(char* msg, clock_t kernel_start){
	cudaThreadSynchronize();
	printf("%s completed in %f seconds\n", msg, ((double)clock() - kernel_start) / CLOCKS_PER_SEC);
	checkCUDAError(msg);
}





