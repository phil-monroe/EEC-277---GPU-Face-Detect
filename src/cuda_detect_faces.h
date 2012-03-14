#ifndef _CUDA_DETECT_FACES_
#define _CUDA_DETECT_FACES_


void cuda_detect_faces(float* intImg, int rows, int cols, size_t stride, int* windowOffsets, int numWindows, int windowSize, float* heatMap);
void cuda_detect_faces2(float* intImg, int rows, int cols, size_t stride, int* windowOffsets, int numWindows, int windowSize, float* heatMap);

#endif // _CUDA_DETECT_FACES_