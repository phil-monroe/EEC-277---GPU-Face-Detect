#ifndef _CUDA_DETECT_FACES_
#define _CUDA_DETECT_FACES_


extern "C" void cuda_detect_faces(float* intImg, int rows, int cols, size_t stride, int* windowOffsets, int numWindows, int windowSize);

#endif // _CUDA_DETECT_FACES_