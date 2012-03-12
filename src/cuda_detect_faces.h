#ifndef _CUDA_DETECT_FACES_
#define _CUDA_DETECT_FACES_


extern "C" void cuda_detect_faces(float* data, int rows, int cols, size_t stride, int* windowOffsets);

#endif // _CUDA_DETECT_FACES_