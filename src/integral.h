#ifndef _INTEGRAL_H_
#define _INTEGRAL_H_


extern "C" void cuda_integrate_image(float* data, int rows, int cols, size_t stride);

#endif // _INTEGRAL_H_