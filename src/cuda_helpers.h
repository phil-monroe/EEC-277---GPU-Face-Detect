#ifndef _CUDA_HELPERS_
#define _CUDA_HELPERS_

// checkCUDAError -------------------------------------------------------------
//		Convience method to check for cuda errors.
//		@param msg - Unique identifier to help debug.
//
//		From Dr Dobbs "CUDA: Supercomputing for the masses, Part 3"
//		http://drdobbs.com/architecture-and-design/207200659      
//-----------------------------------------------------------------------------
void checkCUDAError(const char *msg);

#endif // _CUDA_HELPERS_