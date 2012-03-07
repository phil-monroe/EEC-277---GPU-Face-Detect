// checkCUDAError -------------------------------------------------------------
//		Convience method to check for cuda errors.
//		@param msg - Unique identifier to help debug.
//
//		From Dr Dobbs "CUDA: Supercomputing for the masses, Part 3"
//		http://drdobbs.com/architecture-and-design/207200659      
//-----------------------------------------------------------------------------
void checkCUDAError(const char *msg) {
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }                         
}