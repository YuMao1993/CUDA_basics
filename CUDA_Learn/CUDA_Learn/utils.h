#ifndef UTILS_H
#define UTILS_H

// safe_cuda is a convenient macro to wrap CUDA function invocation
#define safe_cuda(CODE)\
	    {\
  cudaError_t err = CODE;\
  if(err != cudaSuccess) {\
    std::cout<<"CUDA error:"<<cudaGetErrorString(err)<<std::endl;\
	system("pause"); \
	exit(-1); \
      }\
}

// CUDA_timing is a convenient macro for timing
#define CUDA_timing(CLOCK) cudaThreadSynchronize(); CLOCK = clock();

// TIME_ELAPSED is a convenient macro for computing the time elapsed (in ms) given the start clock and end clock 
#define TIME_ELAPSED(CLOCK1, CLOCK2) ((((double)(CLOCK2 - CLOCK1)) / CLOCKS_PER_SEC) * 1000.0)

/**
 * generate random integers in an array
 */
void rand_int(int *dst, int size) {
	for (int i = 0; i < size; i++) dst[i] = rand();
}

#endif