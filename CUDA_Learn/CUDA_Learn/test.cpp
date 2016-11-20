#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <assert.h>
#include <random>
#include <ctime>
#include "utils.h"

// forward declaration
int displayGPUproperty(int GPUidx=0);
void cudaVecAdd(int* a, int* b, int* c, int size);
void cudaStencialAdd(int* a, int *c, int size);
void cudaStencialAdd_optimized(int* a, int *c, int size);

void vector_add();
void stencil_add();

int main(void)  {
	
	// display the property of the GPU on the machine
	displayGPUproperty();
	
	// running vector_add example
	// vector_add();
	
	// running stencil_add example
	// stencil_add();

	system("pause");
}

/**
 * This program parallelize a simple CUDA vector_add on GPU. It also demonstrate the typical overhead of GPU accleration.
 */
void vector_add() {

	int size = 10000000;
	std::cout << std::endl;
	std::cout << "performing vector add of " << size << " elements..." << std::endl;

	// initialize input data
	int *a, *b, *c, *d;
	a = (int *)malloc(size * sizeof(int));
	b = (int *)malloc(size * sizeof(int));
	c = (int *)malloc(size * sizeof(int));
	d = (int *)malloc(size * sizeof(int));
	rand_int(a, size);
	rand_int(b, size);

	clock_t start, end, start0, end0;

	// start timing the CUDA implementation
	CUDA_timing(start);
	{
		int *device_a, *device_b, *device_c;
		
		// allocate device memory
		CUDA_timing(start0);
		safe_cuda(cudaMalloc(&device_a, size * sizeof(int)));
		safe_cuda(cudaMalloc(&device_b, size * sizeof(int)));
		safe_cuda(cudaMalloc(&device_c, size * sizeof(int)));
		CUDA_timing(end0);
		std::cout << "- GPU memory allocation time spent: " << TIME_ELAPSED(start0, end0) << " ms" << std::endl;

		// copy data from host to device
		CUDA_timing(start0);
		safe_cuda(cudaMemcpy(device_a, a, size * sizeof(int), cudaMemcpyHostToDevice));
		safe_cuda(cudaMemcpy(device_b, b, size * sizeof(int), cudaMemcpyHostToDevice));
		CUDA_timing(end0);
		std::cout << "- GPU host->device memory copy time spent: " << TIME_ELAPSED(start0, end0) << " ms" << std::endl;

		// launch our CUDA program
		CUDA_timing(start0);
		cudaVecAdd(device_a, device_b, device_c, size);
		CUDA_timing(end0);
		std::cout << "- GPU execution time spent: " << TIME_ELAPSED(start0, end0) << " ms" << std::endl;

		// copy from device to host
		CUDA_timing(start0);
		safe_cuda(cudaMemcpy(c, device_c, size * sizeof(int), cudaMemcpyDeviceToHost));
		CUDA_timing(end0);
		std::cout << "- GPU device->host memory copy time spent: " << TIME_ELAPSED(start0, end0) << " ms" << std::endl;

		// clean-up device memory
		CUDA_timing(start0);
		safe_cuda(cudaFree(device_a));
		safe_cuda(cudaFree(device_b));
		safe_cuda(cudaFree(device_c));
		CUDA_timing(end0);
		std::cout << "- GPU device memory release time spent: " << TIME_ELAPSED(start0, end0) << " ms" << std::endl;
	}
	// end timing
	CUDA_timing(end);
	std::cout << "GPU total time spent: " << TIME_ELAPSED(start, end) << " ms" << std::endl;

	// start timing the CPU implementation
	start = clock();
	{
		for (int i = 0; i < size; i++) {
			d[i] = a[i] + b[i];
		}
	}
	end = clock();
	std::cout << "CPU implementation time spent: " << TIME_ELAPSED(start, end) << " ms" << std::endl;

	// it looks surprised at first glance to observe that the GPU implementation is much more time-consuming than
	// the CPU version. The main reason is that the program is memory bandwidth bound in this example: it spends much
	// more time copying memory from host to device and the other way around. The actual execution time on GPU is relatively
	// small. This example teaches us that before parallelizing the program with GPU, we need to understand the workload of the
	// application first and answer the following question: is it compute-bound or memory bound when deployed on GPU? For applications
	// whose computation part is relatively simple, parallelizing it on GPU might not worth the overhead. GPU is more suitable for accelerating
	// application with massive and extensive computation.

	// check result	
	bool success = true;
	for (int i = 0; i < size; i++) {
		if (c[i] != d[i]) {
			success = false;
			std::cout << "element at position " << i << " is not correct: " << "should be " << a[i] + b[i] << " ,get " << c[i] << std::endl;
			break;
		}
	}
	if (!success) std::cout << "The GPU implementation is not correct!" << std::endl;

	// clean-up host memory
	free(a); free(b); free(c); free(d);
}

/**
 * This demo demonstrate the usage of shared memory in CUDA. Shared memory is a chunk of on-chip (per sm) memory
 * that is extremely fast. In this example, each element in the input array is loaded by (2*RADIUS+1) threads. We
 * can optimize it by first loading the elements need per block into the shared memory. Notice that the optimized
 * version is significantly faster than the naive implementation.
 */
void stencil_add() {
	int size = 10000000;

	std::cout << std::endl;
	std::cout << "performing stencil add of " << size << " elements..." << std::endl;

	int *a, *c;
	a = (int *)malloc(size * sizeof(int));
	c = (int *)malloc(size * sizeof(int));
	rand_int(a, size);
	rand_int(c, size);

	{
		clock_t start, end;
		int *device_a, *device_c;
		safe_cuda(cudaMalloc(&device_a, size * sizeof(int)));
		safe_cuda(cudaMalloc(&device_c, size * sizeof(int)));
		safe_cuda(cudaMemcpy(device_a, a, size * sizeof(int), cudaMemcpyHostToDevice));
		CUDA_timing(start);
		cudaStencialAdd(device_a, device_c, size);
		CUDA_timing(end);
		safe_cuda(cudaMemcpy(c, device_c, size * sizeof(int), cudaMemcpyDeviceToHost));
		safe_cuda(cudaFree(device_a));
		safe_cuda(cudaFree(device_c));
		std::cout << "[cudaStencialAdd] GPU execution time spent: " << TIME_ELAPSED(start, end) << " ms" << std::endl;
	}


	{
		clock_t start, end;
		int *device_a, *device_c;
		safe_cuda(cudaMalloc(&device_a, size * sizeof(int)));
		safe_cuda(cudaMalloc(&device_c, size * sizeof(int)));
		safe_cuda(cudaMemcpy(device_a, a, size * sizeof(int), cudaMemcpyHostToDevice));
		CUDA_timing(start);
		cudaStencialAdd_optimized(device_a, device_c, size);
		CUDA_timing(end);
		safe_cuda(cudaMemcpy(c, device_c, size * sizeof(int), cudaMemcpyDeviceToHost));
		safe_cuda(cudaFree(device_a));
		safe_cuda(cudaFree(device_c));
		std::cout << "[cudaStencialAdd_optimized] GPU execution time spent: " << TIME_ELAPSED(start, end) << " ms" << std::endl;
	}

	free(a); free(c);
}

