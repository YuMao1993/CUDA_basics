#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define THREAD_PER_BLOCK 1024
#define RADIUS 1000

__global__ void cudaVecAddKernel(int* a, int* b, int* c, int size)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) {
		c[id] = a[id] + b[id];
	}
}

/**
 * an optimized stencial_add kernel with shared_memory
 * the motivation here is that each element is loaded by threads (2*radius+1) times.
 * we definitely want to optimize it.
 */
__global__ void cudaStencialAddKernel_optimized(int* a, int* c, int size)
{
	__shared__ int temp[THREAD_PER_BLOCK + RADIUS * 2];
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= size) return;

	// all threads within the block collaboratievly load the 
	// data required into the on-chip shared memory.
	int sid = threadIdx.x + RADIUS;
	temp[sid] = a[id];

	int offset1 = THREAD_PER_BLOCK + RADIUS;
	if (threadIdx.x < RADIUS) {
		int id1 = id - RADIUS;
		if (id1 >= 0) temp[threadIdx.x] = a[id1];
		int id2 = id + THREAD_PER_BLOCK;
		if (id2 < size) temp[threadIdx.x + offset1] = a[id2];
	}
	__syncthreads(); //synchronize to ensure all data are available

	// perfrom stencil add on the shared memory
	int offset2 = blockIdx.x * blockDim.x - RADIUS;
	for (int i = threadIdx.x; i <= threadIdx.x + RADIUS * 2; i++)
	{
		int g_i = offset2 + i;
		if (g_i < 0 || g_i >= size) continue;
		c[id] += temp[i];
	}
}

__global__ void cudaSencialAddKernel(int* a, int* c, int size)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) {
		for (int i = id - RADIUS; i <= id + RADIUS; i++)
		{
			if (i < 0 || i >= size) continue;
			c[id] += a[i];
		}
	}
}

void cudaVecAdd(int* a, int* b, int* c, int size)
{
	cudaVecAddKernel <<< (size + THREAD_PER_BLOCK - 1)/THREAD_PER_BLOCK, THREAD_PER_BLOCK >>>(a, b, c, size);
}

void cudaStencialAdd(int* a, int *c, int size)
{
	cudaSencialAddKernel <<< (size + THREAD_PER_BLOCK - 1)/THREAD_PER_BLOCK, THREAD_PER_BLOCK >>> (a, c, size);
}

void cudaStencialAdd_optimized(int* a, int *c, int size)
{
	cudaStencialAddKernel_optimized <<< (size + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK, THREAD_PER_BLOCK >>> (a, c, size);
}
