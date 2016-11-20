#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <assert.h>

int getCUDACoresCount(cudaDeviceProp devProp);

int displayGPUproperty(int GPUidx)
{	
	std::cout << "----- GPU Properties -----" << std::endl;

	// check CUDA device
	int n = 0;
	cudaGetDeviceCount(&n);
	if (GPUidx >= n) {
		std::cout << "Number of GPUs found: " << n << std::endl;
		std::cout << "Please provide a valid GPU idx less than " << n << std::endl;
	}
	else {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, GPUidx);
		std::cout << "GPU name: " << prop.name << std::endl;
		std::cout << "number of GPU cores: " << prop.multiProcessorCount << std::endl;						// Just for reference, GTX980 has 16 SMs, GTX980Ti has 22 SMs, 1080 has 20 SMs.
		std::cout << "ALUs per SM: " << getCUDACoresCount(prop) / prop.multiProcessorCount << std::endl;	// This is equal to (32-wide SIMD unit count) * 32.
		std::cout << "CUDA cores: " << getCUDACoresCount(prop) << std::endl;								// maximum concurrent CUDA threads, a.k.a total number of ALUs on the GPU.
		std::cout << "shared memory per block: " << prop.sharedMemPerBlock / 1024 << "KB" << std::endl;
		std::cout << "warp size: " << prop.warpSize << std::endl;
		std::cout << "warp context per SM: " << prop.maxThreadsPerBlock / prop.warpSize << std::endl;		// just for reference, GTX980 is capable of storing 64 warp execution context on each SM.
		std::cout << "max threads per block: " << prop.maxThreadsPerBlock << std::endl;						// each CUDA block is scheduled entirely on a single SM, so there is a limit on the number of CUDA threads per block. It is (the max number of warp execution contexts on each SM x the size of each warp).
	}
	std::cout << "--------------------------" << std::endl;
	return 0;
}

int getCUDACoresCount(cudaDeviceProp devProp)
{
	int cores = 0;
	int mp = devProp.multiProcessorCount;
	switch (devProp.major){
	case 2: // Fermi
		// http://www.nvidia.com/content/PDF/fermi_white_papers/NVIDIA_Fermi_Compute_Architecture_Whitepaper.pdf
		if (devProp.minor == 1) cores = mp * 48;
		else cores = mp * 32;
		break;
	case 3: // Kepler
		// http://www.nvidia.com/content/PDF/kepler/NVIDIA-kepler-GK110-Architecture-Whitepaper.pdf
		cores = mp * 192;
		break;
	case 5: // Maxwell
		// http://international.download.nvidia.com/geforce-com/international/pdfs/GeForce_GTX_980_Whitepaper_FINAL.PDF
		cores = mp * 128;
		break;
	case 6: // Pascal
		// https://images.nvidia.com/content/pdf/tesla/whitepaper/pascal-architecture-whitepaper.pdf
		if (devProp.minor == 1) cores = mp * 128;
		else if (devProp.minor == 0) cores = mp * 64;
		else printf("Unknown device type\n");
		break;
	default:
		printf("Unknown device type\n");
		break;
	}
	return cores;
}
