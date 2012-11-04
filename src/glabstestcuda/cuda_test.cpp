#include "cuda_test_kernel.h"

/*
	CUDA test.
	Simple.
*/
TEST(CUDATest, Simple)
{
	try
	{
		cudaDeviceProp prop;

		int count;
		HandleCudaError(cudaGetDeviceCount(&count));
	
		for (int i=0; i< count; i++)
		{
			HandleCudaError(cudaGetDeviceProperties(&prop, i));

			printf( "   --- General Information for device %d ---\n", i );
			printf( "Name:  %s\n", prop.name );
			printf( "Compute capability:  %d.%d\n", prop.major, prop.minor );
			printf( "Clock rate:  %d\n", prop.clockRate );
			printf( "Device copy overlap:  " );

			if (prop.deviceOverlap)
				printf( "Enabled\n" );
			else
				printf( "Disabled\n");

			printf( "Kernel execution timeout :  " );
			if (prop.kernelExecTimeoutEnabled)
				printf( "Enabled\n" );
			else
				printf( "Disabled\n" );

			printf( "   --- Memory Information for device %d ---\n", i );
			printf( "Total global mem:  %ld\n", prop.totalGlobalMem );
			printf( "Total constant Mem:  %ld\n", prop.totalConstMem );
			printf( "Max mem pitch:  %ld\n", prop.memPitch );
			printf( "Texture Alignment:  %ld\n", prop.textureAlignment );

			printf( "   --- MP Information for device %d ---\n", i );
			printf( "Multiprocessor count:  %d\n", prop.multiProcessorCount );
			printf( "Shared mem per mp:  %ld\n", prop.sharedMemPerBlock );
			printf( "Registers per mp:  %d\n", prop.regsPerBlock );
			printf( "Threads in warp:  %d\n", prop.warpSize );
			printf( "Max threads per block:  %d\n", prop.maxThreadsPerBlock );
			printf( "Max thread dimensions:  (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2] );
			printf( "Max grid dimensions:  (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2] );
			printf( "\n" );
		}

		cudaDeviceReset();
	}
	catch (const GLException& e)
	{
		FAIL() << GLTestUtil::PrintGLException(e);
	}
}

/*
	CUDA test.
	Simple 2.
*/
TEST(CUDATest, Simple2)
{
	try
	{
		int c;
		int *dev_c;

		HandleCudaError(cudaMalloc((void**)&dev_c, sizeof(int)));
		Run_AddKernel(2, 7, dev_c);
		HandleCudaError(cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost));

		printf( "2 + 7 = %d\n", c );

		HandleCudaError(cudaFree(dev_c));
		cudaDeviceReset();
	}
	catch (const GLException& e)
	{
		FAIL() << GLTestUtil::PrintGLException(e);
	}
}
