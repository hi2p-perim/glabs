#include "cuda_test_kernel.h"

__device__ int AddFunc(int a, int b)
{
	return a + b;
}

__global__ void AddKernel( int a, int b, int* c )
{
	*c = AddFunc(a, b);
}

extern "C" void Run_AddKernel( int a, int b, int* c )
{
	AddKernel<<<1, 1>>>(a, b, c);
}
