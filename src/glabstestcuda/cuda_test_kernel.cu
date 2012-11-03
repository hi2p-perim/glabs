#include "cuda_test_kernel.h"

__device__ int AddFunc(int a, int b)
{
	return a + b;
}

__global__ void AddKernel( int a, int b, int* c )
{
	*c = AddFunc(a, b);
}

extern "C" void RunAddKernel( int a, int b, int* c )
{
	AddKernel<<<1, 1>>>(a, b, c);
}
