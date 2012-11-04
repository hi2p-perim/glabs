#include "glinterop_test_kernel.h"

__device__ unsigned int blockCounter;

__device__ float Julia(float x, float y, float xcparam, float ycparam)
{
	float xx = x * x;
	float yy = y * y;
	float xc = xcparam;//-0.8;
	float yc = ycparam;//0.156;

	int i = 0;

	for (i = 0; i < 256; i++)
	{
		if (xx + yy > 4.0)
		{
			return 1.0 - (float)i / 256.0;
		}
		
		y = x * y * 2.0 + yc;
		x = xx - yy + xc;
		yy = y * y;
		xx = x * x;
	}

	return 0.0;
}

__global__ void GLInteropTestJuliaKernel(int width, int height, int gridWidth, int gridNum, float xcparam, float ycparam, uchar4* dst)
{
	__shared__ unsigned int blockIndex;
	__shared__ unsigned int blockX;
	__shared__ unsigned int blockY;

	while (1)
	{
		if (threadIdx.x == 0 && threadIdx.y == 0)
		{
			blockIndex = atomicAdd(&blockCounter, 1);
			blockX = blockIndex % gridWidth;
			blockY = blockIndex / gridWidth;
		}

		__syncthreads();

		if (blockIndex >= gridNum)
		{
			break;
		}

		int ix = blockDim.x * blockX + threadIdx.x;
		int iy = blockDim.y * blockY + threadIdx.y;

		if (ix < width && iy < height)
		{
			const float scale = 1.5;

			// Scale to [-scale, scale]^2
			float x = scale * (float)(width / 2 - ix) / (width / 2);
			float y = scale * (float)(height / 2 - iy) / (width / 2);
			
			float m = Julia(x, y, xcparam, ycparam);
			
			uchar4 color;
			color.x = 255 * m;
			color.y = 255 * m;
			color.z = 255 * m;
			color.w = 255;

			int offset = width * iy + ix;
			dst[offset] = color;
		}
	}
}

inline int iDivUp(int a, int b)
{
	return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

extern "C" void Run_GLInteropTestJuliaKernel(int width, int height, int numSMs, float xcparam, float ycparam, uchar4* dst)
{
	const int threadDim = 24;
	dim3 threads(threadDim, threadDim);
	dim3 grid(iDivUp(width, threadDim), iDivUp(height, threadDim));
	
	unsigned int blockCounterHost = 0;
	cudaMemcpyToSymbol(blockCounter, &blockCounterHost, sizeof(unsigned int), 0);

	GLInteropTestJuliaKernel<<<numSMs, threads>>>(width, height, grid.x, grid.x * grid.y, xcparam, ycparam, dst);
}
