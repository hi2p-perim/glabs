#ifndef __CUDA_UTIL_H__
#define __CUDA_UTIL_H__

class CUDAUtil
{
private:

	CUDAUtil();
	DISALLOW_COPY_AND_ASSIGN(CUDAUtil);

public:

	static void HandleError(cudaError_t err, const char* file, const char* function, int line);

};

#define HandleCudaError(err) (CUDAUtil::HandleError(err, __FILE__, __FUNCTION__, __LINE__))

#endif // __CUDA_UTIL_H__