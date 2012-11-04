#include "cudautil.h"

void CUDAUtil::HandleError( cudaError_t err, const char* file, const char* function, int line )
{
	if (err != cudaSuccess)
	{
		throw GLException(
			GLException::RunTimeError,
			(boost::format("CUDA error : %s") % cudaGetErrorString(err)).str(),
			file, function, line, GLException::GetStackTrace());
	}
}
