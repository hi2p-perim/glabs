#ifndef __GL_INTEROP_TEST_KERNEL_H__
#define __GL_INTEROP_TEST_KERNEL_H__

extern "C" void Run_GLInteropTestJuliaKernel(int width, int height, int numSMs, float xcparam, float ycparam, uchar4* dst);

#endif // __GL_INTEROP_TEST_KERNEL_H__