#include "glinterop_test_kernel.h"

/*
	GL interop test.
	Julia.
*/
TEST(GLInteropTest, Julia)
{
	try
	{
		const int width = 1280;
		const int height = 720;
		const int bufWidth = 1920;
		const int bufHeight = 1080;

		// ------------------------------------------------------------

		GLTestWindow window(width, height, true);
		GLContextParam param;

		param.DebugMode = true;
		param.Multisample = 8;

		GLContext context(window.Handle(), param);
		GLUtil::EnableDebugOutput(GLUtil::DebugOutputFrequencyLow);

		// ------------------------------------------------------------

		// Choose device
		cudaDeviceProp prop;
		memset(&prop, 0, sizeof(cudaDeviceProp));
		prop.major = 2;
		prop.minor = 0;

		int devID;
		HandleCudaError(cudaChooseDevice(&devID, &prop));
		HandleCudaError(cudaGLSetGLDevice(devID));

		// Get properties
		HandleCudaError(cudaGetDeviceProperties(&prop, devID));

		// Create texture and PBO
		GLTexture2D texture;
		texture.SetMagFilter(GL_LINEAR);
		texture.SetMinFilter(GL_LINEAR);
		texture.SetWrap(GL_CLAMP_TO_EDGE);
		texture.Allocate(bufWidth, bufHeight, GL_RGBA8);
		
		GLPixelUnpackBuffer pbo;
		pbo.Allocate(bufWidth * bufHeight * 4, NULL, GL_DYNAMIC_DRAW);

		// Register
		cudaGraphicsResource* cudaPbo;
		HandleCudaError(cudaGraphicsGLRegisterBuffer(&cudaPbo, pbo.ID(), cudaGraphicsMapFlagsWriteDiscard));

		// ------------------------------------------------------------

		GLShader shader;
		shader.Compile("../resources/texturetest_simple2d.vert");
		shader.Compile("../resources/texturetest_simple2d.frag");
		shader.Link();

		GLVertexArray vao;
		GLVertexBuffer positionVbo;
		GLIndexBuffer ibo;

		glm::vec3 v[] =
		{
			glm::vec3( 1.0f,  1.0f, 0.0f),
			glm::vec3(-1.0f,  1.0f, 0.0f),
			glm::vec3(-1.0f, -1.0f, 0.0f),
			glm::vec3( 1.0f, -1.0f, 0.0f)
		};

		GLuint i[] =
		{
			0, 1, 2,
			2, 3, 0
		};

		positionVbo.AddStatic(12, &v[0].x);
		vao.Add(GLDefaultVertexAttribute::Position, &positionVbo);
		ibo.AddStatic(6, i);

		// ------------------------------------------------------------

		double fps = 0.0;
		double timeSum = 0.0;
		double prevTime = GLTestUtil::CurrentTimeMilli();
		int frameCount = 0;

		double start = GLTestUtil::CurrentTimeMilli();
		float xcparam = -0.8f;
		float ycparam = 0.165f;
		float inc = 0.001f;

		while (window.ProcessEvent())
		{
			// ------------------------------------------------------------

			double currentTime = GLTestUtil::CurrentTimeMilli();
			double elapsedTime = currentTime - prevTime;

			timeSum += elapsedTime;
			frameCount++;

			if (frameCount >= 13)
			{
				fps = 1000.0 * 13.0 / timeSum;
				timeSum = 0.0;
				frameCount = 0;
			}

			prevTime = currentTime;
			window.SetTitle((boost::format("GLInteropTest_Julia [FPS %.1f]") % fps).str());

			// ------------------------------------------------------------

			double elapsed = GLTestUtil::CurrentTimeMilli() - start;
			if (elapsed >= 1000.0)
			{
				break;
			}

			xcparam += inc;
			if (xcparam > -0.799f || xcparam < -0.811f) 
			{
				inc *= -1.0f;
			}

			// ------------------------------------------------------------

			HandleCudaError(cudaGraphicsMapResources(1, &cudaPbo, NULL));
			
			// Get device pointer
			uchar4* devPtr;
			size_t bufferSize;
			HandleCudaError(cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &bufferSize, cudaPbo));
			Run_GLInteropTestJuliaKernel(bufWidth, bufHeight, prop.multiProcessorCount, xcparam, ycparam, devPtr);
			HandleCudaError(cudaGraphicsUnmapResources(1, &cudaPbo, NULL));

			texture.Replace(&pbo, glm::ivec4(0, 0, bufWidth, bufHeight), GL_RGBA, GL_UNSIGNED_BYTE);

			// ------------------------------------------------------------

			glClearBufferfv(GL_COLOR, 0, glm::value_ptr(glm::vec4(0.0f)));
			glViewportIndexedfv(0, glm::value_ptr(glm::vec4(0, 0, width, height)));

			shader.Begin();
			shader.SetUniform("tex", 0);
			texture.Bind();
			vao.Draw(GL_TRIANGLES, &ibo);
			texture.Unbind();
			shader.End();

			context.SwapBuffers();
		}

		cudaDeviceReset();
	}
	catch (const GLException& e)
	{
		FAIL() << GLTestUtil::PrintGLException(e);
	}
}