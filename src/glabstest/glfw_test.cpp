/*
	GLFW test.
	Simple.
*/
TEST(GLFWTest, Simple)
{
	try
	{
		if (!glfwInit())
		{
			THROW_GL_EXCEPTION(GLException::RunTimeError, "glfwInit");
		}

		glfwOpenWindowHint(GLFW_OPENGL_VERSION_MAJOR, 4);
		glfwOpenWindowHint(GLFW_OPENGL_VERSION_MINOR, 2);
		glfwOpenWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
		glfwOpenWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
		glfwOpenWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);

		if (!glfwOpenWindow(1280, 720, 0, 0, 0, 0, 0, 0, GLFW_WINDOW))
		{
			glfwTerminate();
			THROW_GL_EXCEPTION(GLException::RunTimeError, "glfwOpenWindow");
		}

		GLUtil::InitializeGlew();
		GLUtil::EnableDebugOutput();

		glfwTerminate();
	}
	catch (const GLException& e)
	{
		FAIL() << GLTestUtil::PrintGLException(e);
	}
}

/*
	GLFW test.
	Simple 2.
*/
TEST(GLFWTest, Simple2)
{
	try
	{
		if (!glfwInit())
		{
			THROW_GL_EXCEPTION(GLException::RunTimeError, "glfwInit");
		}

		glfwOpenWindowHint(GLFW_OPENGL_VERSION_MAJOR, 4);
		glfwOpenWindowHint(GLFW_OPENGL_VERSION_MINOR, 2);
		glfwOpenWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
		glfwOpenWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
		glfwOpenWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);

		if (!glfwOpenWindow(1280, 720, 0, 0, 0, 0, 0, 0, GLFW_WINDOW))
		{
			glfwTerminate();
			THROW_GL_EXCEPTION(GLException::RunTimeError, "glfwOpenWindow");
		}

		GLUtil::InitializeGlew();
		GLUtil::EnableDebugOutput();

		int running = 1;
		double start = GLTestUtil::CurrentTimeMilli();

		while (running)
		{
			double elapsed = GLTestUtil::CurrentTimeMilli() - start;

			if (elapsed >= 500.0)
			{
				break;
			}

			glfwSetWindowTitle((boost::format("Elapsed time: %.2f") % (elapsed / 1000.0)).str().c_str());

			glClearColor((float)elapsed / 1000.0f, 1.0f, 0.0f, 1.0f);
			glClear(GL_COLOR_BUFFER_BIT);
			glfwSwapBuffers();

			running = glfwGetWindowParam(GLFW_OPENED);
		}

		glfwTerminate();
	}
	catch (const GLException& e)
	{
		FAIL() << GLTestUtil::PrintGLException(e);
	}
}