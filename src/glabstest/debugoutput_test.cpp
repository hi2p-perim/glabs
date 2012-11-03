/*
	Debug output test.
	Simple test of GL_ARB_debug_output.
*/
TEST(DebugOutputTest, Simple)
{
	try
	{
		GLTestWindow window;

		GLContextParam param;
		param.DebugMode = true;

		GLContext context(window.Handle(), param);
		GLUtil::EnableDebugOutput();
	}
	catch (const GLException& e)
	{
		FAIL() << GLTestUtil::PrintGLException(e);
	}
}

/*
	Debug output test.
	Raise an error on purpose.
*/
TEST(DebugOutputTest, ErrorTest)
{
	bool noError = true;

	try
	{
		GLTestWindow window;
		
		GLContextParam param;
		param.DebugMode = true;

		GLContext context(window.Handle(), param);
		GLUtil::EnableDebugOutput();

		// The line raises an error
		glEnable(GL_ALPHA);
	}
	catch (const GLException& e)
	{
		if (e.Type() == GLException::DebugOutput)
		{
			noError = false;
			SUCCEED() << GLTestUtil::PrintGLException(e);
		}
		else
		{
			FAIL() << GLTestUtil::PrintGLException(e);
		}
	}

	if (noError)
	{
		FAIL() << "No errors";
	}
}