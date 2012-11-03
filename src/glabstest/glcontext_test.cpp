/*
	GL context test.
	Simple.
*/
TEST(GLContextTest, Simple)
{
	try
	{
		GLTestWindow window;
		GLContext context(window.Handle());
	}
	catch (const GLException& e)
	{
		FAIL() << GLTestUtil::PrintGLException(e);
	}
}

/*
	GL context test.
	Simple 2.
*/
TEST(GLContextTest, Simple2)
{
	try
	{
		GLTestWindow window(true);
		GLContext context(window.Handle());
	}
	catch (const GLException& e)
	{
		FAIL() << GLTestUtil::PrintGLException(e);
	}
}

/*
	GL context test.
	Simple 3.
*/
TEST(GLContextTest, Simple3)
{
	try
	{
		GLTestWindow window(true);
		GLContext context(window.Handle());

		double start = GLTestUtil::CurrentTimeMilli();

		while (window.ProcessEvent())
		{
			double elapsed = GLTestUtil::CurrentTimeMilli() - start;

			if (elapsed >= 100.0)
			{
				break;
			}

			window.SetTitle((boost::format("Elapsed time: %.2f") % (elapsed / 1000.0)).str());
		}
	}
	catch (const GLException& e)
	{
		FAIL() << GLTestUtil::PrintGLException(e);
	}
}

/*
	GL context test.
	Check version.
*/
TEST(GLContextTest, CheckVersion)
{
	// GL3.2 - GL4.2
	int majorVersions[] = { 3, 3, 4, 4, 4 };
	int minorVersions[] = { 2, 3, 0, 1, 2 };

	try
	{
		for (int i = 0; i < 5; i++)
		{
			GLTestWindow window;
			GLContextParam param;

			param.ColorBits = 32;
			param.DepthBits = 24;
			param.StencilBits = 8;
			param.MajorVersion = majorVersions[i];
			param.MinorVersion = minorVersions[i];
			param.ForwardCompatible = true;
			param.DebugMode = false;
			param.UseCoreProfile = true;

			GLContext context(window.Handle(), param);
		}
	}
	catch (const GLException& e)
	{
		FAIL() << GLTestUtil::PrintGLException(e);
	}
}

/*
	GL context test.
	Multisample.
*/
TEST(GLContextTest, Multisample)
{
	try
	{
		GLTestWindow window;
		GLContextParam param;

		param.ColorBits = 32;
		param.DepthBits = 24;
		param.StencilBits = 8;
		param.MajorVersion = 4;
		param.MinorVersion = 2;
		param.Multisample = 16;
		param.ForwardCompatible = true;
		param.DebugMode = false;
		param.UseCoreProfile = true;

		GLContext context(window.Handle(), param);
	}
	catch (const GLException& e)
	{
		FAIL() << GLTestUtil::PrintGLException(e);
	}
}