/*
	Shader test.
	Compile the shader from file.
*/
TEST(ShaderTest, CompileFromFile)
{
	try
	{
		GLTestWindow window(true);
		GLContextParam param;

		// Direct host pointer access of glVertexAttribPointer
		// is not valid in the core profile.
		param.UseCoreProfile = false;
		param.DebugMode = true;

		GLContext context(window.Handle(), param);
		GLUtil::EnableDebugOutput();

		// Quad
		glm::vec3 v[] =
		{
			glm::vec3( 0.5f,  0.5f, 0.0f),
			glm::vec3(-0.5f,  0.5f, 0.0f),
			glm::vec3(-0.5f, -0.5f, 0.0f),
			glm::vec3( 0.5f, -0.5f, 0.0f)
		};

		// Create shader
		GLShader shader;

		shader.Compile("./resources/shadertest_simple.vert");
		shader.Compile("./resources/shadertest_simple.frag");
		shader.Link();

		double start = GLTestUtil::CurrentTimeMilli();

		while (window.ProcessEvent())
		{
			double elapsed = GLTestUtil::CurrentTimeMilli() - start;

			window.SetTitle((boost::format("Elapsed time: %.2f") % (elapsed / 1000.0)).str());
			if (elapsed >= 500.0)
			{
				break;
			}

			glClear(GL_COLOR_BUFFER_BIT);

			shader.Begin();
			glEnableVertexAttribArray(0);
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, &v[0].x);
			glDrawArrays(GL_QUADS, 0, 4);
			glDisableVertexAttribArray(0);
			shader.End();

			context.SwapBuffers();
		}
	}
	catch (const GLException& e)
	{
		FAIL() << GLTestUtil::PrintGLException(e);
	}
}

/*
	Shader test.
	Compile the shader from file 2.
*/
TEST(ShaderTest, CompileFromFile2)
{
	try
	{
		GLTestWindow window(true);
		GLContextParam param;

		// Direct host pointer access of glVertexAttribPointer
		// is not valid in the core profile.
		param.UseCoreProfile = false;
		param.DebugMode = true;

		GLContext context(window.Handle(), param);
		GLUtil::EnableDebugOutput();

		// Quad
		glm::vec3 v[] =
		{
			glm::vec3( 0.5f,  0.5f, 0.0f),
			glm::vec3(-0.5f,  0.5f, 0.0f),
			glm::vec3(-0.5f, -0.5f, 0.0f),
			glm::vec3( 0.5f, -0.5f, 0.0f)
		};

		// Create shader
		GLShader shader;

		shader.Compile(GLShader::VertexShader, "./resources/shadertest_simple.vert");
		shader.Compile(GLShader::FragmentShader, "./resources/shadertest_simple.frag");
		shader.Link();

		double start = GLTestUtil::CurrentTimeMilli();

		while (window.ProcessEvent())
		{
			double elapsed = GLTestUtil::CurrentTimeMilli() - start;

			window.SetTitle((boost::format("Elapsed time: %.2f") % (elapsed / 1000.0)).str());
			if (elapsed >= 500.0)
			{
				break;
			}

			glClear(GL_COLOR_BUFFER_BIT);

			shader.Begin();
			glEnableVertexAttribArray(0);
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, &v[0].x);
			glDrawArrays(GL_QUADS, 0, 4);
			glDisableVertexAttribArray(0);
			shader.End();

			context.SwapBuffers();
		}
	}
	catch (const GLException& e)
	{
		FAIL() << GLTestUtil::PrintGLException(e);
	}
}

namespace
{
	std::string shadertestVs =
		"#version 420 core\n"
		"layout (location = 0) in vec3 position;\n"
		"void main() {\n"
		"	gl_Position = vec4(position, 1.0);\n"
		"}";

	std::string shadertestFs =
		"#version 420 core\n"
		"out vec4 fragColor;\n"
		"void main() {\n"
		"	fragColor = vec4(1.0, 0.0, 0.0, 1.0);\n"
		"}";
}

/*
	Shader test.
	Compile the shader from string.
*/
TEST(ShaderTest, CompileFromString)
{
	try
	{
		GLTestWindow window(true);
		GLContextParam param;

		// Direct host pointer access of glVertexAttribPointer
		// is not valid in the core profile.
		param.UseCoreProfile = false;
		param.DebugMode = true;

		GLContext context(window.Handle(), param);
		GLUtil::EnableDebugOutput();

		// Quad
		glm::vec3 v[] =
		{
			glm::vec3( 0.5f,  0.5f, 0.0f),
			glm::vec3(-0.5f,  0.5f, 0.0f),
			glm::vec3(-0.5f, -0.5f, 0.0f),
			glm::vec3( 0.5f, -0.5f, 0.0f)
		};

		// Create shader
		GLShader shader;

		shader.CompileString(GLShader::VertexShader, shadertestVs);
		shader.CompileString(GLShader::FragmentShader, shadertestFs);
		shader.Link();

		double start = GLTestUtil::CurrentTimeMilli();

		while (window.ProcessEvent())
		{
			double elapsed = GLTestUtil::CurrentTimeMilli() - start;

			window.SetTitle((boost::format("Elapsed time: %.2f") % (elapsed / 1000.0)).str());
			if (elapsed >= 500.0)
			{
				break;
			}

			glClear(GL_COLOR_BUFFER_BIT);

			shader.Begin();
			glEnableVertexAttribArray(0);
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, &v[0].x);
			glDrawArrays(GL_QUADS, 0, 4);
			glDisableVertexAttribArray(0);
			shader.End();

			context.SwapBuffers();
		}
	}
	catch (const GLException& e)
	{
		FAIL() << GLTestUtil::PrintGLException(e);
	}
}

/*
	Shader test.
	Shader parameters.
*/
TEST(ShaderTest, ShaderParameter)
{
	try
	{
		GLTestWindow window(true);
		GLContextParam param;

		// Direct host pointer access of glVertexAttribPointer
		// is not valid in the core profile.
		param.UseCoreProfile = false;
		param.DebugMode = true;

		GLContext context(window.Handle(), param);
		GLUtil::EnableDebugOutput();

		// Create shader
		GLShader shader;

		shader.Compile("./resources/shadertest_param.vert");
		shader.Compile("./resources/shadertest_param.geom");
		shader.Compile("./resources/shadertest_param.frag");
		shader.Link();

		double start = GLTestUtil::CurrentTimeMilli();

		while (window.ProcessEvent())
		{
			double elapsed = GLTestUtil::CurrentTimeMilli() - start;

			window.SetTitle((boost::format("Elapsed time: %.2f") % (elapsed / 1000.0)).str());
			if (elapsed >= 500.0)
			{
				break;
			}

			glClear(GL_COLOR_BUFFER_BIT);

			shader.Begin();
			shader.SetUniform("mvMatrix", glm::mat4(1.0f));
			shader.SetUniform("projectionMatrix", glm::mat4(1.0f));
			shader.SetUniform("size", 1.0f);
			glBegin(GL_POINTS);
			glVertexAttrib3fv(0, glm::value_ptr(glm::vec3(0.0f)));
			glEnd();
			shader.End();

			context.SwapBuffers();
		}
	}
	catch (const GLException& e)
	{
		FAIL() << GLTestUtil::PrintGLException(e);
	}
}
