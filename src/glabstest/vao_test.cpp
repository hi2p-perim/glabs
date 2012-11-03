/*
	Vertex array object test.
	Simple.
*/
TEST(VAOTest, Simple)
{
	try
	{
		GLTestWindow window(true);
		GLContextParam param;

		param.DebugMode = true;

		GLContext context(window.Handle(), param);
		GLUtil::EnableDebugOutput();

		// ------------------------------------------------------------

		GLShader shader;
		shader.Compile("./resources/shadertest_simple.vert");
		shader.Compile("./resources/shadertest_simple.frag");
		shader.Link();

		GLVertexArray vao;
		GLVertexBuffer positionVbo;

		glm::vec3 v[] =
		{
			glm::vec3( 0.5f,  0.5f, 0.0f),
			glm::vec3(-0.5f,  0.5f, 0.0f),
			glm::vec3(-0.5f, -0.5f, 0.0f),
			glm::vec3( 0.5f, -0.5f, 0.0f)
		};

		positionVbo.AddStatic(12, &v[0].x);
		vao.Add(GLDefaultVertexAttribute::Position, &positionVbo);

		// ------------------------------------------------------------

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
			vao.Draw(GL_TRIANGLES, 3);
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
	Vertex array object test.
	Simple 2.
*/
TEST(VAOTest, Simple2)
{
	try
	{
		GLTestWindow window(true);
		GLContextParam param;

		param.DebugMode = true;

		GLContext context(window.Handle(), param);
		GLUtil::EnableDebugOutput();

		// ------------------------------------------------------------

		GLShader shader;
		shader.Compile("./resources/shadertest_simple.vert");
		shader.Compile("./resources/shadertest_simple.frag");
		shader.Link();

		GLVertexArray vao;
		GLVertexBuffer positionVbo;
		GLIndexBuffer ibo;

		glm::vec3 v[] =
		{
			glm::vec3( 0.5f,  0.5f, 0.0f),
			glm::vec3(-0.5f,  0.5f, 0.0f),
			glm::vec3(-0.5f, -0.5f, 0.0f),
			glm::vec3( 0.5f, -0.5f, 0.0f)
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
			vao.Draw(GL_TRIANGLES, &ibo);
			shader.End();

			context.SwapBuffers();
		}
	}
	catch (const GLException& e)
	{
		FAIL() << GLTestUtil::PrintGLException(e);
	}
}

