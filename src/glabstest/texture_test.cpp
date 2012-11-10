/*
	Texture test.
	Simple 2D texture.
*/
TEST(TextureTest, Simple2DTexture)
{
	try
	{
		GLTestWindow window(true);
		window.SetTitle("TextureTest_Simple2DTexture");

		GLContextParam param;
		param.DebugMode = true;

		GLContext context(window.Handle(), param);
		GLUtil::EnableDebugOutput();

		// ------------------------------------------------------------

		GLShader shader;
		shader.Compile("../resources/texturetest_simple2d.vert");
		shader.Compile("../resources/texturetest_simple2d.frag");
		shader.Link();

		// ------------------------------------------------------------

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

		GLTexture2D texture;
		float data[10 * 10 * 3];

		for (int i = 0; i < 10; i++)
		{
			for (int j = 0; j < 10; j++)
			{
				int idx = (i * 10 + j) * 3;
				data[idx    ] = (float)i / 10.0f;
				data[idx + 1] = (float)j / 10.0f;
				data[idx + 2] = 0;
			}
		}

		texture.SetWrap(GL_CLAMP_TO_EDGE);
		texture.Allocate(10, 10, GL_RGBA8, GL_RGB, GL_FLOAT, data);

		// ------------------------------------------------------------

		double start = GLTestUtil::CurrentTimeMilli();

		while (window.ProcessEvent())
		{
			double elapsed = GLTestUtil::CurrentTimeMilli() - start;
			if (elapsed >= 500.0)
			{
				break;
			}

			glClear(GL_COLOR_BUFFER_BIT);

			shader.Begin();
			shader.SetUniform("tex", 0);
			texture.Bind();
			vao.Draw(GL_TRIANGLES, &ibo);
			texture.Unbind();
			shader.End();

			context.SwapBuffers();
		}
	}
	catch (const GLException& e)
	{
		FAIL() << GLTestUtil::PrintGLException(e);
	}
}