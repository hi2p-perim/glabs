#include <glabs/glabs.h>
#include <Windows.h>
#include <tchar.h>

void GLUtil::InitializeGlew(bool experimental)
{
	if (experimental)
	{
		// Some extensions e.g. GL_ARB_debug_output doesn't work
		// unless in the experimental mode.
		glewExperimental = GL_TRUE;
	}
	else
	{
		glewExperimental = GL_FALSE;
	}

	GLenum err = glewInit();

	if (err != GLEW_OK)
	{
		THROW_GL_EXCEPTION(GLException::RunTimeError, 
			(boost::format("Failed to initialize GLEW: %s") % glewGetErrorString(err)).str());
	}
}

void GLUtil::EnableDebugOutput(DebugOutputFrequency freq)
{
	// Initialize GL_ARB_debug_output
	if (GLEW_ARB_debug_output)
	{
		glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
		glDebugMessageControlARB(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, NULL, GL_TRUE);
		
		if (freq == DebugOutputFrequencyMedium)
		{
			glDebugMessageControlARB(GL_DONT_CARE, GL_DONT_CARE, GL_DEBUG_SEVERITY_LOW_ARB, 0, NULL, GL_FALSE);
		}
		else if (freq == DebugOutputFrequencyLow)
		{
			glDebugMessageControlARB(GL_DONT_CARE, GL_DONT_CARE, GL_DEBUG_SEVERITY_MEDIUM_ARB, 0, NULL, GL_FALSE);
			glDebugMessageControlARB(GL_DONT_CARE, GL_DONT_CARE, GL_DEBUG_SEVERITY_LOW_ARB, 0, NULL, GL_FALSE);
		}
		
		glDebugMessageCallbackARB(&GLUtil::DebugOutput, NULL);
	}
	else
	{
		THROW_GL_EXCEPTION(GLException::CapabilityError, "GL_ARB_debug_output is not supported.");
	}
}

bool GLUtil::CheckExtension( const std::string& name )
{
	GLint c;

	glGetIntegerv(GL_NUM_EXTENSIONS, &c);

	for (GLint i = 0; i < c; ++i)
	{
		std::string s = (char const*)glGetStringi(GL_EXTENSIONS, i);
		if (s == name)
		{
			return true;
		}
	}

	return false;
}

void APIENTRY GLUtil::DebugOutput( GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, GLvoid* userParam )
{
	std::string sourceString;
	std::string typeString;
	std::string severityString;

	switch (source)
	{
		case GL_DEBUG_SOURCE_API_ARB:
			sourceString = "OpenGL";
			break;

		case GL_DEBUG_SOURCE_WINDOW_SYSTEM_ARB:
			sourceString = "Windows";
			break;

		case GL_DEBUG_SOURCE_SHADER_COMPILER_ARB:
			sourceString = "Shader Compiler";
			break;

		case GL_DEBUG_SOURCE_THIRD_PARTY_ARB:
			sourceString = "Third Party";
			break;

		case GL_DEBUG_SOURCE_APPLICATION_ARB:
			sourceString = "Application";
			break;

		case GL_DEBUG_SOURCE_OTHER_ARB:
		default:
			sourceString = "Other";
			break;
	}

	switch (type)
	{
		case GL_DEBUG_TYPE_ERROR_ARB:
			typeString = "Error";
			break;

		case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR_ARB:
			typeString = "Deprecated behavior";
			break;

		case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR_ARB:
			typeString = "Undefined behavior";
			break;

		case GL_DEBUG_TYPE_PORTABILITY_ARB:
			typeString = "Portability";
			break;

		case GL_DEBUG_TYPE_OTHER_ARB:
		default:
			typeString = "Message";
			break;
	}

	switch (severity)
	{
		case GL_DEBUG_SEVERITY_HIGH_ARB:
			severityString = "High";
			break;

		case GL_DEBUG_SEVERITY_MEDIUM_ARB:
			severityString = "Medium";
			break;

		case GL_DEBUG_SEVERITY_LOW_ARB:
			severityString = "Low";
			break;
	}

	std::string str =
		(boost::format("%s: %s(%s) %d: %s")
			% sourceString % typeString % severityString % id % message).str();

	std::cerr << str << std::endl;

	if (severity == GL_DEBUG_SEVERITY_HIGH)
	{
		THROW_GL_EXCEPTION(GLException::DebugOutput, str);
	}
}

// ----------------------------------------------------------------------

GLContext::GLContext( void* hwnd )
{
	Create(hwnd, GLContextParam());
}

GLContext::GLContext( void* hwnd, const GLContextParam& param )
{
	Create(hwnd, param);
}

GLContext::~GLContext()
{
	if (hglrc != NULL)
	{
		wglMakeCurrent(NULL, NULL);

		if (!wglDeleteContext((HGLRC)hglrc))
		{
			THROW_GL_EXCEPTION(GLException::RunTimeError, "wglDeleteContext");
		}
	}

	hwnd = NULL;
	hdc = NULL;
	hglrc = NULL;
}

static LRESULT CALLBACK FakeWindowProcedure( HWND window, unsigned int msg, WPARAM wp, LPARAM lp )
{
	// Note: Dispatching WM_QUIT by PostQuitMessage have the event loop
	// of the main window to receive the quit message and to quit immediately.
	return DefWindowProc(window, msg, wp, lp);
}

void GLContext::Create( void* hwnd, const GLContextParam& param )
{
	this->hwnd = hwnd;
	hdc = (void*)GetDC((HWND)hwnd);

	// ------------------------------------------------------------

	// Create a dummy window

	WNDCLASSEX wc;
	const TCHAR* className = _T("DummyWindow");

	wc.cbSize        = sizeof(WNDCLASSEX);
	wc.style         = 0;
	wc.lpfnWndProc   = FakeWindowProcedure;
	wc.cbClsExtra    = 0;
	wc.cbWndExtra    = 0;
	wc.hInstance     = GetModuleHandle(NULL);
	wc.hIcon         = LoadIcon(NULL, IDI_APPLICATION);
	wc.hCursor       = LoadCursor(NULL, IDC_ARROW);
	wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
	wc.lpszMenuName  = NULL;
	wc.lpszClassName = className;
	wc.hIconSm       = LoadIcon(NULL, IDI_APPLICATION);

	if (!RegisterClassEx(&wc))
	{
		THROW_GL_EXCEPTION(GLException::RunTimeError, "RegisterClassEx");
	}

	HWND dummyHwnd = CreateWindowEx(
		WS_EX_CLIENTEDGE,
		className,
		_T("Dummy"),
		WS_OVERLAPPEDWINDOW,
		CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT,
		NULL, NULL, GetModuleHandle(NULL), NULL);

	if (dummyHwnd == NULL)
	{
		THROW_GL_EXCEPTION(GLException::RunTimeError, "CreateWindowEx");
	}

	HDC dummyHdc = GetDC(dummyHwnd);

	// ------------------------------------------------------------

	// Setup pixel format for dummy context

	PIXELFORMATDESCRIPTOR pfd;

	memset(&pfd, 0, sizeof(PIXELFORMATDESCRIPTOR));

	pfd.nSize = sizeof(PIXELFORMATDESCRIPTOR);
	pfd.nVersion = 1;
	pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
	pfd.iPixelType = PFD_TYPE_RGBA;
	pfd.cColorBits = 32;
	pfd.cDepthBits = 24;
	pfd.cStencilBits = 8;
	pfd.iLayerType = PFD_MAIN_PLANE;

	int pixelformat = ChoosePixelFormat(dummyHdc, &pfd);

	if (pixelformat == 0)
	{
		THROW_GL_EXCEPTION(GLException::RunTimeError, "ChoosePixelFormat");
	}

	if (!SetPixelFormat(dummyHdc, pixelformat, &pfd))
	{
		THROW_GL_EXCEPTION(GLException::RunTimeError, "SetPixelFormat");
	}

	// ------------------------------------------------------------

	// Create a dummy context
	HGLRC dummyContext = wglCreateContext(dummyHdc);

	if (dummyContext == NULL)
	{
		THROW_GL_EXCEPTION(GLException::RunTimeError,
			(boost::format("Failed to create context: wglCreateContext: %s") % GetLastError()).str());		
	}

	wglMakeCurrent(dummyHdc, dummyContext); 

	// Load extensions
	GLUtil::InitializeGlew(false);

	// Destroy the dummy window
	if (!DestroyWindow((HWND)dummyHwnd))
	{
		THROW_GL_EXCEPTION(GLException::RunTimeError, "DestroyWindow");
	}

	if (!UnregisterClass(className, GetModuleHandle(NULL)))
	{
		THROW_GL_EXCEPTION(GLException::RunTimeError, "UnregisterClass");
	}

	// ------------------------------------------------------------

	// Get version

	int major;
	int minor;

	if (param.MajorVersion == 0)
	{
		// Get version
		glGetIntegerv(GL_MAJOR_VERSION, &major);
		glGetIntegerv(GL_MINOR_VERSION, &minor);
	}
	else
	{
		major = param.MajorVersion;
		minor = param.MinorVersion;
	}

	// ------------------------------------------------------------

	// Setup pixel format for main context

	int pixelFormatAttr[] =
	{
		WGL_DRAW_TO_WINDOW_ARB, GL_TRUE,
		WGL_SUPPORT_OPENGL_ARB, GL_TRUE,
		WGL_DOUBLE_BUFFER_ARB, GL_TRUE,
		WGL_PIXEL_TYPE_ARB, WGL_TYPE_RGBA_ARB,
		WGL_COLOR_BITS_ARB, param.ColorBits,
		WGL_DEPTH_BITS_ARB, param.DepthBits,
		WGL_STENCIL_BITS_ARB, param.StencilBits,
		WGL_SAMPLE_BUFFERS_ARB, param.Multisample > 0 ? GL_TRUE : GL_FALSE,
		WGL_SAMPLES_ARB, param.Multisample,
		0
	};

	unsigned int numformats;

	if (!wglChoosePixelFormatARB((HDC)hdc, pixelFormatAttr, NULL, 1, &pixelformat, &numformats))
	{
		THROW_GL_EXCEPTION(GLException::RunTimeError, "wglChoosePixelFormatARB");
	}

	if (!SetPixelFormat((HDC)hdc, pixelformat, NULL))
	{
		THROW_GL_EXCEPTION(GLException::RunTimeError, "SetPixelFormat");
	}

	// ------------------------------------------------------------

	// Create GL context

	int flags = 0;

	flags |= param.ForwardCompatible ? WGL_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB : 0;
	flags |= param.DebugMode ? WGL_CONTEXT_DEBUG_BIT_ARB : 0;

	int contextAttr[] =
	{
		WGL_CONTEXT_MAJOR_VERSION_ARB, major,
		WGL_CONTEXT_MINOR_VERSION_ARB, minor,
		WGL_CONTEXT_FLAGS_ARB, flags,
		WGL_CONTEXT_PROFILE_MASK_ARB, (param.UseCoreProfile ? WGL_CONTEXT_CORE_PROFILE_BIT_ARB : WGL_CONTEXT_COMPATIBILITY_PROFILE_BIT_ARB),
		0
	};

	hglrc = (void*)wglCreateContextAttribsARB((HDC)hdc, 0, contextAttr);

	if (hglrc == NULL)
	{
		THROW_GL_EXCEPTION(GLException::RunTimeError, "wglCreateContextAttribsARB");
	}

	// ------------------------------------------------------------

	// Delete the dummy resources
	wglMakeCurrent(NULL, NULL);
	wglDeleteContext(dummyContext);

	// Enable current GL context
	wglMakeCurrent((HDC)hdc, (HGLRC)hglrc);

	// Initialize GLEW
	GLUtil::InitializeGlew(true);
}

void GLContext::SwapBuffers()
{
	::SwapBuffers((HDC)hdc);
}

void GLContext::EnableVSync( bool enable )
{
	wglSwapIntervalEXT(enable ? 1 : 0);
}

// ----------------------------------------------------------------------

const GLVertexAttribute GLDefaultVertexAttribute::Position(0, 3);
const GLVertexAttribute GLDefaultVertexAttribute::Normal(1, 3);
const GLVertexAttribute GLDefaultVertexAttribute::TexCoord0(2, 2);
const GLVertexAttribute GLDefaultVertexAttribute::TexCoord1(3, 2);
const GLVertexAttribute GLDefaultVertexAttribute::TexCoord2(4, 2);
const GLVertexAttribute GLDefaultVertexAttribute::TexCoord3(5, 2);
const GLVertexAttribute GLDefaultVertexAttribute::TexCoord4(6, 2);
const GLVertexAttribute GLDefaultVertexAttribute::Tangent(7, 2);
const GLVertexAttribute GLDefaultVertexAttribute::Color(8, 3);

// ----------------------------------------------------------------------

GLBufferObject::GLBufferObject()
{
	glGenBuffers(1, &id);
}

GLBufferObject::~GLBufferObject()
{
	glDeleteBuffers(1, &id);
}

GLenum GLBufferObject::Target()
{
	return target;
}

void GLBufferObject::Bind()
{
	glBindBuffer(target, id);
}

void GLBufferObject::Unbind()
{
	glBindBuffer(target, 0);
}

void GLBufferObject::Allocate( int size, const void* data, GLenum usage )
{
	Bind();
	glBufferData(target, size, data, usage);
	Unbind();
	this->size = size;
}

void GLBufferObject::Replace( int offset, int size, const void* data )
{
	Bind();
	glBufferSubData(target, offset, size, data);
	Unbind();
}

void GLBufferObject::Clear( GLenum internalformat, GLenum format, GLenum type, const void* data )
{
	Bind();
	glClearBufferData(target, internalformat, format, type, data);
	Unbind();
}

void GLBufferObject::Clear( GLenum internalformat, int offset, int size, GLenum format, GLenum type, const void* data )
{
	Bind();
	glClearBufferSubData(target, internalformat, offset, size, format, type, data);
	Unbind();
}

void GLBufferObject::Copy( GLBufferObject& writetarget, int readoffset, int writeoffset, int size )
{
	Bind();
	writetarget.Bind();
	glCopyBufferSubData(target, writetarget.Target(), readoffset, writeoffset, size);
	writetarget.Unbind();
	Unbind();
}

void GLBufferObject::Map( int offset, int length, unsigned int access, void** data )
{
	Bind();
	*data = glMapBufferRange(target, offset, length, access);
}

void GLBufferObject::Unmap()
{
	glUnmapBuffer(target);
	Unbind();
}

// ----------------------------------------------------------------------

GLPixelPackBuffer::GLPixelPackBuffer()
{
	target = GL_PIXEL_PACK_BUFFER;
}

// ----------------------------------------------------------------------

GLPixelUnpackBuffer::GLPixelUnpackBuffer()
{
	target = GL_PIXEL_UNPACK_BUFFER;
}

// ----------------------------------------------------------------------

GLVertexBuffer::GLVertexBuffer()
{
	target = GL_ARRAY_BUFFER;
}

void GLVertexBuffer::AddStatic( int n, const float* v )
{
	Allocate(n * sizeof(float), v, GL_STATIC_DRAW);
}

// ----------------------------------------------------------------------

GLIndexBuffer::GLIndexBuffer()
{
	target = GL_ELEMENT_ARRAY_BUFFER;
}

void GLIndexBuffer::AddStatic( int n, const GLuint* idx )
{
	Allocate(n * sizeof(GLuint), idx, GL_STATIC_DRAW);
}

void GLIndexBuffer::Draw( GLenum mode )
{
	Bind();
	glDrawElements(mode, size / sizeof(GLuint), GL_UNSIGNED_INT, NULL);
	Unbind();
}

// ----------------------------------------------------------------------

GLVertexArray::GLVertexArray()
{
	glGenVertexArrays(1, &id);
}

GLVertexArray::~GLVertexArray()
{
	glDeleteVertexArrays(1, &id);
}

void GLVertexArray::Bind()
{
	glBindVertexArray(id);
}

void GLVertexArray::Unbind()
{
	glBindVertexArray(0);
}

void GLVertexArray::Add( const GLVertexAttribute& attr, GLVertexBuffer* vb )
{
	Bind();
	vb->Bind();
	glVertexAttribPointer(attr.index, attr.size, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(attr.index);
	vb->Unbind();
	Unbind();
}

void GLVertexArray::Draw( GLenum mode, GLIndexBuffer* ib )
{
	Bind();
	ib->Draw(mode);
	Unbind();
}

void GLVertexArray::Draw( GLenum mode, int count )
{
	Bind();
	glDrawArrays(mode, 0, count);
	Unbind();
}

void GLVertexArray::Draw( GLenum mode, int first, int count )
{
	Bind();
	glDrawArrays(mode, first, count);
	Unbind();
}

// ----------------------------------------------------------------------

GLShader::GLShader()
{
	id = glCreateProgram();
}

GLShader::~GLShader()
{
	glDeleteProgram(id);
}

void GLShader::Begin()
{
	glUseProgram(id);
}

void GLShader::End()
{
	glUseProgram(0);
}

void GLShader::Compile( const std::string& path )
{
	Compile(InferShaderType(path), path);
}

void GLShader::Compile( ShaderType type, const std::string& path )
{
	CompileString(type, LoadShaderFile(path));
}


void GLShader::SetUniform( const std::string& name, const glm::mat4& mat )
{
	GLuint uniformID = GetOrCreateUniformID(name);
	glUniformMatrix4fv(uniformID, 1, GL_FALSE, glm::value_ptr(mat));
}

void GLShader::SetUniform( const std::string& name, const glm::mat3& mat )
{
	GLuint uniformID = GetOrCreateUniformID(name);
	glUniformMatrix3fv(uniformID, 1, GL_FALSE, glm::value_ptr(mat));
}

void GLShader::SetUniform( const std::string& name, float v )
{
	GLuint uniformID = GetOrCreateUniformID(name);
	glUniform1f(uniformID, v);
}

void GLShader::SetUniform( const std::string& name, const glm::vec2& v )
{
	GLuint uniformID = GetOrCreateUniformID(name);
	glUniform2fv(uniformID, 1, glm::value_ptr(v));
}

void GLShader::SetUniform( const std::string& name, const glm::vec3& v )
{
	GLuint uniformID = GetOrCreateUniformID(name);
	glUniform3fv(uniformID, 1, glm::value_ptr(v));
}

void GLShader::SetUniform( const std::string& name, const glm::vec4& v )
{
	GLuint uniformID = GetOrCreateUniformID(name);
	glUniform4fv(uniformID, 1, glm::value_ptr(v));
}

void GLShader::SetUniform( const std::string& name, int v )
{
	GLuint uniformID = GetOrCreateUniformID(name);
	glUniform1i(uniformID, v);
}

void GLShader::CompileString( ShaderType type, const std::string& content )
{
	// Create and compile shader
	GLuint shaderID = glCreateShader(type);
	const char* contentPtr = content.c_str();

	glShaderSource(shaderID, 1, &contentPtr, NULL);
	glCompileShader(shaderID);

	// Check errors
	int ret;
	glGetShaderiv(shaderID, GL_COMPILE_STATUS, &ret);

	if (ret == 0)
	{
		// Required size
		int length;
		glGetShaderiv(shaderID, GL_INFO_LOG_LENGTH, &length);

		// Get info log
		boost::scoped_array<char> infoLog(new char[length]);
		glGetShaderInfoLog(shaderID, length, NULL, infoLog.get());
		glDeleteShader(shaderID);

		std::stringstream ss;
		ss << "[" << ShaderTypeString(type) << "]" << std::endl;
		ss << infoLog.get() << std::endl;

		THROW_GL_EXCEPTION(GLException::ShaderCompileError, ss.str());
	}

	// Attach to the program
	glAttachShader(id, shaderID);
	glDeleteShader(shaderID);
}

void GLShader::Link()
{
	// Link program
	glLinkProgram(id);

	// Check error
	GLint ret;
	glGetProgramiv(id, GL_LINK_STATUS, &ret);

	if (ret == GL_FALSE)
	{
		// Required size
		GLint length;
		glGetProgramiv(id, GL_INFO_LOG_LENGTH, &length);

		boost::scoped_array<char> infoLog(new char[length]);
		glGetProgramInfoLog(id, length, NULL, infoLog.get());

		// Throw exception
		THROW_GL_EXCEPTION(GLException::ProgramLinkError, infoLog.get());
	}
}

std::string GLShader::ShaderTypeString( ShaderType type )
{
	switch (type)
	{
		case VertexShader:
			return "VertexShader";
			break;

		case TessControlShader:
			return "TessControlShader";
			break;

		case TessEvaluationShader:
			return "TessEvaluationShader";
			break;

		case GeometryShader:
			return "GeometryShader";
			break;

		case FragmentShader:
			return "FragmentShader";
			break;
	}

	THROW_GL_EXCEPTION(GLException::ArgumentError, "Invalid shader type");
}

GLShader::ShaderType GLShader::InferShaderType( const std::string& path )
{
	std::string extension = path.substr(path.find_last_of("."));

	if (extension == ".vert" || extension == ".vsh")
	{
		return VertexShader;
	}
	else if (extension == ".tessctrl" || extension == ".tcsh")
	{
		return TessControlShader;
	}
	else if (extension == ".tesseval" || extension == ".tesh")
	{
		return TessEvaluationShader;
	}
	else if (extension == ".geom" || extension == ".gsh")
	{
		return GeometryShader;
	}
	else if (extension == ".frag" || extension == ".fsh")
	{
		return FragmentShader;
	}
	else
	{
		THROW_GL_EXCEPTION(GLException::RunTimeError, "Invalid shader file extension " + extension);
	}
}

std::string GLShader::LoadShaderFile( const std::string& path )
{
	std::ifstream ifs(path.c_str(), std::ios::in);
	if (!ifs.is_open())
	{
		THROW_GL_EXCEPTION(GLException::FileError, "Failed to load shader " + path);
	}

	std::string line = "";
	std::string text = "";

	while (getline(ifs, line))
	{
		text += ("\n" + line);
	}

	ifs.close();

	return text;
}

GLuint GLShader::GetOrCreateUniformID( const std::string& name )
{
	UniformLocationMap::iterator it = uniformLocationMap.find(name);

	if (it == uniformLocationMap.end())
	{
		GLuint loc = glGetUniformLocation(id, name.c_str());

		uniformLocationMap[name] = loc;

		return loc;
	}

	return it->second;
}

// ----------------------------------------------------------------------

GLTexture::GLTexture()
{
	glGenTextures(1, &id);
}

GLTexture::~GLTexture()
{
	glDeleteTextures(1, &id);
}

void GLTexture::Bind( int unit )
{
	glActiveTexture((GLenum)(GL_TEXTURE0 + unit));
	glBindTexture(target, id);
}

void GLTexture::Unbind()
{
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(target, 0);
}

// ----------------------------------------------------------------------

GLTexture2D::GLTexture2D()
{
	target = GL_TEXTURE_2D;
	minFilter = GL_LINEAR_MIPMAP_LINEAR;
	magFilter = GL_LINEAR;
	wrap = GL_REPEAT;
	anisotropicFiltering = true;
}

void GLTexture2D::Allocate( int width, int height )
{
	Allocate(width, height, GL_RGBA8);
}

void GLTexture2D::Allocate( int width, int height, GLenum internalFormat )
{
	Allocate(width, height, internalFormat, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
}

void GLTexture2D::Allocate( int width, int height, GLenum internalFormat, GLenum format, GLenum type, const void* data )
{
	this->width = width;
	this->height = height;
	this->internalFormat = internalFormat;

	Bind();
	glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, width, height, 0, format, type, data);
	GenerateMipmap();
	UpdateTextureParams();
	Unbind();
}

void GLTexture2D::Replace( const glm::ivec4& rect, GLenum format, GLenum type, const void* data )
{
	Bind();
	glTexSubImage2D(GL_TEXTURE_2D, 0, rect.x, rect.y, rect.z, rect.w, format, type, data);
	GenerateMipmap();
	Unbind();
}

void GLTexture2D::Replace( GLPixelUnpackBuffer* pbo, const glm::ivec4& rect, GLenum format, GLenum type, int offset /*= 0*/ )
{
	pbo->Bind();
	Replace(rect, format, type, (void*)offset);
	pbo->Unbind();
}

void GLTexture2D::GetInternalData( GLenum format, GLenum type, void* data )
{
	Bind();
	glGetTexImage(GL_TEXTURE_2D, 0, format, type, data);
	Unbind();
}

void GLTexture2D::GenerateMipmap()
{
	if (minFilter == GL_LINEAR_MIPMAP_LINEAR && magFilter == GL_LINEAR)
	{
		glGenerateMipmap(GL_TEXTURE_2D);
	}
}

void GLTexture2D::UpdateTextureParams()
{
	if (anisotropicFiltering)
	{
		// If the anisotropic filtering can be used,
		// set to the maximum possible value.
		float maxAnisoropy;

		glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &maxAnisoropy);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, maxAnisoropy);
	}

	// Wrap mode
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrap);

	// Filters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, minFilter);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, magFilter);
}

// ----------------------------------------------------------------------

