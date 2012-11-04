#ifndef __GLABS_H__
#define __GLABS_H__

/*
	glabs : OpenGL Abstraction Layer
	The OpenGL 4.2 compatible class library
*/

#include <GL/glew.h>
#include <GL/wglew.h>

class GLUtil
{
public:

	enum DebugOutputFrequency
	{
		DebugOutputFrequencyHigh,
		DebugOutputFrequencyMedium,
		DebugOutputFrequencyLow
	};

private:

	GLUtil() {}
	DISALLOW_COPY_AND_ASSIGN(GLUtil);

public:

	static void InitializeGlew(bool experimental = true);
	static void EnableDebugOutput(DebugOutputFrequency freq = DebugOutputFrequencyHigh);
	static bool CheckExtension(const std::string& name);

private:

	static void APIENTRY DebugOutput(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, GLvoid* userParam);

};

struct GLContextParam
{

	GLContextParam()
		: ColorBits(32)
		, DepthBits(24)
		, StencilBits(8)
		, MajorVersion(4)
		, MinorVersion(2)
		, Multisample(0)
		, ForwardCompatible(true)
		, DebugMode(false)
		, UseCoreProfile(true)
	{

	}

	int ColorBits;
	int DepthBits;
	int StencilBits;
	int MajorVersion;
	int MinorVersion;
	int Multisample;
	bool ForwardCompatible;
	bool DebugMode;
	bool UseCoreProfile;

};

class GLContext
{
public:

	GLContext(void* hwnd);
	GLContext(void* hwnd, const GLContextParam& param);
	~GLContext();

	void EnableVSync(bool enable);
	void SwapBuffers();

private:

	void Create(void* hwnd, const GLContextParam& param);

private:

	void* hwnd;
	void* hdc;
	void* hglrc;

};

class GLVertexAttribute
{
public:

	GLVertexAttribute(int index, int size)
		: index(index)
		, size(size)
	{

	}

public:

	int index;
	int size;

};

class GLDefaultVertexAttribute
{
private:

	GLDefaultVertexAttribute() {}
	DISALLOW_COPY_AND_ASSIGN(GLDefaultVertexAttribute);

public:

	static const GLVertexAttribute Position;
	static const GLVertexAttribute Normal;
	static const GLVertexAttribute TexCoord0;
	static const GLVertexAttribute TexCoord1;
	static const GLVertexAttribute TexCoord2;
	static const GLVertexAttribute TexCoord3;
	static const GLVertexAttribute TexCoord4;
	static const GLVertexAttribute Tangent;
	static const GLVertexAttribute Color;

};

class GLResource
{
public:

	GLuint ID() { return id; }

protected:

	GLuint id;

};

class GLBufferObject : public GLResource
{
public:

	GLBufferObject();
	virtual ~GLBufferObject() = 0;

	GLenum Target();
	void Bind();
	void Unbind();
	void Allocate(int size, const void* data, GLenum usage);
	void Replace(int offset, int size, const void* data);
	void Clear(GLenum internalformat, GLenum format, GLenum type, const void* data);
	void Clear(GLenum internalformat, int offset, int size, GLenum format, GLenum type, const void* data);
	void Copy(GLBufferObject& writetarget, int readoffset, int writeoffset, int size);
	void Map(int offset, int length, unsigned int access, void** data);
	void Unmap();

protected:

	int size;
	GLenum target;

};

class GLPixelPackBuffer : public GLBufferObject
{
public:

	GLPixelPackBuffer();

};

class GLPixelUnpackBuffer : public GLBufferObject
{
public:

	GLPixelUnpackBuffer();

};

class GLVertexBuffer : public GLBufferObject
{
public:

	GLVertexBuffer();
	void AddStatic(int n, const float* v);

};

class GLIndexBuffer : public GLBufferObject
{
public:

	GLIndexBuffer();
	void AddStatic(int n, const GLuint* idx);
	void Draw(GLenum mode);

};

class GLVertexArray : public GLResource
{
public:

	GLVertexArray();
	~GLVertexArray();

	void Bind();
	void Unbind();
	void Add(const GLVertexAttribute& attr, GLVertexBuffer* vb);
	void Draw(GLenum mode, int count);
	void Draw(GLenum mode, int first, int count);
	void Draw(GLenum mode, GLIndexBuffer* ib);

};

class GLShader : public GLResource
{
public:

	enum ShaderType
	{
		VertexShader = GL_VERTEX_SHADER,
		TessControlShader = GL_TESS_CONTROL_SHADER,
		TessEvaluationShader = GL_TESS_EVALUATION_SHADER,
		GeometryShader = GL_GEOMETRY_SHADER,
		FragmentShader = GL_FRAGMENT_SHADER
	};

	typedef boost::unordered_map<std::string, GLuint> UniformLocationMap;

public:

	GLShader();
	~GLShader();

	void Begin();
	void End();
	void Compile(const std::string& path);
	void Compile(ShaderType type, const std::string& path);
	void CompileString(ShaderType type, const std::string& content);
	void Link();
	void SetUniform(const std::string& name, const glm::mat4& mat);
	void SetUniform(const std::string& name, const glm::mat3& mat);
	void SetUniform(const std::string& name, float v);
	void SetUniform(const std::string& name, const glm::vec2& v);
	void SetUniform(const std::string& name, const glm::vec3& v);
	void SetUniform(const std::string& name, const glm::vec4& v);
	void SetUniform(const std::string& name, int v);

private:

	std::string ShaderTypeString(ShaderType type);
	ShaderType InferShaderType(const std::string& path);
	std::string LoadShaderFile(const std::string& path);
	GLuint GetOrCreateUniformID(const std::string& name);

private:

	UniformLocationMap uniformLocationMap;

};

// TODO
//class ProgramPipeline
//{
//
//};

class GLTexture : public GLResource
{
public:

	GLTexture();
	virtual ~GLTexture() = 0;

	void Bind(int unit = 0);
	void Unbind();

protected:

	GLenum target;

};

class GLTexture2D : public GLTexture
{
public:

	GLTexture2D();

	void Allocate(int width, int height);
	void Allocate(int width, int height, GLenum internalFormat);
	void Allocate(int width, int height, GLenum internalFormat, GLenum format, GLenum type, const void* data);
	void Replace(const glm::ivec4& rect, GLenum format, GLenum type, const void* data);
	void Replace(GLPixelUnpackBuffer* pbo, const glm::ivec4& rect, GLenum format, GLenum type, int offset = 0);
	void GetInternalData(GLenum format, GLenum type, void* data);
	void GenerateMipmap();
	void UpdateTextureParams();
	
	int Width() { return width; }
	int Height() { return height; }
	GLenum InternalFormat() { return internalFormat; }

	// TODO: replace with sampler object
	GLenum MinFilter() { return minFilter; }
	GLenum MagFilter() { return magFilter; }
	GLenum Wrap() { return wrap; }
	bool AnisotropicFiltering() { return anisotropicFiltering; }
	void SetMinFilter(GLenum minFilter) { this->minFilter = minFilter; }
	void SetMagFilter(GLenum magFilter) { this->magFilter = magFilter; }
	void SetWrap(GLenum wrap) { this->wrap = wrap; }
	void SetAnisotropicFiltering(bool anisotropicFiltering) { this->anisotropicFiltering = anisotropicFiltering; }

private:

	int width;
	int height;
	GLenum internalFormat;
	GLenum minFilter;
	GLenum magFilter;
	GLenum wrap;
	bool anisotropicFiltering;

};

#endif // __GLABS_H__