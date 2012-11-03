#ifndef __GLABS_EXCEPTION_H__
#define __GLABS_EXCEPTION_H__

#include <exception>

#ifdef _WIN32
#pragma comment(lib, "DbgHelp.lib")
#endif

class GLException : public std::exception
{
public:

	enum ErrorType
	{
		RunTimeError,
		FileError,
		ArgumentError,
		ShaderCompileError,
		ProgramLinkError,
		CapabilityError,
		NotSupported,
		DebugOutput
	};

public:

	GLException(ErrorType type, const std::string& message, const char* fileName, const char* funcName,
		const int line, const std::string& stackTrace)
		: type(type)
		, message(message)
		, fileName(fileName)
		, funcName(funcName)
		, line(line)
		, stackTrace(stackTrace)
	{

	}

	ErrorType Type() const { return type; }
	const char* TypeString() const;
	const char* what() const { return message.c_str(); }
	const char* FileName() const { return fileName; }
	const char* FuncName() const { return funcName; }
	const int Line() const { return line; }
	std::string StackTrace() const { return stackTrace; }

public:

	static std::string GetStackTrace();

private:

	ErrorType type;
	std::string message;
	const char* fileName;
	const char* funcName;
	int line;
	std::string stackTrace;

};

#define THROW_GL_EXCEPTION(type, message) \
	throw GLException(type, message, __FILE__, __FUNCTION__, __LINE__, GLException::GetStackTrace())

#endif // __GLABS_EXCEPTION_H__