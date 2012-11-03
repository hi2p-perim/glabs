#ifndef __GLABS_TEST_LIB_GL_TEST_UTIL_H__
#define __GLABS_TEST_LIB_GL_TEST_UTIL_H__

class GLTestUtil
{
private:

	GLTestUtil() {}
	DISALLOW_COPY_AND_ASSIGN(GLTestUtil);
	
public:

	static std::string PrintGLException(const GLException& e);
	static double CurrentTimeMilli();

};

class GLTestWindow
{
public:

	GLTestWindow(bool showWindow = false);
	~GLTestWindow();

	bool ProcessEvent();
	void SetTitle(const std::string& title);
	void* Handle() { return hwnd; }

private:

	void* hwnd;
	bool done;

};

#endif // __GLABS_TEST_LIB_GL_TEST_UTIL_H__