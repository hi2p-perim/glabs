#include <glabstestlib/gltestutil.h>

#include <Windows.h>
#include <tchar.h>

std::string GLTestUtil::PrintGLException( const GLException& e )
{
	std::stringstream ss;

	ss << "[Exception]" << std::endl;
	ss << e.TypeString() << std::endl;
	ss << std::endl;

	ss << "[File]" << std::endl;
	ss << e.FileName() << std::endl;
	ss << std::endl;

	ss << "[Function]" << std::endl;
	ss << e.FuncName() << std::endl;
	ss << std::endl;

	ss << "[Line]" << std::endl;
	ss << e.Line() << std::endl;
	ss << std::endl;

	ss << "[Stack Trace]" << std::endl;
	ss << e.StackTrace();
	ss << std::endl;

	ss << "[Message]" << std::endl;
	ss << e.what() << std::endl;

	return ss.str();
}

double GLTestUtil::CurrentTimeMilli()
{
	LARGE_INTEGER frequency;
	LARGE_INTEGER t;

	QueryPerformanceFrequency(&frequency);
	QueryPerformanceCounter(&t);

	return t.QuadPart * 1000.0 / frequency.QuadPart;
}

// ----------------------------------------------------------------------

static LRESULT CALLBACK WindowProcedure( HWND window, unsigned int msg, WPARAM wp, LPARAM lp )
{
	switch (msg)
	{
		// Dispatch WM_QUIT only when WM_CLOSE
		// if WM_QUIT is dispatched from the hidden window which has no event loop, 
		// PeekMessage in another window receives the message and something wrong will happen.
		case WM_CLOSE:
			PostQuitMessage(0) ;
			return 0 ;

		default:
			return DefWindowProc(window, msg, wp, lp);
	}
}

GLTestWindow::GLTestWindow( bool showWindow )
	: done(false)
{
	WNDCLASSEX wc;
	const TCHAR* className = _T("TestWindow");

	wc.cbSize        = sizeof(WNDCLASSEX);
	wc.style         = 0;
	wc.lpfnWndProc   = WindowProcedure;
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

	hwnd = (HWND)CreateWindowEx(
		WS_EX_CLIENTEDGE,
		className,
		_T("GLTest"),
		WS_OVERLAPPEDWINDOW,
		CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT,
		NULL, NULL, GetModuleHandle(NULL), NULL);

	if (hwnd == NULL)
	{
		THROW_GL_EXCEPTION(GLException::RunTimeError, "CreateWindowEx");
	}

	if (showWindow)
	{
		ShowWindow((HWND)hwnd, SW_SHOWDEFAULT);
		UpdateWindow((HWND)hwnd);
	}
}

GLTestWindow::~GLTestWindow()
{
	const TCHAR* className = _T("TestWindow");

	if (!DestroyWindow((HWND)hwnd))
	{
		THROW_GL_EXCEPTION(GLException::RunTimeError, "DestroyWindow");
	}

	if (!UnregisterClass(className, GetModuleHandle(NULL)))
	{
		THROW_GL_EXCEPTION(GLException::RunTimeError, "UnregisterClass");
	}
}

bool GLTestWindow::ProcessEvent()
{
	MSG msg;

	while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
	{
		if (msg.message == WM_QUIT)
		{
			return false;
		}

		TranslateMessage(&msg);
		DispatchMessage(&msg);
	}

	return true;
}

void GLTestWindow::SetTitle( const std::string& title )
{
	SetWindowTextA((HWND)hwnd, title.c_str());
}
