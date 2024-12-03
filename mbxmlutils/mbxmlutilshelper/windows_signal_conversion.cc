#include "windows_signal_conversion.h"
#include <thread>
#ifdef _WIN32
#include <windows.h>
#endif

using namespace std;

namespace MBXMLUtils {

void convertWMCLOSEtoSIGTERM() {
#ifdef _WIN32
  thread messageLoopThread([](){
    const char DUMMYWINDOWCLASS[] = "DummyWindowClass";
    WNDCLASS wc = {};
    wc.lpfnWndProc = [](HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) -> LRESULT {
      switch (uMsg) {
        case WM_CLOSE:
          raise(SIGTERM); // convert a WM_CLOSE to a SIGTERM signal
          PostQuitMessage(0); // exit this thread (we close the application)
          return 0;
        default:
          return DefWindowProc(hwnd, uMsg, wParam, lParam);
      }
    };
    wc.hInstance = GetModuleHandle(nullptr);
    wc.lpszClassName = DUMMYWINDOWCLASS;
    RegisterClass(&wc);
    CreateWindowEx(0, DUMMYWINDOWCLASS, "Dummy Window", 0, 0, 0, 0, 0, nullptr, nullptr, GetModuleHandle(nullptr), nullptr);
    MSG msg = {};
    while(GetMessage(&msg, nullptr, 0, 0)) {
      TranslateMessage(&msg);
      DispatchMessage(&msg);
    }
  });
  messageLoopThread.detach();
#endif
}

}
