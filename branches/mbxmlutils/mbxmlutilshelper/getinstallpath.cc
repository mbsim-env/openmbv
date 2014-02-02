#include "getinstallpath.h"
#include <string.h>
#include <unistd.h>
#include <stdlib.h>

#ifdef _WIN32 // Windows
#include <windows.h>
#endif

using namespace std;
using namespace boost::filesystem;

namespace MBXMLUtils {

path getInstallPath() {
  // get path of this executable
  static char exePath[4096]="";
  if(strcmp(exePath, "")!=0) return path(exePath)/"..";

#ifdef _WIN32 // Windows
  GetModuleFileName(NULL, exePath, sizeof(exePath));
  for(size_t i=0; i<strlen(exePath); i++) if(exePath[i]=='\\') exePath[i]='/'; // convert '\' to '/'
  *strrchr(exePath, '/')=0; // remove the program name
#else // Linux
  if(getenv("DEVELOPER_HACK_INSTALLPATH")) return getenv("DEVELOPER_HACK_INSTALLPATH");
  int exePathLength=readlink("/proc/self/exe", exePath, sizeof(exePath)); // get abs path to this executable
  exePath[exePathLength]=0; // null terminate
  *strrchr(exePath, '/')=0; // remove the program name
#endif

  return path(exePath)/"..";
}

} // end namespace MBXMLUtils
