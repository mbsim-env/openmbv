#include <mbxmlutils/mbxmlutils.h>
#include <string.h>
#include <unistd.h>

using namespace std;

namespace MBXMLUtils {

string getInstallPath() {
  // get path of this executable
  static char exePath[4096]="";
  if(strcmp(exePath, "")!=0) return string(exePath)+"/..";

#ifdef _WIN32 // Windows
  GetModuleFileName(NULL, exePath, sizeof(exePath));
  for(size_t i=0; i<strlen(exePath); i++) if(exePath[i]=='\\') exePath[i]='/'; // convert '\' to '/'
  *strrchr(exePath, '/')=0; // remove the program name
#else // Linux
#ifdef DEVELOPER_HACK_EXEPATH
  // use hardcoded exePath
  strcpy(exePath, DEVELOPER_HACK_EXEPATH);
#else
  int exePathLength=readlink("/proc/self/exe", exePath, sizeof(exePath)); // get abs path to this executable
  exePath[exePathLength]=0; // null terminate
  *strrchr(exePath, '/')=0; // remove the program name
#endif
#endif

  return string(exePath)+"/..";
}

} // end namespace MBXMLUtils
