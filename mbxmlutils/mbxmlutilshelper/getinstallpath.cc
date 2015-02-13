#include <config.h>
#include "getinstallpath.h"

#ifdef _WIN32 // Windows
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif

using namespace std;
using namespace boost::filesystem;

namespace {

#ifndef _WIN32
// stores the current directory at the time this shared object was loaded
static char currentDirOnLoad[2048];

// initialize the currentDirOnLoad variable when loading the shared object (this is achived by the GCC constructor attribute)
__attribute__((constructor))
static void initCurrentDirOnLoad() {
  char* unused __attribute__((unused));
  unused = getcwd(currentDirOnLoad, 2048);
}
#endif

}

namespace MBXMLUtils {

path getInstallPath() {
  // Get the file containing this function
#ifdef _WIN32
  wchar_t moduleName[2048];
  GetModuleFileNameW(GetModuleHandle("libmbxmlutilshelper-0.dll"), moduleName, sizeof(moduleName));
  path dllPath(moduleName);
#else
  Dl_info info;
#ifdef __GNUC__
__extension__
#endif
  dladdr(reinterpret_cast<void*>(&getInstallPath), &info);
  path dllPath;
  if(info.dli_fname[0]=='/') // use an absolute path as it
     dllPath=info.dli_fname;
  else // prefix a relative path with the current path at the time this shared object was loaded.
       // This is required since dladdr returns the string which was used by dlopen to open the shared object
       // which may be a relative path which has to be interpreted relative to the current directory at the time the shared object was loaded.
     dllPath=path(currentDirOnLoad)/info.dli_fname;
#endif
  dllPath.remove_filename();
  dllPath=dllPath.parent_path();
  return boost::filesystem::canonical(dllPath);
}

} // end namespace MBXMLUtils
