#include "shared_library.h"
#include "last_write_time.h"
#ifndef _WIN32
#  include <dlfcn.h>
#else
#  include <windows.h>
#endif

using namespace std;

namespace MBXMLUtils {

SharedLibrary::SharedLibrary(const boost::filesystem::path &file_) : file(file_),
  writeTime(boost::myfilesystem::last_write_time(file.generic_string())) {
  init();
}

SharedLibrary::SharedLibrary(const SharedLibrary& src) : file(src.file), writeTime(src.writeTime) {
  init();
}

void SharedLibrary::init() {
#ifndef _WIN32
  handle=dlopen(file.generic_string().c_str(), RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND);
#else
  handle=LoadLibraryEx(file.generic_string().c_str(), NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
#endif
  if(!handle)
    throw runtime_error("Unable to load the MBSim module: Library '"+file.generic_string()+"' not found.");
}

SharedLibrary::~SharedLibrary() {
#ifndef _WIN32
  dlclose(handle);
#else
  FreeLibrary(handle);
#endif
}

}
