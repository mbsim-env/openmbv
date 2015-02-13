#include "config.h"
#include "shared_library.h"
#include "last_write_time.h"
#include <boost/lexical_cast.hpp>
#ifndef _WIN32
#  include <dlfcn.h>
#endif

using namespace std;

namespace MBXMLUtils {

SharedLibrary::SharedLibrary(const string &file_) : file(file_),
  writeTime(boost::myfilesystem::last_write_time(file)) {
  init();
}

SharedLibrary::SharedLibrary(const SharedLibrary& src) : file(src.file), writeTime(src.writeTime) {
  init();
}

void SharedLibrary::init() {
#ifndef _WIN32
  handle=dlopen(file.c_str(), RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND);
#else
  handle=LoadLibraryEx(file.generic_string().c_str(), NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
#endif
  if(!handle)
    throw runtime_error("Unable to load the library '"+file+"': "+getLastError());
}

SharedLibrary::~SharedLibrary() {
#ifndef _WIN32
  dlclose(handle);
#else
  FreeLibrary(handle);
#endif
}

void* SharedLibrary::getAddress(const std::string &symbolName) {
#ifndef _WIN32
  void *addr=dlsym(handle, symbolName.c_str());
#else
  void *addr=reinterpret_cast<void*>(GetProcAddress(handle, symbolName.c_str()));
#endif
  if(!addr)
    throw runtime_error("Unable to load the symbol '"+symbolName+"' from library '"+
                        file+"': "+getLastError());
  return addr;
}

string SharedLibrary::getLastError() {
#ifndef _WIN32
  const char *err=dlerror();
  return err?err:"";
#else
  return boost::lexical_cast<string>(GetLastError());
#endif
}

}
