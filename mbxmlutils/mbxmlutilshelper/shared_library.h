/*
 * Author: Markus Friedrich
 *
 * This file is free and unencumbered software released into the public domain.
 * 
 * Anyone is free to copy, modify, publish, use, compile, sell, or
 * distribute this software, either in source code form or as a compiled
 * binary, for any purpose, commercial or non-commercial, and by any
 * means.
 * 
 * In jurisdictions that recognize copyright laws, the author or authors
 * of this software dedicate any and all copyright interest in the
 * software to the public domain. We make this dedication for the benefit
 * of the public at large and to the detriment of our heirs and
 * successors. We intend this dedication to be an overt act of
 * relinquishment in perpetuity of all present and future rights to this
 * software under copyright law.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 * 
 * For more information, please refer to <http://unlicense.org/>
 */

#ifndef _MBXMLUTILS_SHAREDLIBRARY_H_
#define _MBXMLUTILS_SHAREDLIBRARY_H_

#include <string>
#include <boost/date_time/posix_time/posix_time.hpp>
#include "last_write_time.h"
#include <boost/lexical_cast.hpp>
#ifdef _WIN32
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif

namespace MBXMLUtils {

class SharedLibrary {
  public:
    inline SharedLibrary(const std::string &file_);
    inline SharedLibrary(const SharedLibrary& src);
    inline ~SharedLibrary();
    template<typename T>
    inline T getSymbol(const std::string &symbolName);
    const std::string file;
    const boost::posix_time::ptime writeTime;
    inline bool operator<(const SharedLibrary& b) const { return file<b.file; }
  private:
    inline std::string getLastError();
    inline void init();
#ifndef _WIN32
    void* handle;
#else
    HMODULE handle;
#endif
};

template<typename T>
T SharedLibrary::getSymbol(const std::string &symbolName) {
#ifndef _WIN32
  void *addr=dlsym(handle, symbolName.c_str());
#else
  void *addr=reinterpret_cast<void*>(GetProcAddress(handle, symbolName.c_str()));
#endif
  if(!addr)
    throw std::runtime_error("Unable to load the symbol '"+symbolName+"' from library '"+
                             file+"': "+getLastError());
  return reinterpret_cast<T>(addr);
}

SharedLibrary::SharedLibrary(const std::string &file_) : file(file_),
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
  std::string fileWinSep=file;
  replace(fileWinSep.begin(), fileWinSep.end(), '/', '\\'); // LoadLibraryEx can not handle '/' as path separator
  handle=LoadLibraryEx(fileWinSep.c_str(), NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
#endif
  if(!handle)
    throw std::runtime_error("Unable to load the library '"+file+"': "+getLastError());
}

SharedLibrary::~SharedLibrary() {
#ifndef _WIN32
  dlclose(handle);
#else
  FreeLibrary(handle);
#endif
}

std::string SharedLibrary::getLastError() {
#ifndef _WIN32
  const char *err=dlerror();
  return err?err:"";
#else
  return boost::lexical_cast<std::string>(GetLastError());
#endif
}

}

#endif
