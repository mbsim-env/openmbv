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
#include <map>
#include <stdexcept>
#include <algorithm>
#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif

namespace {

inline std::string getLastError() {
#ifndef _WIN32
  const char *err=dlerror();
  return err?err:"";
#else
  return std::to_string(GetLastError());
#endif
}

}

namespace MBXMLUtils::SharedLibrary {

#ifndef _WIN32
  using Handle = void *;
#else
  using Handle = HMODULE;
#endif

  using InitFuncType = int (*)();

  template<typename T> 
  inline T getSymbol(const std::string &file, const std::string &symbolName, bool throwOnError=true);

// Load the libary file, if not already loaded.
// Unloads the library only at program exit!!!
// file should be a canonical path.
inline Handle load(const std::string &file, bool global=false) {
  static std::map<std::string, Handle> library;
  std::pair<std::map<std::string, Handle>::iterator, bool> res=library.insert(std::pair<std::string, Handle>(file, NULL));
  if(res.second) {
#ifndef _WIN32
    res.first->second=dlopen(file.c_str(), RTLD_NOW | (global ? RTLD_GLOBAL : RTLD_LOCAL) | RTLD_DEEPBIND | RTLD_NODELETE);
#else
    std::string fileWinSep=file;
    std::replace(fileWinSep.begin(), fileWinSep.end(), '/', '\\'); // LoadLibraryEx can not handle '/' as path separator
    res.first->second=LoadLibraryEx(fileWinSep.c_str(), NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
#endif
    if(!res.first->second)
      throw std::runtime_error("Unable to load the library '"+file+"': "+getLastError());

    // call the init function if it exists
    auto initFunc=getSymbol<InitFuncType>(file, "MBXMLUtils_SharedLibrary_init", false);
    if(initFunc) {
      int ret=initFunc();
      if(ret)
        throw std::runtime_error("Unable to initialize the library '"+file+"'.\n"
          "The function 'MBXMLUtils_SharedLibrary_init' returned with error code "+std::to_string(ret)+".");
    }
  }
  return res.first->second;
}

// Load the libary file, if not already loaded and return the symbol symbolName.
// Unloads the library only at program exit!!!
template<typename T> 
inline T getSymbol(const std::string &file, const std::string &symbolName, bool throwOnError) {
  Handle h=load(file);
#ifndef _WIN32
  void *addr=dlsym(h, symbolName.c_str());
#else
  void *addr=reinterpret_cast<void*>(GetProcAddress(h, symbolName.c_str()));
#endif
  if(!addr) {
    if(throwOnError)
      throw std::runtime_error("Unable to load the symbol '"+symbolName+"' from library '"+
                               file+"': "+getLastError());
    else
      return nullptr;
  }
  return reinterpret_cast<T>(reinterpret_cast<size_t>(addr));
}

}

#endif
