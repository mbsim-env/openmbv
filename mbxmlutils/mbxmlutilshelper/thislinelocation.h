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

#ifndef _MBXMLUTILSHELPER_THISLINELOCATION_H_
#define _MBXMLUTILSHELPER_THISLINELOCATION_H_

#include <string>
#include <filesystem>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  ifndef _GNU_SOURCE
#    define _GNU_SOURCE // dladdr requires _GNU_SOURCE to be defined
#  endif
#  include <dlfcn.h>
#  include <unistd.h>
#endif

namespace {
  #ifdef _WIN32
    extern "C" void *__ImageBase;
  #endif
}

namespace MBXMLUtils {

//! We do not use boost::dll or boost::filesystem here to enable this class header only.
//! Usage:
//! ThisLineLocation loc; // as a global variable at the code where you want to get the location.
//! std::string location=loc(); // to get the absolute path of the lib where loc is defined in.
class ThisLineLocation {
  public:
    ThisLineLocation() {
#ifdef _WIN32
      char moduleName[2048];
      GetModuleFileName(reinterpret_cast<HMODULE>(&__ImageBase), moduleName, sizeof(moduleName));
      p=moduleName;
#else
      char buffer[2048];
      std::string curDir=getcwd(buffer, sizeof(buffer));

      Dl_info info;
      dladdr(reinterpret_cast<void*>(&dummyFunc), &info);
      // convert to absolute path and return
      std::string name(info.dli_fname);
      p=name[0]=='/'?name:curDir+"/"+name;
#endif
      p=std::filesystem::canonical(p);
    }
    std::string operator()() { return p; }
  private:
    std::string p;
#ifndef _WIN32
    static void dummyFunc() {}
#endif
};

}

#endif
