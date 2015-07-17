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
#ifdef _WIN32
#  include <windows.h>
#endif

namespace MBXMLUtils {

class SharedLibrary {
  public:
    SharedLibrary(const std::string &file_);
    SharedLibrary(const SharedLibrary& src);
    ~SharedLibrary();
    void* getAddress(const std::string &symbolName);
    const std::string file;
    const boost::posix_time::ptime writeTime;
    bool operator<(const SharedLibrary& b) const { return file<b.file; }
  private:
    std::string getLastError();
    void init();
#ifndef _WIN32
    void* handle;
#else
    HMODULE handle;
#endif
};

}

#endif
