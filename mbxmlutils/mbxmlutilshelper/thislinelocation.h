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

#include <boost/dll.hpp>

namespace MBXMLUtils {

//! boost::dll::this_line_location does not return a absolute path and make the path absolute may fail
//! if the current dir has changed between the call and the time the lib was loaded.
//! This helper class fixes this.
//! Usage:
//! ThisLineLocation loc; // as a global variable at the code where you want to get the location.
//! boost::filesystem::path location=loc(); // to get the absolute path of the lib where loc is defined in.
class ThisLineLocation {
  public:
    //ThisLineLocation() : p(boost::filesystem::absolute(boost::dll::this_line_location())) {} we do not use this since this
    //  will required to link against boost::filesystem. Without this only boost::system is needed and boost::system
    //  an even be used in header only mode with "#define BOOST_ERROR_CODE_HEADER_ONLY"
    ThisLineLocation() {
      p=boost::dll::this_line_location();
      std::string pstr(p.string());
      if(pstr[0]=='/' || // Linux abs path
         pstr.substr(0,2)=="//" || pstr.substr(0,2)=="\\\\" || (isalpha(pstr[0]) && pstr[1]==':') ) // Windows abs path
        return;
      char buffer[2048];
      std::string curDir(getcwd(buffer, sizeof(buffer)));
      p=curDir+"/"+p.string();
    }
    boost::filesystem::path operator()() { return p; }
  private:
    boost::filesystem::path p;
};

}

#endif
