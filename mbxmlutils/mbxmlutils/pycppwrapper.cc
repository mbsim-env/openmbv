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

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#  undef __STRICT_ANSI__ // to define _controlfp which is not part of ANSI and hence not defined in mingw
#  include <cfloat>
#  define __STRICT_ANSI__
#else
#  include <cfenv>
#endif
#include "pycppwrapper.h"
//#include <memory>
#include <sstream>
//#include <memory>
#include <boost/locale/encoding_utf.hpp> // gcc does not support <codecvt> yet -> use boost
#include <boost/filesystem.hpp>
//#include <cfenv>

namespace PythonCpp {

void checkPythonError() {
  if(PyErr_Occurred())
    throw PythonException("", 0);
}

void initializePython(const boost::filesystem::path &main, const std::string &pythonVersion,
                      const std::vector<boost::filesystem::path> &sysPathPrepend,
                      const std::vector<boost::filesystem::path> &sysPathAppend,
                      const std::vector<boost::filesystem::path> &possiblePrefix,
                      const std::vector<boost::filesystem::path> &PATHAppend) {
#ifdef _WIN32
  boost::filesystem::path dllDir("bin");
  std::string pathsep(";");
#else
  boost::filesystem::path dllDir("lib");
  std::string pathsep(":");
#endif

  boost::filesystem::path prefix;
  for(auto &p : possiblePrefix) {
    if(boost::filesystem::is_directory(p/"lib64"/("python"+pythonVersion)/"encodings") ||
       boost::filesystem::is_directory(p/"lib"/("python"+pythonVersion)/"encodings") ||
#ifdef _WIN32
       boost::filesystem::exists(p/"bin"/"python.exe") ||
       boost::filesystem::exists(p/"bin"/"python3.exe") ||
       boost::filesystem::exists(p/"bin"/"python.bat") ||
       boost::filesystem::exists(p/"bin"/"python3.bat")
#else
       boost::filesystem::exists(p/"bin"/"python") ||
       boost::filesystem::exists(p/"bin"/"python3")
#endif
    ) {
      prefix=p;
      break;
    }
  }

  boost::filesystem::path PYTHONHOME;
  const char *PH=getenv("PYTHONHOME");
  if(!PH) {
    PYTHONHOME = prefix;
    if(!PYTHONHOME.empty()) {
      // the string for putenv must have program life time
      static char PYTHONHOME_ENV[2048] { 0 };
      if(PYTHONHOME_ENV[0]==0)
        strcpy(PYTHONHOME_ENV, (std::string("PYTHONHOME=")+PYTHONHOME.string()).c_str());
      putenv(PYTHONHOME_ENV);
    }
  }
  else
    PYTHONHOME=PH;

#ifdef _WIN32
  {
    auto *pp = getenv("PYTHONPATH");
    std::string PYTHONPATHstr(pp ? pp : "");
    PYTHONPATHstr += (PYTHONPATHstr.empty() ? "" : pathsep)+(prefix/"lib").string();
    PYTHONPATHstr += pathsep+(prefix/"lib"/"lib-dynload").string();
    PYTHONPATHstr += pathsep+(prefix/"lib"/"site-packages").string();
    // the string for putenv must have program life time
    static char PYTHONPATHstr_ENV[2048] { 0 };
    if(PYTHONPATHstr_ENV[0]==0)
      strcpy(PYTHONPATHstr_ENV, (std::string("PYTHONPATH=")+PYTHONPATHstr).c_str());
    putenv(PYTHONPATHstr_ENV);
  }
#endif

  static auto mainW=boost::locale::conv::utf_to_utf<wchar_t>(main.string());
  #if __GNUC__ >= 11
    // python >= 3.8 has deprecated this call
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  #endif
  Py_SetProgramName(const_cast<wchar_t*>(mainW.c_str()));
  #if __GNUC__ >= 11
    #pragma GCC diagnostic pop
  #endif
  Py_InitializeEx(0);

  auto appendToPATH = [&pathsep](const boost::filesystem::path &path) {
    if(!boost::filesystem::is_directory(path))
      return;
#if defined(_WIN32) && PY_MAJOR_VERSION==3 && PY_MINOR_VERSION>=8
    PyO os=CALLPY(PyImport_ImportModule, "os");
    PyO os_add_dll_directory=CALLPY(PyObject_GetAttrString, os, "add_dll_directory");
    PyO arg(CALLPY(PyTuple_New, 1));
    PyO libdir(CALLPY(PyUnicode_FromString, path.string()));
    CALLPY(PyTuple_SetItem, arg, 0, libdir.incRef());
    CALLPY(PyObject_CallObject, os_add_dll_directory, arg);
#elif defined(_WIN32)
    // the string for putenv must have program life time
    static char PATH_ENV[2048] { 0 };
    if(PATH_ENV[0]==0) {
      auto *p=getenv("PATH");
      std::string PATH_OLD(p ? p : "");
      PATH_OLD += (PATH_OLD.empty() ? "" : pathsep)+path.string();
      strcpy(PATH_ENV, ("PATH="+PATH_OLD).c_str());
    }
    putenv(PATH_ENV);
#else
    // the string for putenv must have program life time
    static char LD_LIBRARY_PATH_ENV[2048] { 0 };
    if(LD_LIBRARY_PATH_ENV[0]==0) {
      auto *llp=getenv("LD_LIBRARY_PATH");
      std::string LD_LIBRARY_PATH_OLD(llp ? llp : "");
      LD_LIBRARY_PATH_OLD += (LD_LIBRARY_PATH_OLD.empty() ? "" : pathsep)+path.string();
      strcpy(LD_LIBRARY_PATH_ENV, ("LD_LIBRARY_PATH="+LD_LIBRARY_PATH_OLD).c_str());
    }
    putenv(LD_LIBRARY_PATH_ENV);
#endif
  };
  if(!PYTHONHOME.empty())
    appendToPATH(PYTHONHOME/dllDir);

  for(auto &p : PATHAppend)
    appendToPATH(p);

  // add to sys.path
  PyO sysPath(CALLPYB(PySys_GetObject, const_cast<char*>("path")));
  for(auto it = sysPathPrepend.rbegin(); it != sysPathPrepend.rend(); ++it)
    CALLPY(PyList_Insert, sysPath, 0, CALLPY(PyUnicode_FromString, it->string()));
  for(auto &p : sysPathAppend)
    CALLPY(PyList_Append, sysPath, CALLPY(PyUnicode_FromString, p.string()));
}

// c++ PythonException exception with the content of a python exception
PythonException::PythonException(const char *file_, int line_) : file(file_), line(line_) {
  // fetch if error has occured
  if(!PyErr_Occurred())
    throw std::runtime_error("Internal error: PythonException object created but no python error occured.");
  // fetch error objects and save objects
  PyObject *type_, *value_, *traceback_;
  PyErr_Fetch(&type_, &value_, &traceback_);
  type=PyO(type_);
  value=PyO(value_);
  traceback=PyO(traceback_);
}

const char* PythonException::what() const noexcept {
  if(!msg.empty())
    return msg.c_str();

  GilState gil;

  PyObject *savedstderr=nullptr;
  PyObject *io=nullptr;
  PyObject *fileIO=nullptr;
  PyObject *buf=nullptr;
  PyObject *getvalue=nullptr;
  PyObject *pybufstr=nullptr;
  #define RETURN(msg) \
    Py_XDECREF(savedstderr); \
    Py_XDECREF(io); \
    Py_XDECREF(fileIO); \
    Py_XDECREF(buf); \
    Py_XDECREF(getvalue); \
    Py_XDECREF(pybufstr); \
    return msg;

  // redirect stderr
  savedstderr=PySys_GetObject(const_cast<char*>("stderr"));
  Py_XINCREF(savedstderr);
  io=PyImport_ImportModule("io");
  if(!io)
    RETURN("Unable to create Python error message: cannot load io module.");
  fileIO=PyObject_GetAttrString(io, "StringIO"); // sys.stderr is a file in text mode
  if(!fileIO)
    RETURN("Unable to create Python error message: cannot get in memory file class.");
  buf=PyObject_CallObject(fileIO, nullptr);
  if(!buf)
    RETURN("Unable to create Python error message: cannot create new in memory file instance");
  if(PySys_SetObject(const_cast<char*>("stderr"), buf)!=0)
    RETURN("Unable to create Python error message: cannot redirect stderr");
  // restore error
  Py_XINCREF(type.get());
  Py_XINCREF(value.get());
  Py_XINCREF(traceback.get());
  PyErr_Restore(type.get(), value.get(), traceback.get());
  // print to redirected stderr
  PyErr_Print();
  // unredirect stderr
  if(PySys_SetObject(const_cast<char*>("stderr"), savedstderr)!=0)
    RETURN("Unable to create Python error message: cannot revert redirect stderr");
  // get redirected output as string
  getvalue=PyObject_GetAttrString(buf, "getvalue");
  if(!getvalue)
    RETURN("Unable to create Python error message: cannot get getvalue attribute");
  pybufstr=PyObject_CallObject(getvalue, nullptr);
  if(!pybufstr)
    RETURN("Unable to create Python error message: cannot get string from in memory file output");
  std::string str=PyUnicode_AsUTF8(pybufstr); // sys.stderr is a file in text mode
  if(PyErr_Occurred())
    RETURN("Unable to create Python error message: cannot get c string");
  std::stringstream strstr;
  strstr<<"Python exception";
#ifndef NDEBUG
  if(line>0)
    strstr<<" at "<<file<<":"<<line;
#endif
  strstr<<":"<<std::endl<<str;
  msg=strstr.str();
  RETURN(msg.c_str());
}

DisableFPE::DisableFPE() {
#ifdef _WIN32
  savedFPE=_controlfp(0, 0);
  _controlfp(~0, _MCW_EM);
#else
  savedFPE=fedisableexcept(FE_DIVBYZERO | FE_INEXACT | FE_INVALID | FE_OVERFLOW | FE_UNDERFLOW);
  assert(savedFPE!=-1);
#endif
}

DisableFPE::~DisableFPE() {
#ifdef _WIN32
  _controlfp(savedFPE, _MCW_EM);
#else
  assert(feenableexcept(savedFPE)!=-1);
#endif
}

}
