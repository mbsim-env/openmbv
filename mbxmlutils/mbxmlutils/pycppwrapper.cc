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
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
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
                      const std::vector<boost::filesystem::path> &sysPathAppend,
                      const std::vector<boost::filesystem::path> &possiblePrefix) {
  boost::filesystem::path PYTHONHOME;
  if(!getenv("PYTHONHOME")) {
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
        PYTHONHOME=p;
        break;
      }
    }
    if(!PYTHONHOME.empty()) {
      // the string for putenv must have program life time
      static std::string PYTHONHOME_ENV(std::string("PYTHONHOME=")+PYTHONHOME.string());
      putenv((char*)PYTHONHOME_ENV.c_str());
    }
  }

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

#ifdef _WIN32
  boost::filesystem::path dllDir("bin");
  std::string pathsep(";");
#else
  boost::filesystem::path dllDir("lib");
  std::string pathsep(":");
#endif
  if(!PYTHONHOME.empty()) {
#if _WIN32 && PY_MAJOR_VERSION==3 && PY_MINOR_VERSION>=8
    PyO os=CALLPY(PyImport_ImportModule, "os");
    PyO os_add_dll_directory=CALLPY(PyObject_GetAttrString, os, "add_dll_directory");
    PyO arg(CALLPY(PyTuple_New, 1));
    PyO libdir(CALLPY(PyUnicode_FromString, (PYTHONHOME/dllDir).string()));
    CALLPY(PyTuple_SetItem, arg, 0, libdir.incRef());
    CALLPY(PyObject_CallObject, os_add_dll_directory, arg);
#else
    // the string for putenv must have program life time
    std::string PATH_OLD(getenv("PATH"));
    static std::string PATH_ENV("PATH="+PATH_OLD+pathsep+(PYTHONHOME/dllDir).string());
    putenv((char*)PATH_ENV.c_str());
#endif
  }

  // add to sys.path
  PyO sysPath(CALLPYB(PySys_GetObject, const_cast<char*>("path")));
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

  // redirect stderr
  PyObject *savedstderr=PySys_GetObject(const_cast<char*>("stderr"));
  if(!savedstderr)
    return "Unable to create Python error message: no sys.stderr available.";
  PyObject *io=PyImport_ImportModule("io");
  if(!io)
    return "Unable to create Python error message: cannot load io module.";
  PyObject *fileIO=PyObject_GetAttrString(io, "StringIO"); // sys.stderr is a file in text mode
  if(!fileIO)
    return "Unable to create Python error message: cannot get in memory file class.";
  Py_DECREF(io);
  PyObject *buf=PyObject_CallObject(fileIO, nullptr);
  Py_DECREF(fileIO);
  if(!buf)
    return "Unable to create Python error message: cannot create new in memory file instance";
  if(PySys_SetObject(const_cast<char*>("stderr"), buf)!=0)
    return "Unable to create Python error message: cannot redirect stderr";
  // restore error
  PyErr_Restore(type.get(), value.get(), traceback.get());
  Py_XINCREF(type.get());
  Py_XINCREF(value.get());
  Py_XINCREF(traceback.get());
  // print to redirected stderr
  PyErr_Print();
  // unredirect stderr
  if(PySys_SetObject(const_cast<char*>("stderr"), savedstderr)!=0)
    return "Unable to create Python error message: cannot revert redirect stderr";
  // get redirected output as string
  PyObject *getvalue=PyObject_GetAttrString(buf, "getvalue");
  if(!getvalue)
    return "Unable to create Python error message: cannot get getvalue attribute";
  PyObject *pybufstr=PyObject_CallObject(getvalue, nullptr);
  if(!pybufstr)
    return "Unable to create Python error message: cannot get string from in memory file output";
  Py_DECREF(getvalue);
  Py_DECREF(buf);
  std::string str=PyUnicode_AsUTF8(pybufstr); // sys.stderr is a file in text mode
  if(PyErr_Occurred())
    return "Unable to create Python error message: cannot get c string";
  Py_DECREF(pybufstr);
  std::stringstream strstr;
  strstr<<"Python exception";
  if(!file.empty())
    strstr<<" at "<<file<<":"<<line;
  strstr<<":"<<std::endl<<str;
  msg=strstr.str();
  return msg.c_str();
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
