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

#ifndef _PY2PY3CPPWRAPPER_H_
#define _PY2PY3CPPWRAPPER_H_

#include <Python.h>
#include <string>
#include <stdexcept>
#include <memory>
#include <sstream>
#include <boost/shared_ptr.hpp>
#include <boost/locale/encoding_utf.hpp>

namespace PythonCpp {

// initialize python giving main as program name to python
void initializePython(const std::string &main, const std::vector<std::string> &args=std::vector<std::string>()) {
#if PY_MAJOR_VERSION < 3
  Py_SetProgramName(const_cast<char*>(main.c_str()));
  Py_InitializeEx(0);
  std::vector<char*> argv(args.size());
  for(size_t i=0; i<args.size(); ++i)
    argv[i]=const_cast<char*>(args[i].c_str());
  PySys_SetArgvEx(args.size(), &argv[0], 0);
#else
  Py_SetProgramName(const_cast<wchar_t*>(boost::locale::conv::utf_to_utf<wchar_t>(main).c_str()));
  Py_InitializeEx(0);
  std::vector<wchar_t*> argv(args.size());
  std::vector<std::wstring> argsw;
  argsw.reserve(args.size());
  for(size_t i=0; i<args.size(); ++i) {
    argsw.push_back(boost::locale::conv::utf_to_utf<wchar_t>(args[i]));
    argv[i]=const_cast<wchar_t*>(argsw[i].c_str());
  }
  PySys_SetArgvEx(args.size(), &argv[0], 0);
#endif
}

// wrap some python 3 function to also work in python 2 (the wrappers have suffix _Py2Py2
// and are later defined without the suffix as a macro)

inline bool PyLong_Check_Py2Py3(PyObject *o) {
#if PY_MAJOR_VERSION < 3
  return PyLong_Check(o) || PyInt_Check(o);
#else
  return PyLong_Check(o);
#endif
}

inline long PyLong_AsLong_Py2Py3(PyObject *o) {
#if PY_MAJOR_VERSION < 3
  if(PyInt_Check(o))
    return PyInt_AsLong(o);
  return PyLong_AsLong(o);
#else
  return PyLong_AsLong(o);
#endif
}

inline bool PyUnicode_Check_Py2Py3(PyObject *o) {
#if PY_MAJOR_VERSION < 3
  return PyUnicode_Check(o) || PyString_Check(o);
#else
  return PyUnicode_Check(o);
#endif
}

// we cannot return char* here since for python 3 this would lead to a reference to a temporary
inline std::string PyUnicode_AsUTF8_Py2Py3(PyObject *o) {
#if PY_MAJOR_VERSION < 3
  if(PyString_Check(o)) {
    char *retc=PyString_AsString(o);
    if(!retc) return "";
    return retc;
  }
  PyObject *str=PyUnicode_AsUTF8String(o);
  if(!str) return "";
  char *retc=PyString_AsString(str);
  if(!retc) {
    Py_DECREF(str);
    return "";
  }
  std::string ret=retc;
  Py_DECREF(str);
  return ret;
#else
  char *retc=PyUnicode_AsUTF8(o);
  if(!retc) return "";
  return retc;
#endif
}

// now define the wrappers without the suffix: now the wrappers are active!!!!!

#undef PyLong_Check
#undef PyUnicode_Check
#define PyLong_Check PythonCpp::PyLong_Check_Py2Py3
#define PyLong_AsLong PythonCpp::PyLong_AsLong_Py2Py3
#define PyUnicode_Check PythonCpp::PyUnicode_Check_Py2Py3
#define PyUnicode_AsUTF8 PythonCpp::PyUnicode_AsUTF8_Py2Py3

// make PyRun_String a function
inline PyObject* PyRun_String_func(const char *str, int start, PyObject *globals, PyObject *locals) { return PyRun_String(str, start, globals, locals); }
#undef PyRun_String
#define PyRun_String PythonCpp::PyRun_String_func

// make PyFloat_Check a function
inline int PyFloat_Check_func(PyObject *p) { return PyFloat_Check(p); }
#undef PyFloat_Check
#define PyFloat_Check PythonCpp::PyFloat_Check_func

// make PyFloat_AS_DOUBLE a function
inline int PyFloat_AS_DOUBLE_func(PyObject *p) { return PyFloat_AS_DOUBLE(p); }
#undef PyFloat_AS_DOUBLE
#define PyFloat_AS_DOUBLE PythonCpp::PyFloat_AS_DOUBLE_func

// make PyBool_Check a function
inline int PyBool_Check_func(PyObject *p) { return PyBool_Check(p); }
#undef PyBool_Check
#define PyBool_Check PythonCpp::PyBool_Check_func

// make PyList_Check a function
inline int PyList_Check_func(PyObject *p) { return PyList_Check(p); }
#undef PyList_Check
#define PyList_Check PythonCpp::PyList_Check_func

// make PyObject_TypeCheck a function
inline int PyObject_TypeCheck_func(PyObject *p, PyTypeObject *type) { return PyObject_TypeCheck(p, type); }
#undef PyObject_TypeCheck
#define PyObject_TypeCheck PythonCpp::PyObject_TypeCheck_func

// we use this for python object for c++ reference counting
typedef boost::shared_ptr<PyObject> PyO;

// A Python error exception object.
// Stores the file and line number of the C++ file where the error occured.
// Stores also the Python error objects type, value and traceback.
class PythonException : public std::exception {
  public:
    PythonException(const char *file_, int line_);
    ~PythonException() throw() {}
    std::string getFile() { return file; }
    int getLine() { return line; }
    PyO getType() { return type; }
    PyO getValue() { return value; }
    PyO getTraceback() { return traceback; }
    const char* what() const throw();
  private:
    std::string file;
    int line;
    PyO type, value, traceback;
    mutable std::string msg;
};

// check for a python exception and throw a PythonException if one exists
void checkPythonError() {
  if(PyErr_Occurred())
    throw PythonException("", 0);
}

// helper struct to map the return type of the callPy function
// default: map to the same type
template<typename T>
struct MapRetType {
  typedef T type;
  inline static type convert(const T &r) {
    return r;
  }
};
// specialization: map PyObject* to PyO
template<> struct MapRetType<PyObject*> {
  typedef PyO type;
  inline static PyO convert(PyObject *r) {
    if(!r)
      throw std::runtime_error("Internal error: Expected python object but got NULL pointer and not python exception is set.");
    return PyO(r, &Py_DecRef);
  }
};

// helper function to map the argument types of the callPy function
// default: map to the same type
template<typename Arg>
inline Arg convertArg(const Arg &arg) {
  return arg;
}
// specialization:: map PyO to PyObject*
inline PyObject* convertArg(const PyO &o) {
  if(o && Py_REFCNT(o.get())<=0)
    throw std::runtime_error("Internal error: access object with reference count <= 0. Check the source code.");
  return o.get();
}
// specialization:: map std::string to const char*
inline const char* convertArg(const std::string &o) {
  return o.c_str();
}

// Call Python function func with arguments args.
// Use the macro CALLPY or CALLPYB, see below.
/* when we used c++11 replace all the following code with this c++11 code:
template<typename PyRet, typename... PyArgs, typename... CallArgs>
inline typename MapRetType<PyRet>::type callPy(const char *file, int line, PyRet (*func)(PyArgs...), CallArgs... args) {
  PyRet ret=func(convertArg(args1)...);
  if(PyErr_Occurred())
    throw PythonException(file, line);
  return MapRetType<PyRet>::convert(ret);
}
*/
// 0 arg
template<typename PyRet>
inline typename MapRetType<PyRet>::type callPy(const char *file, int line, PyRet (*func)()) {
  PyRet ret=func();
  if(PyErr_Occurred())
    throw PythonException(file, line);
  return MapRetType<PyRet>::convert(ret);
}
// 1 arg
template<typename PyRet, typename PyArgs1, typename CallArgs1>
inline typename MapRetType<PyRet>::type callPy(const char *file, int line, PyRet (*func)(PyArgs1), CallArgs1 args1) {
  PyRet ret=func(convertArg(args1));
  if(PyErr_Occurred())
    throw PythonException(file, line);
  return MapRetType<PyRet>::convert(ret);
}
// 2 arg
template<typename PyRet, typename PyArgs1, typename PyArgs2, typename CallArgs1, typename CallArgs2>
inline typename MapRetType<PyRet>::type callPy(const char *file, int line, PyRet (*func)(PyArgs1, PyArgs2), CallArgs1 args1, CallArgs2 args2) {
  PyRet ret=func(convertArg(args1), convertArg(args2));
  if(PyErr_Occurred())
    throw PythonException(file, line);
  return MapRetType<PyRet>::convert(ret);
}
// 3 arg
template<typename PyRet, typename PyArgs1, typename PyArgs2, typename PyArgs3, typename CallArgs1, typename CallArgs2, typename CallArgs3>
inline typename MapRetType<PyRet>::type callPy(const char *file, int line, PyRet (*func)(PyArgs1, PyArgs2, PyArgs3), CallArgs1 args1, CallArgs2 args2, CallArgs3 args3) {
  PyRet ret=func(convertArg(args1), convertArg(args2), convertArg(args3));
  if(PyErr_Occurred())
    throw PythonException(file, line);
  return MapRetType<PyRet>::convert(ret);
}
// 4 arg
template<typename PyRet, typename PyArgs1, typename PyArgs2, typename PyArgs3, typename PyArgs4, typename CallArgs1, typename CallArgs2, typename CallArgs3, typename CallArgs4>
inline typename MapRetType<PyRet>::type callPy(const char *file, int line, PyRet (*func)(PyArgs1, PyArgs2, PyArgs3, PyArgs4), CallArgs1 args1, CallArgs2 args2, CallArgs3 args3, CallArgs4 args4) {
  PyRet ret=func(convertArg(args1), convertArg(args2), convertArg(args3), convertArg(args4));
  if(PyErr_Occurred())
    throw PythonException(file, line);
  return MapRetType<PyRet>::convert(ret);
}
// 5 arg
template<typename PyRet, typename PyArgs1, typename PyArgs2, typename PyArgs3, typename PyArgs4, typename PyArgs5, typename CallArgs1, typename CallArgs2, typename CallArgs3, typename CallArgs4, typename CallArgs5>
inline typename MapRetType<PyRet>::type callPy(const char *file, int line, PyRet (*func)(PyArgs1, PyArgs2, PyArgs3, PyArgs4, PyArgs5), CallArgs1 args1, CallArgs2 args2, CallArgs3 args3, CallArgs4 args4, CallArgs5 args5) {
  PyRet ret=func(convertArg(args1), convertArg(args2), convertArg(args3), convertArg(args4), convertArg(args5));
  if(PyErr_Occurred())
    throw PythonException(file, line);
  return MapRetType<PyRet>::convert(ret);
}

// Macro to call callPy(...)
// Use this macro to call a python function returning a new reference to a python object or any other return type.
// Note, if the python function steals a reference of any of this arguments you have to call PyIncRef on
// each such arguments after the call.
#define CALLPY(...) PythonCpp::callPy(__FILE__, __LINE__, __VA_ARGS__)

// Macro to call PyIncRef(callPy(...))
// Use this macro to call a python function returning a borrowed reference to a python object.
// Note, if the python function steals a reference of any of this arguments you have to call PyIncRef on
// each such arguments after the call.
#define CALLPYB(...) PythonCpp::PyIncRef(PythonCpp::callPy(__FILE__, __LINE__, __VA_ARGS__))

// increment the reference count of a python object.
// This function MUST be called after a reference is stolen by a python call.
inline const PyO& PyIncRef(const PyO &o) {
  if(Py_REFCNT(o.get())<=0)
    throw std::runtime_error("Internal error: Access object with reference count <= 0. Check the source code.");
  Py_XINCREF(o.get());
  return o;
}

// c++ PythonException exception with the content of a python exception
PythonException::PythonException(const char *file_, int line_) : file(file_), line(line_) {
  // fetch if error has occured
  if(!PyErr_Occurred())
    throw std::runtime_error("Internal error: PythonException object created but no python error occured.");
  // fetch error objects and save objects
  PyObject *type_, *value_, *traceback_;
  PyErr_Fetch(&type_, &value_, &traceback_);
  type=PyO(type_, &Py_DecRef);
  value=PyO(value_, &Py_DecRef);
  traceback=PyO(traceback_, &Py_DecRef);
}

const char* PythonException::what() const throw() {
  if(!msg.empty())
    return msg.c_str();

  // redirect stderr
  PyObject *savedstderr=PySys_GetObject(const_cast<char*>("stderr"));
  if(!savedstderr)
    throw std::runtime_error("Internal error: no sys.stderr available.");
  PyObject *io=PyImport_ImportModule("io");
  if(!io)
    throw std::runtime_error("Internal error: cannot load io module.");
#if PY_MAJOR_VERSION < 3
  PyObject *fileIO=PyObject_GetAttrString(io, "BytesIO"); // sys.stderr is a file is bytes mode
#else
  PyObject *fileIO=PyObject_GetAttrString(io, "StringIO"); // sys.stderr is a file in text mode
#endif
  if(!fileIO)
    throw std::runtime_error("Internal error: cannot get in memory file class.");
  Py_DECREF(io);
  PyObject *buf=PyObject_CallObject(fileIO, NULL);
  Py_DECREF(fileIO);
  if(!buf)
    throw std::runtime_error("Internal error: cannot create new in memory file instance");
  if(PySys_SetObject(const_cast<char*>("stderr"), buf)!=0)
    throw std::runtime_error("Internal error: cannot redirect stderr");
  // restore error
  PyErr_Restore(type.get(), value.get(), traceback.get());
  Py_XINCREF(type.get());
  Py_XINCREF(value.get());
  Py_XINCREF(traceback.get());
  // print to redirected stderr
  PyErr_Print();
  // unredirect stderr
  if(PySys_SetObject(const_cast<char*>("stderr"), savedstderr)!=0)
    throw std::runtime_error("Internal error: cannot revert redirect stderr");
  // get redirected output as string
  PyObject *getvalue=PyObject_GetAttrString(buf, "getvalue");
  if(!getvalue)
    throw std::runtime_error("Internal error: cannot get getvalue attribute");
  PyObject *pybufstr=PyObject_CallObject(getvalue, NULL);
  if(!pybufstr)
    throw std::runtime_error("Internal error: cannot get string from in memory file output");
  Py_DECREF(getvalue);
  Py_DECREF(buf);
#if PY_MAJOR_VERSION < 3
  char *strc=PyBytes_AsString(pybufstr); // sys.stderr is a file in bytes mode
  if(!strc)
    throw std::runtime_error("Internal error: cannot get c string");
  std::string str(strc);
#else
  std::string str=PyUnicode_AsUTF8(pybufstr); // sys.stderr is a file in text mode
  if(PyErr_Occurred())
    throw std::runtime_error("Internal error: cannot get c string");
#endif
  Py_DECREF(pybufstr);
  std::stringstream strstr;
  strstr<<"Python exception";
  if(!file.empty())
    strstr<<" at "<<file<<":"<<line;
  strstr<<":"<<std::endl<<str;;
  msg=strstr.str();
  return msg.c_str();
}

}

#endif
