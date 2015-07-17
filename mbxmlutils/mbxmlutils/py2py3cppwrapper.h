#ifndef _PY2PY3CPPWRAPPER_H_
#define _PY2PY3CPPWRAPPER_H_

#include <Python.h>
#include <string>
#include <stdexcept>
#include <memory>
#include <sstream>
#include <boost/shared_ptr.hpp>
#include <boost/locale/encoding_utf.hpp>

namespace {
  void throwPythonException(const std::string &file, int line, const std::string &func);
}

namespace PythonCpp {

// initialize python giving main as program name to python
void initializePython(const std::string &main) {
#if PY_MAJOR_VERSION < 3
  Py_SetProgramName(const_cast<char*>(main.c_str()));
  Py_Initialize();
  const char *argv[]={"abc.py"};
  PySys_SetArgvEx(1, const_cast<char**>(argv), 0);
#else
  wstring wmain=locale::conv::utf_to_utf<wchar_t>(main);
  Py_SetProgramName(const_cast<wchar_t*>(wmain.c_str()));
  Py_Initialize();
  const wchar_t *argv[]={L"abc.py"};
  PySys_SetArgvEx(1, const_cast<wchar_t**>(argv), 0);
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
  if(!retc) return "";
  std::string ret=retc;
  Py_XDECREF(str);
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
#define PyLong_Check PyLong_Check_Py2Py3
#define PyLong_AsLong PyLong_AsLong_Py2Py3
#define PyUnicode_Check PyUnicode_Check_Py2Py3
#define PyUnicode_AsUTF8 PyUnicode_AsUTF8_Py2Py3

// we use this for python object for c++ reference counting
typedef boost::shared_ptr<PyObject> PyO;

// helper struct to map the return type of the cpy(...) function
// default: map to the same type
template<typename T>
struct TypeMap {
  typedef T type;
};
// specialization: map PyObject to PyO
template<> struct TypeMap<PyObject*> { typedef PyO type; };

// call a python function with c++ exception handling
// default: just return the argument (pass through) but throw if a python exception has occured
template<typename T>
inline typename TypeMap<T>::type cpy(T o, bool borrowedReference=false) {
  if(PyErr_Occurred())
    throwPythonException("", 0, "func");
  return o;
}
// specialization: for a PyObject* as argument return a c++ reference counted PyO but throw if a
// python exception has occured. Also handle borrowed python objects by incrementing the python ref count.
template<>
inline typename TypeMap<PyObject*>::type cpy(PyObject* o, bool borrowedReference) {
  if(PyErr_Occurred())
    throwPythonException("", 0, "func");
  if(!o)
    throw std::runtime_error("Internal error: Expected python object but got NULL pointer and not python exception is set.");
  if(borrowedReference)
    Py_INCREF(o);
  return PyO(o, &Py_DecRef);
}

}

namespace {

// throw a c++ runtime_error exception with the content of a python exception
void throwPythonException(const std::string &file, int line, const std::string &func) {
  // fetch the error
  PyObject *type, *value, *traceback;
  PyErr_Fetch(&type, &value, &traceback);
  // redirect stderr
  PyObject *savedstderr=PySys_GetObject(const_cast<char*>("stderr"));
  if(!savedstderr) throw std::runtime_error("Internal error: no sys.stderr available.");
  Py_INCREF(savedstderr);
  static PyObject *fileIO=NULL;
  if(!fileIO) {
    PyObject *io=PyImport_ImportModule("io");
    if(!io) {
      PyErr_Print(); // print pyexception to stderr
      throw std::runtime_error("Internal error: cannot load io module.");
    }
#if PY_MAJOR_VERSION < 3
    fileIO=PyObject_GetAttrString(io, "BytesIO"); // sys.stderr is a file is bytes mode
#else
    fileIO=PyObject_GetAttrString(io, "StringIO"); // sys.stderr is a file in text mode
#endif
    if(!fileIO) throw std::runtime_error("Internal error: cannot get in memory file class.");
    Py_XDECREF(io);
  }
  PyObject *buf=PyObject_CallObject(fileIO, NULL);
  if(!buf) throw std::runtime_error("Internal error: cannot create new in memory file instance");
  PySys_SetObject(const_cast<char*>("stderr"), buf);
  // restore the error and print to redirected stderr
  PyErr_Restore(type, value, traceback);
  PyErr_Print();
  // unredirect stderr
  PySys_SetObject(const_cast<char*>("stderr"), savedstderr);
  Py_XDECREF(savedstderr);
  // get redirected output as string
  PyObject *getvalue=PyObject_GetAttrString(buf, "getvalue");
  if(!getvalue) throw std::runtime_error("Internal error: cannot get getvalue attribute");
  PyObject *pybufstr=PyObject_CallObject(getvalue, NULL);
  Py_XDECREF(getvalue);
  if(!pybufstr) throw std::runtime_error("Internal error: cannot get string from in memory file output");
  Py_XDECREF(buf);
#if PY_MAJOR_VERSION < 3
  std::string str=PyBytes_AsString(pybufstr); // sys.stderr is a file in bytes mode
#else
  std::string str=PyUnicode_AsUTF8(pybufstr); // sys.stderr is a file in text mode
#endif
  if(PyErr_Occurred()) {
    PyErr_Print(); // print pyexception to stderr
    throw std::runtime_error("Internal error: cannot get the string from in memory file output string");
  }
  Py_XDECREF(pybufstr);
  std::stringstream msg;
  msg<<"Python exception";
  if(!file.empty())
    msg<<" at "<<file<<":"<<line<<":"<<func;
  msg<<":"<<std::endl<<str;;
  throw std::runtime_error(msg.str());
}

}

#endif
