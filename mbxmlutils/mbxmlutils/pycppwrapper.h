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

#ifndef _MBXMLUTILS_PYCPPWRAPPER_H_
#define _MBXMLUTILS_PYCPPWRAPPER_H_

#include <Python.h>
#include <string>
#include <vector>
#include <stdexcept>
#include <boost/filesystem/path.hpp>
#if PY_MAJOR_VERSION < 3
  #error "This file can only handle python >= 3"
#endif

namespace PythonCpp {

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

// make PyLong_Check a function
inline int PyLong_Check_func(PyObject *p) { return PyLong_Check(p); }
#undef PyLong_Check
#define PyLong_Check PythonCpp::PyLong_Check_func

// make PyUnicode_Check a function
inline int PyUnicode_Check_func(PyObject *p) { return PyUnicode_Check(p); }
#undef PyUnicode_Check
#define PyUnicode_Check PythonCpp::PyUnicode_Check_func

// create a valid GIL in the ctor and release it again in the dtor
class GilState {
  public:
    GilState() { state=PyGILState_Ensure(); }
    ~GilState() { PyGILState_Release(state); }
    GilState(const GilState &) = delete;
    GilState(GilState &&) = delete;
    GilState& operator=(const GilState &) = delete;
    GilState& operator=(GilState &&) = delete;
  private:
    PyGILState_STATE state;
};

// we use this for python object for c++ reference counting
class PyO {
  public:
    constexpr PyO() = default;
    explicit PyO(PyObject *src, bool srcIsBorrowedRef=false) : p(src) { if(srcIsBorrowedRef) Py_XINCREF(p); } // use srcIsBorrowedRef=true if src is a borrowed reference
    PyO(const PyO &r) : p(r.p) { Py_XINCREF(p); }
    PyO(PyO &&r)  noexcept : p(r.p) { r.p=nullptr; }
    ~PyO() {
      if(!p) return;
      // the dtor may be called outside of a valid Python GIL -> create one
      GilState gil;
      Py_XDECREF(p);
    }
    PyO& operator=(const PyO &r) { Py_XDECREF(p); p=r.p; Py_XINCREF(p); return *this; }
    PyO& operator=(PyO &&r) noexcept { Py_XDECREF(p); p=r.p; r.p=nullptr; return *this; }
    void reset() { Py_XDECREF(p); p=nullptr; }
    void reset(PyObject *src, bool srcIsBorrowedRef=false) { Py_XDECREF(p); p=src; if(srcIsBorrowedRef) Py_XINCREF(p); } // use srcIsBorrowedRef=true if src is a borrowed reference
    void swap(PyO &r) { PyObject *temp=p; p=r.p; r.p=temp; }
    PyObject* get(bool incRef=false) const { if(incRef) Py_XINCREF(p); return p; } // use incRef=true if the caller steals a reference of the returned PyObject
    long use_count() const { return p ? Py_REFCNT(p) : 0; }
    PyObject* operator->() const { return p; }
    PyO& incRef() { Py_XINCREF(p); return *this; } // use if the caller steals a reference of the returned PyObject
    operator bool() const { return p!=nullptr; }
  protected:
    PyObject *p{nullptr};
};

inline bool operator==(const PyO& l, const PyO& r) { return l.get()==r.get(); }
inline bool operator!=(const PyO& l, const PyO& r) { return l.get()!=r.get(); }
inline bool operator< (const PyO& l, const PyO& r) { return l.get()< r.get(); }
inline bool operator> (const PyO& l, const PyO& r) { return l.get()> r.get(); }
inline bool operator<=(const PyO& l, const PyO& r) { return l.get()<=r.get(); }
inline bool operator>=(const PyO& l, const PyO& r) { return l.get()>=r.get(); }

// A Python error exception object.
// Stores the file and line number of the C++ file where the error occured.
// Stores also the Python error objects type, value and traceback.
class PythonException : public std::exception {
  public:
    PythonException(const char *file_, int line_);
    virtual ~PythonException() noexcept override = default;
    std::string getFile() { return file; }
    int getLine() { return line; }
    PyO getType() { return type; }
    PyO getValue() { return value; }
    PyO getTraceback() { return traceback; }
    const char* what() const noexcept override;
  private:
    std::string file;
    int line;
    PyO type, value, traceback;
    mutable std::string msg;
};

// check for a python exception and throw a PythonException if one exists
void checkPythonError();

// helper struct to map the return type of the callPy function
// default: map to the same type
template<typename T>
struct MapRetType {
  using type = T;
  inline static type convert(const T &r) {
    return r;
  }
};
// specialization: map PyObject* to PyO
template<> struct MapRetType<PyObject*> {
  using type = PyO;
  inline static PyO convert(PyObject *r) {
    if(!r)
      throw std::runtime_error("Internal error: Expected python object but got NULL pointer and no python exception is set.");
    if(Py_REFCNT(r)<=0)
      throw std::runtime_error("Internal error: Expected python object but a python object with 0 refcount and no python exception is set.");
    return PyO(r);
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
  if(o && o.use_count()<=0)
    throw std::runtime_error("Internal error: access object with reference count <= 0. Check the source code.");
  return o.get();
}
// specialization:: map std::string to const char*
inline const char* convertArg(const std::string &o) {
  return o.c_str();
}

class DisableFPE {
  public:
    DisableFPE();
    ~DisableFPE();
  private:
    int savedFPE;
};

// Call Python function func with arguments args.
// Use the macro CALLPY or CALLPYB, see below.
template<typename PyRet, typename... PyArgs, typename... CallArgs>
inline typename MapRetType<PyRet>::type callPy(const char *file, int line, PyRet (*func)(PyArgs...), CallArgs&&... args) {
  PyRet ret;
  {
    // sympy/numpy/others may generate FPEs during calls -> disable FPE exceptions for python calls (save it first to restore it next)
    DisableFPE disableFPE;
    ret=func(convertArg(std::forward<CallArgs>(args))...);
  }
  if(PyErr_Occurred())
    throw PythonException(file, line);
  return MapRetType<PyRet>::convert(ret);
}

// Macro to call callPy(...)
// Use this macro to call a python function returning a new reference to a python object or any other return type.
// Note, if the python function steals a reference of any of this arguments you have to call arg.incRef() on
// each such arguments.
#define CALLPY(...) PythonCpp::callPy(__FILE__, __LINE__, __VA_ARGS__)

// Macro to call callPy(...).incRef()
// Use this macro to call a python function returning a borrowed reference to a python object.
// Note, if the python function steals a reference of any of this arguments you have to call arg.incRef() on
// each such arguments.
#define CALLPYB(...) PythonCpp::callPy(__FILE__, __LINE__, __VA_ARGS__).incRef()

// Initialize python giving main as program name to python.
// All path in sysPathAppend are added to python's sys.path array.
// If PYTHONHOME is not set all possiblePrefix dirs are tested for a possible PYTHONHOME
// and if one is found envvar is set.
// The calling thread holds the python GIL after this function finished. To release it
// call PyEval_SaveThread() or PyEval_ReleaseThread(PyThreadState_Get()). In this case ensure to acquire the GIL in each thread before
// each call to python using the class GilState
void initializePython(const boost::filesystem::path &main, const std::string &pythonVersion,
                      const std::vector<boost::filesystem::path> &sysPathPrepend={},
                      const std::vector<boost::filesystem::path> &sysPathAppend={},
                      const std::vector<boost::filesystem::path> &possiblePrefix={},
                      const std::vector<boost::filesystem::path> &PATHAppend={});

// a Py_BuildValue variant working like called with CALLPY
template<typename... Args>
PyO Py_BuildValue_(const char *format, Args&&... args) {
  return PyO(Py_BuildValue(format, convertArg(std::forward<Args>(args))...));
}

}

#endif
