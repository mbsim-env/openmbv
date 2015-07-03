#include <config.h>

// python includes
#include <Python.h> // due to some bugs in python 3.2 we need to include this first
#define PY_ARRAY_UNIQUE_SYMBOL mbxmlutils_pyeval_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

// normal includes
#include "pyeval.h"

// other includes
#include <boost/lexical_cast.hpp>

using namespace std;
using namespace boost;
using namespace boost::filesystem;
using namespace xercesc;

namespace {

// forward declaration
void throwPyExcIfOccurred();

// python object deleter/deref helper function
void pydecref(PyObject *p) {
  Py_XDECREF(p);
}

// a PyObject smart pointer
typedef shared_ptr<PyObject> PyO;

// a PyObject smart pointer creator
PyO mkpyo(PyObject *p=NULL, bool borrowedReference=false, bool simpleThrow=false) {
  if(simpleThrow && PyErr_Occurred())
    throw runtime_error("Internal error: During exception handling of a python exception another exception occured.");
  throwPyExcIfOccurred();
  if(!p)
    throw runtime_error("No python object provided.");
  if(borrowedReference)
    Py_INCREF(p);
  return PyO(p, &pydecref);
}

// throw a c++ runtime_error exception with the content of a python exception if occurred
void throwPyExcIfOccurred() {
  if(!PyErr_Occurred())
    return;
  // fetch the exception first
  PyObject *type_, *value_, *traceback_;
  PyErr_Fetch(&type_, &value_, &traceback_);
  PyO type=mkpyo(type_, false, true);
  PyO value=mkpyo(value_, false, true);
  PyO traceback=mkpyo(traceback_, false, true);
  // now print the exception to a stream
  stringstream str;
  // get traceback if avaialble
  if(traceback) {
    static PyObject* stringIO=NULL;
    if(!stringIO) {
#if PY_MAJOR_VERSION < 3
      PyO io=mkpyo(PyImport_ImportModule("StringIO"), false, true);
#else
      PyO io=mkpyo(PyImport_ImportModule("io"), false, true);
#endif
      stringIO=PyObject_GetAttrString(io.get(), "StringIO");
      throwPyExcIfOccurred();
    }
    PyO file=mkpyo(PyObject_CallObject(stringIO, NULL), false, true);
    PyTraceBack_Print(traceback.get(), file.get());
    PyO getvalue=mkpyo(PyObject_GetAttrString(file.get(), "getvalue"), false, true);
    PyO cont=mkpyo(PyObject_CallObject(getvalue.get(), NULL), false, true);
    str<<PyBytes_AsString(PyUnicode_AsUTF8String(cont.get()));
  }
  else
    str<<"Python exception:\n";
  // get exception message
  str<<reinterpret_cast<PyTypeObject*>(type.get())->tp_name<<": ";
  if(PyUnicode_Check(value.get()))
    str<<PyBytes_AsString(PyUnicode_AsUTF8String(value.get()))<<"\n";
  if(PyBytes_Check(value.get()))
    str<<PyBytes_AsString(value.get())<<"\n";
  else if(PyCallable_Check(type.get())) {
//MFMF    PyO excObj=mkpyo(PyObject_CallObject(type.get(), value.get()), false, true);
//MFMF    PyO valueStr=mkpyo(PyObject_Str(excObj.get()), false, true);
//MFMF    str<<PyBytes_AsString(PyUnicode_AsUTF8String(valueStr.get()))<<"\n";
    str<<"MFMFxxxxxxxxxxxx";
  }
  else
    str<<"Internal error: should not happen\n";
  // throw as c++ runtime_error exception
  PyErr_Clear();
  throw runtime_error(str.str());
}

inline PyO C(const shared_ptr<void> &value) {
  return static_pointer_cast<PyObject>(value);
}

vector<double> cast_vector_double(const shared_ptr<void> &value, bool checkOnly);
vector<vector<double> > cast_vector_vector_double(const shared_ptr<void> &value, bool checkOnly);
string cast_string(const shared_ptr<void> &value, bool checkOnly);

}

namespace MBXMLUtils {

bool PyEval::initialized=false;

XMLUTILS_EVAL_REGISTER(PyEval)

PyEval::PyEval(vector<path> *dependencies_) {
  // we initialize python only ones and never deinit it since numpy cannot be reinitialized
  if(initialized)
    return;

  Py_SetProgramName(const_cast<wchar_t*>(L"mbxmlutilspp"));
  Py_Initialize();
  const wchar_t *argv[]={L""};
  PySys_SetArgvEx(1, const_cast<wchar_t**>(argv), 0);
  _import_array();
  throwPyExcIfOccurred();
  initialized=true;
}

PyEval::~PyEval() {
  // Py_Finalize(); // we never deinit python it since numpy cannot be reinitialized
}

void PyEval::addPath(const path &dir, const DOMElement *e) {
  throw DOMEvalException("mfmf addPath", e);
}

bool PyEval::valueIsOfType(const shared_ptr<void> &value, ValueType type) const {
  PyO v=C(value);
  switch(type) {
    case ScalarType: return PyFloat_Check(v.get());
    case VectorType: try { ::cast_vector_double(value, true); return true; } catch(...) { return false; }
    case MatrixType: try { ::cast_vector_vector_double(value, true); return true; } catch(...) { return false; }
    case StringType: try { ::cast_string(value, true); return true; } catch(...) { return false; }
    case SXFunctionType: return false;//MFMF
  }
  throw DOMEvalException("Internal error: Unknwon ValueType.");
}

map<path, pair<path, bool> >& PyEval::requiredFiles() const {
  //MFMF
  static map<path, pair<path, bool> > files;
  return files;
}

shared_ptr<void> PyEval::createSwigByTypeName(const string &typeName) const {
  throw runtime_error("mfmf createSwigByTypeName");
}

shared_ptr<void> PyEval::callFunction(const string &name, const vector<shared_ptr<void> >& args) const {
  throw runtime_error("mfmf callFunction");
}

shared_ptr<void> PyEval::fullStringToValue(const string &str, const DOMElement *e) const {
  // check some common string to avoid time consiming evaluation
  // check true and false
  if(str=="True") return mkpyo(PyBool_FromLong(1));
  if(str=="False") return mkpyo(PyBool_FromLong(0));
  // check for floating point values
  try { return mkpyo(PyFloat_FromDouble(lexical_cast<double>(str))); }
  catch(const boost::bad_lexical_cast &) {}
  // no common string detected -> evaluate using python now

  // restore current dir on exit and change current dir
  PreserveCurrentDir preserveDir;
  if(e) {
    path chdir=E(e)->getOriginalFilename().parent_path();
    if(!chdir.empty())
      current_path(chdir);
  }
  //MFMF
  {
    PyO globals=mkpyo(PyDict_New());
    PyDict_SetItemString(globals.get(), "__builtins__", PyEval_GetBuiltins());
    PyO locals=mkpyo(PyDict_New());
    mkpyo(PyRun_String("a=5;b=9;import os; from os import path; from os import *", Py_file_input, globals.get(), locals.get()));
    PyObject *key;
    Py_ssize_t pos=0;
    while(PyDict_Next(globals.get(), &pos, &key, NULL))
      cerr<<"MFMFg "<<PyBytes_AsString(PyUnicode_AsUTF8String(key))<<endl;
    pos=0;
    while(PyDict_Next(locals.get(), &pos, &key, NULL))
      cerr<<"MFMFl "<<PyBytes_AsString(PyUnicode_AsUTF8String(key))<<endl;
    exit(0);
  }
  //MFMF

  // pyhton globals (fill with builtins
  PyO globals=mkpyo(PyDict_New());
  PyDict_SetItemString(globals.get(), "__builtins__", PyEval_GetBuiltins());

  // pyhton locals (fill with current parameters)
  PyO locals=mkpyo(PyDict_New());
  for(map<string, shared_ptr<void> >::const_iterator i=currentParam.begin(); i!=currentParam.end(); i++)
    PyDict_SetItemString(locals.get(), i->first.c_str(), C(i->second).get());

  PyO ret;
  try {
    // evaluate as expression and save result in ret
    ret=mkpyo(PyRun_String(str.c_str(), Py_eval_input, globals.get(), locals.get()));
  }
  catch(const runtime_error&) { // on failure ...
    PyErr_Clear();
    try {
      // ... evaluate as statement
      mkpyo(PyRun_String(str.c_str(), Py_file_input, globals.get(), locals.get()));
    }
    catch(const runtime_error& ex) { // on failure -> report error
      PyErr_Clear();
      throw DOMEvalException(string(ex.what())+"Unable to evaluate expression: "+str, e);
    }
    try {
      // get 'ret' variable from statement
      ret=mkpyo(PyDict_GetItemString(locals.get(), "ret"), true);
    }
    catch(const runtime_error&) {
      // 'ret' variable not found or invalid expression
      throw DOMEvalException("Invalid expression or statement does not define the 'ret' variable in expression: "+str, e);
    }
  }
  // return result
  return ret;
}

void* PyEval::getSwigThis(const shared_ptr<void> &value) const {
  throw runtime_error("mfmf getSwigThis");
}

string PyEval::getSwigType(const shared_ptr<void> &value) const {
  throw runtime_error("mfmf getSwigType");
}

double PyEval::cast_double(const shared_ptr<void> &value) const {
  double ret=PyFloat_AsDouble(C(value).get());
  throwPyExcIfOccurred();
  return ret;
}

vector<double> PyEval::cast_vector_double(const shared_ptr<void> &value) const {
  return ::cast_vector_double(value, false);
}

vector<vector<double> >PyEval::cast_vector_vector_double(const shared_ptr<void> &value) const {
  return ::cast_vector_vector_double(value, false);
}

string PyEval::cast_string(const shared_ptr<void> &value) const {
  return ::cast_string(value, false);
}

shared_ptr<void> PyEval::create_double(const double& v) const {
  return mkpyo(PyFloat_FromDouble(v));
}

shared_ptr<void> PyEval::create_vector_double(const vector<double>& v) const {
  npy_intp dims[1];
  dims[0]=v.size();
  PyO ret=mkpyo(PyArray_SimpleNew(1, dims, NPY_DOUBLE));
  copy(v.begin(), v.end(), static_cast<double*>(PyArray_GETPTR1(reinterpret_cast<PyArrayObject*>(ret.get()), 0)));
  return ret;
}

shared_ptr<void> PyEval::create_vector_vector_double(const vector<vector<double> >& v) const {
  npy_intp dims[2];
  dims[0]=v.size();
  dims[1]=v[0].size();
  PyO ret=mkpyo(PyArray_SimpleNew(2, dims, NPY_DOUBLE));
  int r=0;
  for(vector<vector<double> >::const_iterator it=v.begin(); it!=v.end(); ++it, ++r)
    copy(it->begin(), it->end(), static_cast<double*>(PyArray_GETPTR2(reinterpret_cast<PyArrayObject*>(ret.get()), r, 0)));
  return ret;
}

shared_ptr<void> PyEval::create_string(const string& v) const {
  return mkpyo(PyUnicode_FromString(v.c_str()));
}

}

namespace {

vector<double> cast_vector_double(const shared_ptr<void> &value, bool checkOnly) {
  PyO v=C(value);
  if(PyList_Check(v.get())) {
    size_t size=PyList_Size(v.get());
    vector<double> ret(size);
    for(size_t i=0; i<size; ++i) {
      ret[i]=PyFloat_AsDouble(PyList_GetItem(v.get(), i));
      throwPyExcIfOccurred();
    }
    return ret;
  }
  if(PyArray_Check(v.get())) {
    PyArrayObject *a=reinterpret_cast<PyArrayObject*>(v.get());
    if(PyArray_NDIM(a)!=1)
      throw runtime_error("Value is not of type vector (wrong dimension).");
    int type=PyArray_TYPE(a);
    if(type!=NPY_DOUBLE && type!=NPY_INT)
      throw runtime_error("Value is not of type vector (wrong element type).");
    if(checkOnly)
      return vector<double>();
    npy_intp *dims=PyArray_SHAPE(a);
    vector<double> ret(dims[0]);
    for(size_t i=0; i<dims[0]; ++i) {
      if(type==NPY_DOUBLE)
        ret[i]=*static_cast<double*>(PyArray_GETPTR1(a, i));
      if(type==NPY_INT)
        ret[i]=*static_cast<int*>(PyArray_GETPTR1(a, i));
    }
    return ret;
  }
  throw runtime_error("Value is not of type vector (wrong type).");
}

vector<vector<double> > cast_vector_vector_double(const shared_ptr<void> &value, bool checkOnly) {
  PyO v=C(value);
  if(PyList_Check(v.get())) {
    size_t row=PyList_Size(v.get());
    size_t col=PyList_Size(PyList_GetItem(v.get(), 0));
    throwPyExcIfOccurred();
    vector<vector<double> > ret(row, vector<double>(col));
    for(size_t r=0; r<row; ++r)
      for(size_t c=0; c<col; ++c) {
        ret[r][c]=PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(v.get(), r), c));
        throwPyExcIfOccurred();
      }
    return ret;
  }
  if(PyArray_Check(v.get())) {
    PyArrayObject *a=reinterpret_cast<PyArrayObject*>(v.get());
    if(PyArray_NDIM(a)!=2)
      throw runtime_error("Value is not of type matrix (wrong dimension).");
    int type=PyArray_TYPE(a);
    if(type!=NPY_DOUBLE && type!=NPY_INT)
      throw runtime_error("Value is not of type matrix (wrong element type).");
    if(checkOnly)
      return vector<vector<double> >();
    npy_intp *dims=PyArray_SHAPE(a);
    vector<vector<double> > ret(dims[0], vector<double>(dims[1]));
    for(size_t r=0; r<dims[0]; ++r)
      for(size_t c=0; c<dims[1]; ++c) {
        if(type==NPY_DOUBLE)
          ret[r][c]=*static_cast<double*>(PyArray_GETPTR2(a, r, c));
        if(type==NPY_INT)
          ret[r][c]=*static_cast<int*>(PyArray_GETPTR2(a, r, c));
      }
    return ret;
  }
  throw runtime_error("Value is not of type matrix (wrong type).");
}

string cast_string(const shared_ptr<void> &value, bool checkOnly) {
  string ret=PyBytes_AsString(PyUnicode_AsUTF8String(C(value).get()));
  throwPyExcIfOccurred();
  return ret;
}

}
