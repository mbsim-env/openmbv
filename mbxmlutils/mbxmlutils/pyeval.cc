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
#include <boost/algorithm/string/trim.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/join.hpp>
#include <mbxmlutilshelper/getinstallpath.h>
#include "pyeval-config.h"

using namespace std;
using namespace boost;
using namespace boost::filesystem;
using namespace xercesc;
using namespace PythonCpp;

namespace {

inline PyO C(const shared_ptr<void> &value) {
  return static_pointer_cast<PyObject>(value);
}

vector<double> cast_vector_double(const shared_ptr<void> &value, bool checkOnly);
vector<vector<double> > cast_vector_vector_double(const shared_ptr<void> &value, bool checkOnly);
string cast_string(const shared_ptr<void> &value, bool checkOnly);

}

namespace MBXMLUtils {

bool PyEval::initialized=false;
PyO PyEval::mbxmlutils;
PyO PyEval::numpy;

XMLUTILS_EVAL_REGISTER(PyEval)

PyEval::PyEval(vector<path> *dependencies_) {
  // we initialize python only ones and never deinit it since numpy cannot be reinitialized
  if(initialized)
    return;

  initializePython("mbxmlutilspp");
  cpy(_import_array());
  initialized=true;

  PyO path=cpy(PySys_GetObject(const_cast<char*>("path")), true);
  PyO mbxmlutilspath=cpy(PyUnicode_FromString((getInstallPath()/"share"/"mbxmlutils"/"python").string().c_str()));
  cpy(PyList_Append(path.get(), mbxmlutilspath.get()));
  PyO casadipath=cpy(PyUnicode_FromString(CASADI_PREFIX "/python2.7/site-packages/casadi"));
  cpy(PyList_Append(path.get(), casadipath.get()));

  mbxmlutils=cpy(PyImport_ImportModule("mbxmlutils"));
  numpy=cpy(PyImport_ImportModule("numpy"));
  casadiValue=cpy(PyImport_ImportModule("casadi"));

  currentImport=cpy(PyDict_New());
}

PyEval::~PyEval() {
  // Py_Finalize(); // we never deinit python it since numpy cannot be reinitialized
}

void PyEval::addImport(const string &code, const DOMElement *e, bool deprecated) {
  if(deprecated)
    throw DOMEvalException("The deprecated <searchPath .../> element is not supported.", e);

  // restore current dir on exit and change current dir
  PreserveCurrentDir preserveDir;
  if(e) {
    path chdir=E(e)->getOriginalFilename().parent_path();
    if(!chdir.empty())
      current_path(chdir);
  }

  // python globals (fill with builtins)
  PyO globals=cpy(PyDict_New());
  cpy(PyDict_SetItemString(globals.get(), "__builtins__", cpy(PyEval_GetBuiltins(), true).get()));
  // python globals (fill with current parameters)
  for(map<string, shared_ptr<void> >::const_iterator i=currentParam.begin(); i!=currentParam.end(); i++)
    cpy(PyDict_SetItemString(globals.get(), i->first.c_str(), C(i->second).get()));

  // evaluate as statement
  PyO locals=cpy(PyDict_New());
  cpy(PyRun_String(code.c_str(), Py_file_input, globals.get(), locals.get()));

  // get all locals and add to currentImport
  cpy(PyDict_Merge(C(currentImport).get(), locals.get(), true));
}

bool PyEval::valueIsOfType(const shared_ptr<void> &value, ValueType type) const {
  PyObject *v=C(value).get();
  switch(type) {
    case ScalarType: return PyFloat_Check(v) || PyLong_Check(v) || PyBool_Check(v);
    case VectorType: try { ::cast_vector_double(value, true); return true; } catch(...) { return false; }
    case MatrixType: try { ::cast_vector_vector_double(value, true); return true; } catch(...) { return false; }
    case StringType: try { ::cast_string(value, true); return true; } catch(...) { return false; }
    case SXFunctionType: return (v->ob_type && v->ob_type->tp_name==string("SXFunction") ? true : false);
  }
  throw DOMEvalException("Internal error: Unknwon ValueType.");
}

map<path, pair<path, bool> >& PyEval::requiredFiles() const {
  static map<path, pair<path, bool> > files;
  return files;
}

shared_ptr<void> PyEval::createSwigByTypeName(const string &typeName) const {
  return cpy(PyObject_CallObject(cpy(PyDict_GetItemString(cpy(PyModule_GetDict(C(casadiValue).get()), true).get(), typeName.c_str()), true).get(), NULL));
}

shared_ptr<void> PyEval::callFunction(const string &name, const vector<shared_ptr<void> >& args) const {
  static map<string, PyO> functionValue;
  pair<map<string, PyO>::iterator, bool> f=functionValue.insert(make_pair(name, PyO(static_cast<PyObject*>(NULL))));
  if(f.second)
    f.first->second=cpy(PyDict_GetItemString(cpy(PyModule_GetDict(mbxmlutils.get()), true).get(), name.c_str()), true);
  PyO pyargs=cpy(PyTuple_New(args.size()));
  int idx=0;
  for(vector<shared_ptr<void> >::const_iterator it=args.begin(); it!=args.end(); ++it, ++idx) {
    PyObject *v=C(*it).get();
    cpy(PyTuple_SetItem(pyargs.get(), idx, v));
    Py_INCREF(v); // PyTuple_SetItem steals a reference of the argument
  }
  return cpy(PyObject_CallObject(f.first->second.get(), pyargs.get()));
}

shared_ptr<void> PyEval::fullStringToValue(const string &str, const DOMElement *e) const {
  string strtrim=str;
  trim(strtrim);

  // check some common string to avoid time consiming evaluation
  // check true and false
  if(strtrim=="True") return cpy(PyBool_FromLong(1));
  if(strtrim=="False") return cpy(PyBool_FromLong(0));
  // check for integer and floating point values
  try { return cpy(PyLong_FromLong(lexical_cast<int>(strtrim))); } catch(const boost::bad_lexical_cast &) {}
  try { return cpy(PyFloat_FromDouble(lexical_cast<double>(strtrim))); } catch(const boost::bad_lexical_cast &) {}
  // no common string detected -> evaluate using python now

  // restore current dir on exit and change current dir
  PreserveCurrentDir preserveDir;
  if(e) {
    path chdir=E(e)->getOriginalFilename().parent_path();
    if(!chdir.empty())
      current_path(chdir);
  }

  // python globals (fill with builtins)
  PyO globals=cpy(PyDict_New());
  cpy(PyDict_SetItemString(globals.get(), "__builtins__", cpy(PyEval_GetBuiltins(), true).get()));
  // python globals (fill with imports)
  cpy(PyDict_Merge(globals.get(), C(currentImport).get(), true));
  // python globals (fill with current parameters)
  for(map<string, shared_ptr<void> >::const_iterator i=currentParam.begin(); i!=currentParam.end(); i++)
    cpy(PyDict_SetItemString(globals.get(), i->first.c_str(), C(i->second).get()));

  PyO ret;
  PyCompilerFlags flags;
  flags.cf_flags=CO_FUTURE_DIVISION; // we evaluate the code in python 3 mode (future python 2 mode)
  try {
    // evaluate as expression (using the trimmed str) and save result in ret
    PyO locals=cpy(PyDict_New());
    ret=cpy(PyRun_StringFlags(strtrim.c_str(), Py_eval_input, globals.get(), locals.get(), &flags));
  }
  catch(const runtime_error&) { // on failure ...
    PyO locals=cpy(PyDict_New());
    try {
      // ... evaluate as statement

      // fix python indentation
      vector<string> lines;
      split(lines, str, is_any_of("\n")); // split to a vector of lines
      size_t indent=string::npos;
      size_t lineNr=0;
      for(vector<string>::iterator it=lines.begin(); it!=lines.end(); ++it, ++lineNr) {
        size_t pos=it->find_first_not_of(' '); // get first none space character
        if(pos==string::npos) continue; // not found -> pure empty line -> do not modify
        if(pos!=string::npos && (*it)[pos]=='#') continue; // found and first char is '#' -> pure comment line -> do not modify
        // now we have a line with a python statement
        if(indent==string::npos) indent=pos; // at the first python statement line use the current indent as indent for all others
        if(it->substr(0, indent)!=string(indent, ' ')) // check if line starts with at least indent spaces ...
          // ... if not its an indentation error
          throw DOMEvalException("Unexpected indentation at line "+lexical_cast<string>(lineNr)+": "+str, e);
        *it=it->substr(indent); // remove the first indent spaces from the line
      }
      strtrim=join(lines, "\n"); // join the lines to a single string

      // evaluate as statement
      cpy(PyRun_StringFlags(strtrim.c_str(), Py_file_input, globals.get(), locals.get(), &flags));
    }
    catch(const runtime_error& ex) { // on failure -> report error
      throw DOMEvalException(string(ex.what())+"Unable to evaluate expression:\n"+str, e);
    }
    try {
      // get 'ret' variable from statement
      ret=cpy(PyDict_GetItemString(locals.get(), "ret"), true);
    }
    catch(const runtime_error&) {
      // 'ret' variable not found or invalid expression
      throw DOMEvalException("Invalid expression or statement does not define the 'ret' variable in expression:\nX"+str, e);
    }
  }
  // convert a list or list of lists to a numpy array
  if(PyList_Check(ret.get())) {
    static PyO asarray=cpy(PyObject_GetAttrString(numpy.get(), "asarray"));
    PyO args=cpy(PyTuple_New(1));
    cpy(PyTuple_SetItem(args.get(), 0, ret.get()));
    Py_INCREF(ret.get()); // PyTuple_SetItem steals a reference of the argument
    return cpy(PyObject_CallObject(asarray.get(), args.get()));
  }
  // return result
  return ret;
}

string PyEval::getSwigType(const shared_ptr<void> &value) const {
  return C(value)->ob_type->tp_name;
}

double PyEval::cast_double(const shared_ptr<void> &value) const {
  PyObject *v=C(value).get();
  if(PyFloat_Check(v))
    return cpy(PyFloat_AsDouble(v));
  if(PyLong_Check(v))
    return cpy(PyLong_AsLong(v));
  throw runtime_error("Cannot cast this value to double.");
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
  try { return cpy(PyLong_FromLong(lexical_cast<int>(v))); } catch(const boost::bad_lexical_cast &) {}
  return cpy(PyFloat_FromDouble(v));
}

shared_ptr<void> PyEval::create_vector_double(const vector<double>& v) const {
  npy_intp dims[1];
  dims[0]=v.size();
  PyO ret=cpy(PyArray_SimpleNew(1, dims, NPY_DOUBLE));
  copy(v.begin(), v.end(), static_cast<double*>(PyArray_GETPTR1(reinterpret_cast<PyArrayObject*>(ret.get()), 0)));
  return ret;
}

shared_ptr<void> PyEval::create_vector_vector_double(const vector<vector<double> >& v) const {
  npy_intp dims[2];
  dims[0]=v.size();
  dims[1]=v[0].size();
  PyO ret=cpy(PyArray_SimpleNew(2, dims, NPY_DOUBLE));
  int r=0;
  for(vector<vector<double> >::const_iterator it=v.begin(); it!=v.end(); ++it, ++r)
    copy(it->begin(), it->end(), static_cast<double*>(PyArray_GETPTR2(reinterpret_cast<PyArrayObject*>(ret.get()), r, 0)));
  return ret;
}

shared_ptr<void> PyEval::create_string(const string& v) const {
  return cpy(PyUnicode_FromString(v.c_str()));
}

}

namespace {

void checkNumPyDoubleType(int type) {
  if(type!=NPY_SHORT    && type!=NPY_USHORT    &&
     type!=NPY_INT      && type!=NPY_UINT      &&
     type!=NPY_LONG     && type!=NPY_ULONG     &&
     type!=NPY_LONGLONG && type!=NPY_ULONGLONG &&
     type!=NPY_FLOAT    && type!=NPY_DOUBLE    && type!=NPY_LONGDOUBLE)
    throw runtime_error("Value is not of type double.");
}

double arrayGetDouble(PyArrayObject *a, int type, int r, int c=-1) {
  switch(type) {
    case NPY_SHORT:      return *static_cast<npy_short*>     (c==-1 ? PyArray_GETPTR1(a, r) : PyArray_GETPTR2(a, r, c));
    case NPY_USHORT:     return *static_cast<npy_ushort*>    (c==-1 ? PyArray_GETPTR1(a, r) : PyArray_GETPTR2(a, r, c));
    case NPY_INT:        return *static_cast<npy_int*>       (c==-1 ? PyArray_GETPTR1(a, r) : PyArray_GETPTR2(a, r, c));
    case NPY_UINT:       return *static_cast<npy_uint*>      (c==-1 ? PyArray_GETPTR1(a, r) : PyArray_GETPTR2(a, r, c));
    case NPY_LONG:       return *static_cast<npy_long*>      (c==-1 ? PyArray_GETPTR1(a, r) : PyArray_GETPTR2(a, r, c));
    case NPY_ULONG:      return *static_cast<npy_ulong*>     (c==-1 ? PyArray_GETPTR1(a, r) : PyArray_GETPTR2(a, r, c));
    case NPY_LONGLONG:   return *static_cast<npy_longlong*>  (c==-1 ? PyArray_GETPTR1(a, r) : PyArray_GETPTR2(a, r, c));
    case NPY_ULONGLONG:  return *static_cast<npy_ulonglong*> (c==-1 ? PyArray_GETPTR1(a, r) : PyArray_GETPTR2(a, r, c));
    case NPY_FLOAT:      return *static_cast<npy_float*>     (c==-1 ? PyArray_GETPTR1(a, r) : PyArray_GETPTR2(a, r, c));
    case NPY_DOUBLE:     return *static_cast<npy_double*>    (c==-1 ? PyArray_GETPTR1(a, r) : PyArray_GETPTR2(a, r, c));
    case NPY_LONGDOUBLE: return *static_cast<npy_longdouble*>(c==-1 ? PyArray_GETPTR1(a, r) : PyArray_GETPTR2(a, r, c));
  }
  throw runtime_error("Value is not of type double (wrong element type).");
}

vector<double> cast_vector_double(const shared_ptr<void> &value, bool checkOnly) {
  PyObject *v=C(value).get();
  if(PyArray_Check(v)) {
    PyArrayObject *a=reinterpret_cast<PyArrayObject*>(v);
    if(PyArray_NDIM(a)!=1)
      throw runtime_error("Value is not of type vector (wrong dimension).");
    int type=PyArray_TYPE(a);
    checkNumPyDoubleType(type);
    if(checkOnly)
      return vector<double>();
    npy_intp *dims=PyArray_SHAPE(a);
    vector<double> ret(dims[0]);
    for(size_t i=0; i<dims[0]; ++i)
      ret[i]=arrayGetDouble(a, type, i);
    return ret;
  }
  throw runtime_error("Value is not of type vector (wrong type).");
}

vector<vector<double> > cast_vector_vector_double(const shared_ptr<void> &value, bool checkOnly) {
  PyObject *v=C(value).get();
  if(PyArray_Check(v)) {
    PyArrayObject *a=reinterpret_cast<PyArrayObject*>(v);
    if(PyArray_NDIM(a)!=2)
      throw runtime_error("Value is not of type matrix (wrong dimension).");
    int type=PyArray_TYPE(a);
    checkNumPyDoubleType(type);
    if(checkOnly)
      return vector<vector<double> >();
    npy_intp *dims=PyArray_SHAPE(a);
    vector<vector<double> > ret(dims[0], vector<double>(dims[1]));
    for(size_t r=0; r<dims[0]; ++r)
      for(size_t c=0; c<dims[1]; ++c)
        ret[r][c]=arrayGetDouble(a, type, r, c);
    return ret;
  }
  throw runtime_error("Value is not of type matrix (wrong type).");
}

string cast_string(const shared_ptr<void> &value, bool checkOnly) {
  return cpy(PyUnicode_AsUTF8(C(value).get()));
}


}

namespace MBXMLUtils {

// We include swigpyrun.h at the end here to avoid the usage of functions macros
// defined in this file which we do not want to use.

#include "swigpyrun.h"

void* PyEval::getSwigThis(const shared_ptr<void> &value) const {
  return SWIG_Python_GetSwigThis(C(value).get())->ptr;
}

}
